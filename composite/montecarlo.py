"""Stage 3 - Monte Carlo equilibration on the GPU.

Moves
  * Hybrid Monte Carlo (Duane et al., Phys. Lett. B 195, 216 (1987)): all polymer
    atoms are moved together along a short, randomised-length leapfrog trajectory
    and the whole trajectory is accepted or rejected with the Metropolis rule.
    Trajectories must be long (a few hundred steps): the momenta are redrawn at
    every move, so short trajectories turn collective relaxation into a slow
    random walk and the density does not equilibrate.
    Every evaluation is a full-system force call, which is what the GPU is good at.
  * Volume moves in ln V with rigid scaling of the chain centres of mass (NPT).

A stage with `converge = true` runs until density and energy show no trend over
`converge_window` moves (linear drift below one standard deviation of the scatter
and a density change below 0.2 %), with `moves` as the upper limit.

Stage types: "nvt", "npt", and "ramp" - a preparation stage that shrinks the box
by a fixed factor before every hybrid-MC move to bring the loosely built cell to
its target density (not an equilibrium ensemble; the NPT stages that follow are).

The nanoparticle is a rigid body fixed at the box centre.  Step sizes are tuned
only in stages flagged `tune = true`; production stages keep them fixed.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import cupy as cp
import numpy as np

from . import forcefield as ff
from .analysis import drift
from .gpu_engine import Engine
from .io import System, write_system_xyz

LOG_FIELDS = (["stage", "move", "T_K", "P_atm", "E_total"]
              + [f"E_{t}" for t in Engine.TERMS]
              + ["box_A", "density_gcc", "acc_hmc", "acc_vol", "dt_fs", "dlnV", "moves_per_s"])


class MonteCarlo:
    def __init__(self, system: System, cfg, outdir, expected_density=0.0):
        self.cfg, self.mc = cfg, cfg["mc"]
        self.expected_density = expected_density
        self.system = system
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        f = cfg["forcefield"]
        self.engine = Engine(system, f["cutoff"], f["skin"], f["dsf_alpha"], f["lj_tail"],
                             self.mc["device"], self.mc["precision"],
                             f.get("cross_epsilon_scale", 1.0), f.get("ester_variant"))
        eng = self.engine
        self.total_mass = float(system.mass.sum())
        self.n_mol = int(system.mol.max())                       # mobile molecules
        self.mol_mass = cp.bincount(eng.mol, weights=eng.mass, minlength=self.n_mol + 1)[:, None]
        hmc_mass = float(self.mc["hmc_mass"])
        # HMC samples the same configurational distribution for any choice of masses
        self.m = (cp.full((eng.n, 1), hmc_mass) if hmc_mass > 0 else eng.mass[:, None].copy())
        self.dt = float(self.mc["dt_fs"]) / ff.TIME_UNIT_FS
        self.dlnv = float(self.mc["dlnV"])
        self.stage_index, self.move_index = 0, 0
        self.pe = eng.compute()

    # ------------------------------------------------------------------ moves
    def minimise(self, steps, max_step=0.2, f_tol=10.0):
        """Steepest descent with an adaptive step: removes the build-stage contacts."""
        eng = self.engine
        h, pe, fmax = 0.01, self.pe, float("nan")
        for it in range(steps):
            fmax = float(cp.sqrt(cp.max(cp.sum(eng.f * eng.f, axis=1))))
            if it % 100 == 0:
                print(f"    minimise {it:5d}  E = {pe:14.2f}  Fmax = {fmax:12.2f}")
            if fmax < f_tol:
                break
            snap = eng.snapshot()
            eng.pos += (h / fmax) * eng.f
            new = eng.compute()
            if math.isfinite(new) and new < pe:
                pe, h = new, min(1.2 * h, max_step)
            else:
                eng.restore(snap)
                h *= 0.5
        self.pe = pe
        print(f"    minimised: E = {pe:.2f} kcal/mol, Fmax = {fmax:.2f} kcal/mol/A")

    def hmc_move(self, kt):
        eng, m, dt = self.engine, self.m, self.dt
        mean = int(self.mc["hmc_steps"])
        n_steps = int(self.rng.integers(max(1, int(0.7 * mean)), int(1.3 * mean) + 1))
        v = cp.sqrt(kt / m) * cp.random.standard_normal((eng.n, 3)) * eng.mobile_f
        h0 = self.pe + 0.5 * float(cp.sum(m * v * v))
        snap = eng.snapshot()
        pe = self.pe
        v += (0.5 * dt) * eng.f / m
        for step in range(n_steps):
            eng.pos += dt * v
            pe = eng.compute()
            v += ((1.0 if step < n_steps - 1 else 0.5) * dt) * eng.f / m
        dh = pe + 0.5 * float(cp.sum(m * v * v)) - h0
        if math.isfinite(dh) and (dh <= 0.0 or self.rng.random() < math.exp(-dh / kt)):
            self.pe = pe
            return True
        eng.restore(snap)
        return False

    def _scale_box(self, box1):
        """Resize the box, moving every chain rigidly with its centre of mass."""
        eng = self.engine
        com = cp.stack([cp.bincount(eng.mol, weights=eng.mass * eng.pos[:, d],
                                    minlength=self.n_mol + 1) for d in range(3)], axis=1)
        com /= self.mol_mass
        eng.pos = eng.pos + (box1 / eng.box - 1.0) * com[eng.mol] * eng.mobile_f
        eng.box = box1
        eng.detach_nlist()
        return eng.compute()

    def ramp_box(self, target_box, moves_left):
        """Deterministic preparation step: one increment of a geometric box ramp."""
        eng = self.engine
        self.pe = self._scale_box(eng.box * (target_box / eng.box) ** (1.0 / max(1, moves_left)))

    def volume_move(self, kt, pressure):
        eng = self.engine
        v0 = eng.box ** 3
        v1 = v0 * math.exp(self.dlnv * (2.0 * self.rng.random() - 1.0))
        box1 = v1 ** (1.0 / 3.0)
        if box1 < 2.0 * (eng.rc + eng.skin):
            return False
        snap = eng.snapshot()
        pe = self._scale_box(box1)
        arg = (-(pe - self.pe + pressure * ff.ATM * (v1 - v0)) / kt
               + (self.n_mol + 1) * math.log(v1 / v0))
        if math.isfinite(arg) and (arg >= 0.0 or self.rng.random() < math.exp(arg)):
            self.pe = pe
            return True
        eng.restore(snap)
        return False

    # ------------------------------------------------------------------ bookkeeping
    @property
    def density(self):
        return self.total_mass / self.engine.box ** 3 * ff.AMU_A3_TO_GCC

    def current_system(self):
        s = self.system
        return System(self.engine.box, cp.asnumpy(self.engine.pos), s.type, s.mol, s.mobile,
                      s.bonds, s.angles, s.dihedrals, s.impropers, s.cmaps)

    def checkpoint(self):
        tmp = self.outdir / "checkpoint.tmp.npz"
        np.savez(tmp, pos=cp.asnumpy(self.engine.pos), box=self.engine.box,
                 stage=self.stage_index, move=self.move_index, dt=self.dt, dlnv=self.dlnv)
        tmp.replace(self.outdir / "checkpoint.npz")

    def load_checkpoint(self, from_stage=None):
        path = self.outdir / "checkpoint.npz"
        if not path.exists():
            if from_stage:
                raise FileNotFoundError(f"no checkpoint in {self.outdir} to continue from")
            return False
        with np.load(path) as ck:
            self.engine.pos = cp.asarray(ck["pos"])
            self.engine.box = float(ck["box"])
            self.stage_index, self.move_index = int(ck["stage"]), int(ck["move"])
            self.dt, self.dlnv = float(ck["dt"]), float(ck["dlnv"])
        if from_stage:                 # keep the configuration, restart the schedule at a stage
            names = [st["name"] for st in self.mc["stage"]]
            if from_stage not in names:
                raise ValueError(f"unknown stage '{from_stage}'; stages are: {', '.join(names)}")
            self.stage_index, self.move_index = names.index(from_stage), 0
        self.engine.detach_nlist()
        self.pe = self.engine.compute()
        print(f"  resumed from stage {self.stage_index}, move {self.move_index}")
        return True

    def _seed(self):
        seed = (self.cfg["system"]["seed"] + 104729 * int(self.cfg["system"].get("replica", 1))
                + 7919 * self.stage_index + self.move_index)
        self.rng = np.random.default_rng(seed)
        cp.random.seed(seed)

    # ------------------------------------------------------------------ driver
    def run(self, resume=False, from_stage=None):
        mc, eng = self.mc, self.engine
        stages = mc["stage"]
        resumed = (resume or bool(from_stage)) and self.load_checkpoint(from_stage)
        converged = {}
        log_path = self.outdir / "log.csv"
        traj_path = self.outdir / "trajectory.xyz"
        if not resumed:
            traj_path.unlink(missing_ok=True)
            print(f"  initial energy {self.pe:.2f} kcal/mol, density {self.density:.3f} g/cm3")
            self.minimise(int(mc["minimise_steps"]))
        log_file = open(log_path, "a" if resumed else "w", newline="")
        log = csv.writer(log_file)
        if not resumed:
            log.writerow(LOG_FIELDS)

        while self.stage_index < len(stages):
            st = stages[self.stage_index]
            n_moves = int(st["moves"])
            ensemble = st["ensemble"].lower()
            if ensemble not in ("nvt", "npt", "ramp"):
                raise ValueError(f"stage '{st['name']}': ensemble must be nvt, npt or ramp")
            npt, ramp = ensemble == "npt", ensemble == "ramp"
            if ramp:
                target_density = float(st["target_density_fraction"]) * self.expected_density
                target_box = (self.total_mass * ff.AMU_A3_TO_GCC / target_density) ** (1.0 / 3.0)
            t0, t1 = float(st["T"]), float(st.get("T_end", st["T"]))
            pressure = float(st.get("P", 1.0))
            tune = bool(st.get("tune", True))
            stop_density = float(st.get("stop_density_fraction", 0.0)) * self.expected_density
            converge = bool(st.get("converge", False))
            window = max(8, int(st.get("converge_window", 400)) // int(mc["log_every"]))
            history = []                              # (density, energy) per log record
            print(f"\n  stage {self.stage_index + 1}/{len(stages)} '{st['name']}': "
                  f"{st['ensemble'].upper()}, {n_moves} moves, T = {t0:g}"
                  + (f" -> {t1:g}" if t1 != t0 else "") + " K"
                  + (f", P = {pressure:g} atm" if npt else "")
                  + (f", box ramp to {target_density:.3f} g/cm3" if ramp else "")
                  + ("" if tune else ", fixed step sizes")
                  + (f"; ends when density and energy show no trend over "
                     f"{window * int(mc['log_every'])} moves" if converge else ""))
            self._seed()
            acc = {"hmc": [0, 0], "vol": [0, 0]}      # window counters [accepted, tried]
            clock, clock_move = time.time(), self.move_index
            while self.move_index < n_moves:
                i = self.move_index
                temp = t0 + (t1 - t0) * i / max(1, n_moves - 1)
                kt = ff.KB * temp
                if ramp:
                    self.ramp_box(target_box, n_moves - i)
                acc["hmc"][0] += self.hmc_move(kt)
                acc["hmc"][1] += 1
                if npt:
                    for _ in range(int(st.get("volume_moves", 1))):
                        acc["vol"][0] += self.volume_move(kt, pressure)
                        acc["vol"][1] += 1
                self.move_index = i + 1

                if self.move_index % int(mc["log_every"]) == 0 or self.move_index == n_moves:
                    a_h = acc["hmc"][0] / max(1, acc["hmc"][1])
                    a_v = acc["vol"][0] / max(1, acc["vol"][1])
                    rate = (self.move_index - clock_move) / max(1e-9, time.time() - clock)
                    clock, clock_move = time.time(), self.move_index
                    t = eng.terms
                    log.writerow([st["name"], self.move_index, f"{temp:.2f}", pressure if npt else "",
                                  f"{self.pe:.4f}", *(f"{t[k]:.4f}" for k in Engine.TERMS),
                                  f"{eng.box:.5f}", f"{self.density:.5f}", f"{a_h:.3f}",
                                  f"{a_v:.3f}" if npt else "", f"{self.dt * ff.TIME_UNIT_FS:.4f}",
                                  f"{self.dlnv:.6f}", f"{rate:.3f}"])
                    log_file.flush()
                    print(f"    {self.move_index:7d}  T {temp:6.1f}  E {self.pe:13.2f}  "
                          f"rho {self.density:.4f}  box {eng.box:7.3f}  acc {a_h:.2f}"
                          + (f"/{a_v:.2f}" if npt else "")
                          + f"  dt {self.dt * ff.TIME_UNIT_FS:.2f} fs  {rate:.2f} moves/s")
                    if tune:
                        target = float(mc["target_acceptance"])
                        if a_h > target + 0.1:
                            self.dt = min(self.dt * 1.1, 4.0 / ff.TIME_UNIT_FS)
                        elif a_h < target - 0.1:
                            self.dt *= 0.5 if a_h < 0.2 else 0.85
                        if npt and a_v > 0.5:
                            self.dlnv = min(self.dlnv * 1.2, 0.05)
                        elif npt and a_v < 0.3:
                            self.dlnv *= 0.8
                    acc = {"hmc": [0, 0], "vol": [0, 0]}
                    history.append((self.density, self.pe))
                    if converge and len(history) >= window:
                        rho, energy = np.array(history[-window:]).T
                        d_rho, d_e = drift(rho), drift(energy)
                        change = abs(np.polyfit(np.linspace(0, 1, window), rho, 1)[0]) / rho.mean()
                        print(f"             convergence over the last {window * int(mc['log_every'])} "
                              f"moves: density drift {d_rho:+.2f} sigma ({100 * change:.2f} %), "
                              f"energy drift {d_e:+.2f} sigma")
                        if (abs(d_rho) < float(st.get("converge_sigma", 1.0))
                                and abs(d_e) < float(st.get("converge_sigma", 1.0))
                                and change < float(st.get("converge_density_change", 0.002))):
                            converged[st["name"]] = self.move_index
                            print(f"    converged after {self.move_index} moves: stage complete")
                            self.checkpoint()
                            break
                if self.move_index % int(mc["trajectory_every"]) == 0:
                    write_system_xyz(traj_path, self.system, cp.asnumpy(eng.pos), eng.box,
                                     f'stage={st["name"]} move={self.move_index}', append=True,
                                     wrap=self.cfg["system"].get("xyz_wrap", "atom"))
                if self.move_index % int(mc["checkpoint_every"]) == 0:
                    self.checkpoint()
                if stop_density and self.density >= stop_density:
                    print(f"    density {self.density:.4f} g/cm3 reached after "
                          f"{self.move_index} moves: stage complete")
                    break
            if converge and st["name"] not in converged:
                converged[st["name"]] = None
                print(f"    WARNING: stage '{st['name']}' reached its limit of {n_moves} moves "
                      f"without meeting the convergence criterion")
            self.stage_index += 1
            self.move_index = 0
            self.checkpoint()
        log_file.close()

        final = self.current_system()
        final.save(self.outdir / "system_final.npz")
        write_system_xyz(self.outdir / "composite_equilibrated.xyz", final,
                         info=f"density={final.density:.4f}",
                         wrap=self.cfg["system"].get("xyz_wrap", "atom"))
        summary = {"box_A": final.box, "density_gcc": final.density,
                   "energy_kcal_mol": self.pe, "energy_terms": eng.terms,
                   "glass_weight_percent": 100.0 * final.glass_weight_fraction,
                   "converged_after_moves": converged,
                   "neighbour_list_builds": eng.n_builds}
        (self.outdir / "equilibration.json").write_text(json.dumps(summary, indent=2))
        print(f"\n  final: box {final.box:.3f} A, density {final.density:.4f} g/cm3, "
              f"E = {self.pe:.2f} kcal/mol")
        return summary


def equilibrate(cfg, build_dir, outdir, resume=False, from_stage=None):
    system = System.load(Path(build_dir) / "system.npz")
    start = cfg["mc"].get("start_from", "")
    if start and not resume and not from_stage:
        # continue from an equilibrated configuration of the same system (sensitivity runs)
        ref = System.load(Path(cfg["system"]["root"]) / start)
        if ref.n != system.n or not np.array_equal(ref.type, system.type):
            raise ValueError(f"mc.start_from: {start} is not a configuration of this system")
        system.pos, system.box = ref.pos, ref.box
        cfg["mc"]["minimise_steps"] = 0
        print(f"  starting from {start} (box {ref.box:.3f} A, density {ref.density:.4f} g/cm3)")
    built = json.loads((Path(build_dir) / "composite.json").read_text())
    expected = system.density * (built["box_A"] / built["expected_equilibrium_box_A"]) ** 3
    print(f"  {system.n} atoms; composite density expected near {expected:.3f} g/cm3 "
          f"(PCL at {cfg['composite']['polymer_density']} g/cm3)")
    return MonteCarlo(system, cfg, outdir, expected).run(resume, from_stage)
