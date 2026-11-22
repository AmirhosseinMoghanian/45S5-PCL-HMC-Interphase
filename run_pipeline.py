"""45S5 nanoparticle / PCL composite pipeline.

    python run_pipeline.py                     # all stages
    python run_pipeline.py cut build           # selected stages
    python run_pipeline.py equilibrate --resume
    python run_pipeline.py equilibrate analyse --from-stage equilibrate
    python run_pipeline.py --config my.toml --set nanoparticle.radius=15 --set mc.stage.1.moves=8000
    python run_pipeline.py --profile sensitivity_eps0.8  # config.toml + profiles/sensitivity_eps0.8.toml

Stages:  cut -> build -> equilibrate -> analyse -> structure
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path

STAGES = ("cut", "build", "equilibrate", "analyse", "structure")


def merge(base, extra):
    """Recursive dictionary update; lists (such as mc.stage) are replaced as a whole."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path, overrides, profiles=()):
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    for name in profiles:
        p = Path(name)
        if not p.is_file():
            p = Path(path).resolve().parent / "profiles" / f"{name}.toml"
        merge(cfg, tomllib.loads(p.read_text(encoding="utf-8")))
    for item in overrides:
        key, value = item.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node[int(p)] if isinstance(node, list) else node[p]
        node[leaf] = tomllib.loads(f"v = {value}")["v"]
    root = Path(path).resolve().parent
    cfg["system"]["root"] = str(root)
    for key in ("bulk_glass", "output_dir"):
        cfg["system"][key] = str((root / cfg["system"][key]).resolve())
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="*", default=[], metavar="stage",
                        help=f"any of: {', '.join(STAGES)} (default: all)")
    parser.add_argument("--config", default=Path(__file__).with_name("config.toml"))
    parser.add_argument("--profile", action="append", default=[], metavar="NAME",
                        help="merge profiles/NAME.toml over the config (repeatable)")
    parser.add_argument("--set", action="append", default=[], metavar="section.key=value",
                        help="override a config value (TOML syntax)")
    parser.add_argument("--resume", action="store_true",
                        help="continue the equilibration from its last checkpoint")
    parser.add_argument("--from-stage", metavar="NAME",
                        help="continue from the last checkpoint, restarting the schedule at this "
                             "Monte Carlo stage (e.g. equilibrate)")
    args = parser.parse_args()
    unknown = set(args.stages) - set(STAGES)
    if unknown:
        parser.error(f"unknown stage(s): {', '.join(sorted(unknown))}")
    stages = [s for s in STAGES if s in args.stages] or list(STAGES)

    cfg = load_config(args.config, args.set, args.profile)
    out = Path(cfg["system"]["output_dir"])
    dirs = {"cut": out / "01_nanoparticle", "build": out / "02_composite",
            "equilibrate": out / "03_equilibration", "analyse": out / "04_analysis",
            "structure": out / "05_structure"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_used.json").write_text(json.dumps(cfg, indent=2))      # after profiles and --set

    for stage in stages:
        print(f"\n=== {stage.upper()} " + "=" * (66 - len(stage)))
        start = time.time()
        if stage == "cut":
            from composite.nanoparticle import cut_nanoparticle
            cut_nanoparticle(cfg, dirs["cut"])
        elif stage == "build":
            from composite.packing import build_composite
            build_composite(cfg, dirs["cut"], dirs["build"])
        elif stage == "equilibrate":
            from composite.montecarlo import equilibrate
            equilibrate(cfg, dirs["build"], dirs["equilibrate"], args.resume, args.from_stage)
        elif stage == "analyse":
            from composite.analysis import analyse
            analyse(cfg, dirs["equilibrate"], dirs["analyse"])
        elif stage == "structure":
            from composite.structure import analyse_structure
            analyse_structure(cfg, dirs["equilibrate"], dirs["cut"], dirs["structure"])
        print(f"  [{stage} finished in {time.time() - start:.1f} s] -> {dirs[stage]}")


if __name__ == "__main__":
    main()
