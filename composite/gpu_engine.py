"""GPU energy / force engine (CuPy + hand-written CUDA kernels).

Precision  : "mixed"  - pair terms in single, positions / accumulation in double
             "double" - everything in double (validation).

Non-bonded : Lennard-Jones truncated at the cutoff (plus a long-range dispersion
             correction) and damped-shifted-force Coulomb (Fennell & Gezelter,
             J. Chem. Phys. 124, 234104 (2006)), evaluated from a cell-list based
             Verlet neighbour list.  Excluded and scaled intramolecular pairs follow
             the LAMMPS lj/cut/coul/dsf convention, against which the energies
             have been checked term by term.
Bonded     : harmonic bonds and angles, OPLS Fourier torsions.

Frozen atoms (the nanoparticle) act on the mobile atoms but receive no force
and contribute no frozen-frozen energy.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

# conda keeps the CUDA runtime DLLs (nvrtc, ...) in <env>/Library/bin
_dll_dir = Path(sys.prefix) / "Library" / "bin"
if os.name == "nt" and _dll_dir.is_dir():
    os.environ["PATH"] = f"{_dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    os.add_dll_directory(str(_dll_dir))

import cupy as cp          # noqa: E402
import numpy as np         # noqa: E402

from . import forcefield as ff          # noqa: E402
from .polymer import exclusions         # noqa: E402

FLAG_14 = 0x40000000
BLOCK = 128

_CUDA = r"""
#define FLAG_14 0x40000000
#define IDX_MASK 0x3FFFFFFF
typedef REAL real;

extern "C" __global__
void build_nlist(const real* pos, const int* cell, const int* order, const int* cell_start,
                 const int* mobile, const int* mol, const int* excl, const int* p14,
                 int n, int nc, int nex, int n14, int maxn, double box_, double rlist2_,
                 int* nl, int* nn)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (!mobile[i]) { nn[i] = 0; return; }
    const real xi = pos[3*i], yi = pos[3*i+1], zi = pos[3*i+2];
    const real box = (real)box_, inv = (real)(1.0 / box_), rlist2 = (real)rlist2_;
    const int ci = cell[i];
    const int cx = ci % nc, cy = (ci / nc) % nc, cz = ci / (nc * nc);
    const int span = (nc >= 3) ? 1 : 0;
    int count = 0;
    for (int dz = -span; dz <= span; ++dz)
    for (int dy = -span; dy <= span; ++dy)
    for (int dx = -span; dx <= span; ++dx) {
        const int c = (((cz + dz + nc) % nc) * nc + (cy + dy + nc) % nc) * nc + (cx + dx + nc) % nc;
        for (int s = cell_start[c]; s < cell_start[c + 1]; ++s) {
            const int j = order[s];
            if (j == i) continue;
            real rx = xi - pos[3*j], ry = yi - pos[3*j+1], rz = zi - pos[3*j+2];
            rx -= box * rint(rx * inv); ry -= box * rint(ry * inv); rz -= box * rint(rz * inv);
            if (rx*rx + ry*ry + rz*rz >= rlist2) continue;
            int entry = j;
            if (mol[j] == mol[i]) {
                bool skip = false;
                for (int k = 0; k < nex; ++k) {
                    const int e = excl[i*nex + k];
                    if (e < 0) break;
                    if (e == j) { skip = true; break; }
                }
                if (skip) continue;
                for (int k = 0; k < n14; ++k) {
                    const int e = p14[i*n14 + k];
                    if (e < 0) break;
                    if (e == j) { entry = j | FLAG_14; break; }
                }
            }
            if (count < maxn) nl[(long long)i * maxn + count] = entry;
            ++count;
        }
    }
    nn[i] = count;
}

extern "C" __global__
void nonbonded(const double* pos, const int* type, const double* q, const int* mobile,
               const int* nl, const int* nn, const double* lj_a, const double* lj_b,
               const double* lj_s, int n, int maxn, int ntype, double box, double rc_,
               double alpha_, double dsf_e_, double dsf_f_, double qqr2e, double scale14_lj,
               double scale14_coul, double* f, double* e_lj, double* e_coul)
{
    // pair arithmetic in `real` (float in mixed precision), accumulation in double
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double fx = 0.0, fy = 0.0, fz = 0.0, elj = 0.0, ecoul = 0.0;
    if (mobile[i]) {
        const double xi = pos[3*i], yi = pos[3*i+1], zi = pos[3*i+2];
        const double inv = 1.0 / box;
        const real rc = (real)rc_, rc2 = rc * rc, alpha = (real)alpha_;
        const real dsf_e = (real)dsf_e_, dsf_f = (real)dsf_f_;
        const real qi = (real)(qqr2e * q[i]);
        const real two_a_sqrtpi = (real)(1.1283791670955126 * alpha_);
        const real s14_lj = (real)scale14_lj, s14_c = (real)scale14_coul, one = (real)1.0;
        const int ti = type[i] * ntype;
        const int count = nn[i];
        for (int k = 0; k < count; ++k) {
            const int raw = nl[(long long)i * maxn + k];
            const int j = raw & IDX_MASK;
            double dx = xi - pos[3*j], dy = yi - pos[3*j+1], dz = zi - pos[3*j+2];
            dx -= box * rint(dx * inv); dy -= box * rint(dy * inv); dz -= box * rint(dz * inv);
            const real rx = (real)dx, ry = (real)dy, rz = (real)dz;
            const real r2 = rx*rx + ry*ry + rz*rz;
            if (r2 >= rc2) continue;
            const real s_lj = (raw & FLAG_14) ? s14_lj : one;
            const real s_c = (raw & FLAG_14) ? s14_c : one;
            const real w = mobile[j] ? (real)0.5 : one;      // mobile pairs are visited twice
            const real r = sqrt(r2), ir2 = one / r2, ir6 = ir2 * ir2 * ir2;
            const int tp = ti + type[j];
            const real a = (real)lj_a[tp], b = (real)lj_b[tp];
            real fr = s_lj * ((real)12.0 * a * ir6 - (real)6.0 * b) * ir6 * ir2;
            elj += w * s_lj * ((a * ir6 - b) * ir6 - (real)lj_s[tp]);
            const real qq = qi * (real)q[j];
            if (qq != (real)0.0) {
                const real ar = alpha * r;
                const real erfc_r = erfc(ar);
                // scaled 1-4 pairs: full damped term minus (1 - scale) of the bare Coulomb term
                const real bare = (one - s_c) * qq / r;
                ecoul += w * (qq * (erfc_r / r - dsf_e + dsf_f * (r - rc)) - bare);
                fr += (qq * (erfc_r * ir2 + two_a_sqrtpi * exp(-ar * ar) / r - dsf_f) / r
                       - bare * ir2);
            }
            fx += fr * rx; fy += fr * ry; fz += fr * rz;
        }
    }
    f[3*i] = fx; f[3*i+1] = fy; f[3*i+2] = fz;
    e_lj[i] = elj; e_coul[i] = ecoul;
}

extern "C" __global__
void excluded(const double* pos, const double* q, const int* idx, int n, double alpha,
              double dsf_e, double dsf_f, double rc, double qqr2e, double* f, double* e)
{
    // damped-shifted-force correction for excluded (1-2, 1-3) pairs: DSF(r) - q_i q_j / r
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int i = idx[2*t], j = idx[2*t+1];
    const double rx = pos[3*i] - pos[3*j], ry = pos[3*i+1] - pos[3*j+1], rz = pos[3*i+2] - pos[3*j+2];
    const double r2 = rx*rx + ry*ry + rz*rz, r = sqrt(r2), ar = alpha * r;
    const double qq = qqr2e * q[i] * q[j], erf_r = erf(ar);
    e[t] = qq * (-erf_r / r - dsf_e + dsf_f * (r - rc));
    const double fr = qq * (-erf_r / r2 + 1.1283791670955126 * alpha * exp(-ar * ar) / r - dsf_f) / r;
    atomicAdd(&f[3*i], fr*rx); atomicAdd(&f[3*i+1], fr*ry); atomicAdd(&f[3*i+2], fr*rz);
    atomicAdd(&f[3*j], -fr*rx); atomicAdd(&f[3*j+1], -fr*ry); atomicAdd(&f[3*j+2], -fr*rz);
}

__device__ double dihedral_grad(const double* pos, int i, int j, int k, int l, double* g)
{
    // IUPAC torsion angle and its gradient with respect to the four atoms (g[0..11])
    const double b1x = pos[3*j] - pos[3*i], b1y = pos[3*j+1] - pos[3*i+1], b1z = pos[3*j+2] - pos[3*i+2];
    const double b2x = pos[3*k] - pos[3*j], b2y = pos[3*k+1] - pos[3*j+1], b2z = pos[3*k+2] - pos[3*j+2];
    const double b3x = pos[3*l] - pos[3*k], b3y = pos[3*l+1] - pos[3*k+1], b3z = pos[3*l+2] - pos[3*k+2];
    const double n1x = b1y*b2z - b1z*b2y, n1y = b1z*b2x - b1x*b2z, n1z = b1x*b2y - b1y*b2x;
    const double n2x = b2y*b3z - b2z*b3y, n2y = b2z*b3x - b2x*b3z, n2z = b2x*b3y - b2y*b3x;
    const double n1sq = fmax(n1x*n1x + n1y*n1y + n1z*n1z, 1e-12);
    const double n2sq = fmax(n2x*n2x + n2y*n2y + n2z*n2z, 1e-12);
    const double b2sq = b2x*b2x + b2y*b2y + b2z*b2z, b2len = sqrt(b2sq);
    const double gi = -b2len / n1sq, gl = b2len / n2sq;
    const double p = (b1x*b2x + b1y*b2y + b1z*b2z) / b2sq, q = (b3x*b2x + b3y*b2y + b3z*b2z) / b2sq;
    g[0] = gi*n1x; g[1] = gi*n1y; g[2] = gi*n1z;
    g[9] = gl*n2x; g[10] = gl*n2y; g[11] = gl*n2z;
    for (int d = 0; d < 3; ++d) {
        g[3+d] = q*g[9+d] - (1.0 + p)*g[d];
        g[6+d] = p*g[d] - (1.0 + q)*g[9+d];
    }
    return atan2(b2len * (b1x*n2x + b1y*n2y + b1z*n2z), n1x*n2x + n1y*n2y + n1z*n2z);
}

extern "C" __global__
void cmap(const double* pos, const int* idx, const double* coef, int n, int ngrid, double dgrid,
          double* f, double* e)
{
    // PLAFF3 CMAP: bicubic patch interpolation of E(phi, psi) on a periodic grid
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int a0 = idx[5*t], a1 = idx[5*t+1], a2 = idx[5*t+2], a3 = idx[5*t+3], a4 = idx[5*t+4];
    double g1[12], g2[12];
    const double phi = dihedral_grad(pos, a0, a1, a2, a3, g1);
    const double psi = dihedral_grad(pos, a1, a2, a3, a4, g2);
    const double sp = (phi + 3.141592653589793) / dgrid, ss = (psi + 3.141592653589793) / dgrid;
    const double fp = floor(sp), fs = floor(ss);
    const double tt = sp - fp, uu = ss - fs;
    const int ip = (((int)fp % ngrid) + ngrid) % ngrid, js = (((int)fs % ngrid) + ngrid) % ngrid;
    const double* c = coef + (long long)(ip * ngrid + js) * 16;
    const double tp[4] = {1.0, tt, tt*tt, tt*tt*tt}, up[4] = {1.0, uu, uu*uu, uu*uu*uu};
    double en = 0.0, dt = 0.0, du = 0.0;
    for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
            const double cab = c[4*a + b];
            en += cab * tp[a] * up[b];
            if (a) dt += a * cab * tp[a-1] * up[b];
            if (b) du += b * cab * tp[a] * up[b-1];
        }
    e[t] = en;
    const double dphi = dt / dgrid, dpsi = du / dgrid;
    for (int d = 0; d < 3; ++d) {
        atomicAdd(&f[3*a0 + d], -dphi*g1[d]);
        atomicAdd(&f[3*a1 + d], -dphi*g1[3+d] - dpsi*g2[d]);
        atomicAdd(&f[3*a2 + d], -dphi*g1[6+d] - dpsi*g2[3+d]);
        atomicAdd(&f[3*a3 + d], -dphi*g1[9+d] - dpsi*g2[6+d]);
        atomicAdd(&f[3*a4 + d], -dpsi*g2[9+d]);
    }
}

extern "C" __global__
void bonds(const double* pos, const int* idx, const double* par, int n, double* f, double* e)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int i = idx[2*t], j = idx[2*t+1];
    const double rx = pos[3*i] - pos[3*j], ry = pos[3*i+1] - pos[3*j+1], rz = pos[3*i+2] - pos[3*j+2];
    const double r = sqrt(rx*rx + ry*ry + rz*rz);
    const double k = par[2*t], dr = r - par[2*t+1];
    e[t] = k * dr * dr;
    const double fr = -2.0 * k * dr / r;
    atomicAdd(&f[3*i], fr*rx); atomicAdd(&f[3*i+1], fr*ry); atomicAdd(&f[3*i+2], fr*rz);
    atomicAdd(&f[3*j], -fr*rx); atomicAdd(&f[3*j+1], -fr*ry); atomicAdd(&f[3*j+2], -fr*rz);
}

extern "C" __global__
void angles(const double* pos, const int* idx, const double* par, int n, double* f, double* e)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int i = idx[3*t], j = idx[3*t+1], k = idx[3*t+2];
    const double ax = pos[3*i] - pos[3*j], ay = pos[3*i+1] - pos[3*j+1], az = pos[3*i+2] - pos[3*j+2];
    const double bx = pos[3*k] - pos[3*j], by = pos[3*k+1] - pos[3*j+1], bz = pos[3*k+2] - pos[3*j+2];
    const double la = sqrt(ax*ax + ay*ay + az*az), lb = sqrt(bx*bx + by*by + bz*bz);
    double c = (ax*bx + ay*by + az*bz) / (la * lb);
    c = fmin(1.0, fmax(-1.0, c));
    const double s = fmax(sqrt(1.0 - c*c), 1e-8);
    const double dth = acos(c) - par[2*t+1];
    e[t] = par[2*t] * dth * dth;
    const double pre = 2.0 * par[2*t] * dth / s;           // dE/dtheta / sin(theta)
    const double iab = 1.0 / (la * lb), ca = c / (la * la), cb = c / (lb * lb);
    const double fix = pre * (bx*iab - ax*ca), fiy = pre * (by*iab - ay*ca), fiz = pre * (bz*iab - az*ca);
    const double fkx = pre * (ax*iab - bx*cb), fky = pre * (ay*iab - by*cb), fkz = pre * (az*iab - bz*cb);
    atomicAdd(&f[3*i], fix); atomicAdd(&f[3*i+1], fiy); atomicAdd(&f[3*i+2], fiz);
    atomicAdd(&f[3*k], fkx); atomicAdd(&f[3*k+1], fky); atomicAdd(&f[3*k+2], fkz);
    atomicAdd(&f[3*j], -fix-fkx); atomicAdd(&f[3*j+1], -fiy-fky); atomicAdd(&f[3*j+2], -fiz-fkz);
}

extern "C" __global__
void torsions(const double* pos, const int* idx, const double* par, int n, double* f, double* e)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int i = idx[4*t], j = idx[4*t+1], k = idx[4*t+2], l = idx[4*t+3];
    const double b1x = pos[3*j] - pos[3*i], b1y = pos[3*j+1] - pos[3*i+1], b1z = pos[3*j+2] - pos[3*i+2];
    const double b2x = pos[3*k] - pos[3*j], b2y = pos[3*k+1] - pos[3*j+1], b2z = pos[3*k+2] - pos[3*j+2];
    const double b3x = pos[3*l] - pos[3*k], b3y = pos[3*l+1] - pos[3*k+1], b3z = pos[3*l+2] - pos[3*k+2];
    const double n1x = b1y*b2z - b1z*b2y, n1y = b1z*b2x - b1x*b2z, n1z = b1x*b2y - b1y*b2x;
    const double n2x = b2y*b3z - b2z*b3y, n2y = b2z*b3x - b2x*b3z, n2z = b2x*b3y - b2y*b3x;
    const double n1sq = fmax(n1x*n1x + n1y*n1y + n1z*n1z, 1e-12);
    const double n2sq = fmax(n2x*n2x + n2y*n2y + n2z*n2z, 1e-12);
    const double b2sq = b2x*b2x + b2y*b2y + b2z*b2z, b2len = sqrt(b2sq);
    const double phi = atan2(b2len * (b1x*n2x + b1y*n2y + b1z*n2z), n1x*n2x + n1y*n2y + n1z*n2z);
    const double v1 = par[3*t], v2 = par[3*t+1], v3 = par[3*t+2];
    e[t] = 0.5 * (v1 * (1.0 + cos(phi)) + v2 * (1.0 - cos(2.0*phi)) + v3 * (1.0 + cos(3.0*phi)));
    const double de = 0.5 * (-v1 * sin(phi) + 2.0 * v2 * sin(2.0*phi) - 3.0 * v3 * sin(3.0*phi));
    // dphi/dr_i = -|b2| n1/|n1|^2 ,  dphi/dr_l = +|b2| n2/|n2|^2
    const double gi = -b2len / n1sq, gl = b2len / n2sq;
    const double gix = gi*n1x, giy = gi*n1y, giz = gi*n1z;
    const double glx = gl*n2x, gly = gl*n2y, glz = gl*n2z;
    const double p = (b1x*b2x + b1y*b2y + b1z*b2z) / b2sq, q = (b3x*b2x + b3y*b2y + b3z*b2z) / b2sq;
    const double gjx = q*glx - (1.0 + p)*gix, gjy = q*gly - (1.0 + p)*giy, gjz = q*glz - (1.0 + p)*giz;
    const double gkx = p*gix - (1.0 + q)*glx, gky = p*giy - (1.0 + q)*gly, gkz = p*giz - (1.0 + q)*glz;
    atomicAdd(&f[3*i], -de*gix); atomicAdd(&f[3*i+1], -de*giy); atomicAdd(&f[3*i+2], -de*giz);
    atomicAdd(&f[3*j], -de*gjx); atomicAdd(&f[3*j+1], -de*gjy); atomicAdd(&f[3*j+2], -de*gjz);
    atomicAdd(&f[3*k], -de*gkx); atomicAdd(&f[3*k+1], -de*gky); atomicAdd(&f[3*k+2], -de*gkz);
    atomicAdd(&f[3*l], -de*glx); atomicAdd(&f[3*l+1], -de*gly); atomicAdd(&f[3*l+2], -de*glz);
}
"""


def _grid(n):
    return ((n + BLOCK - 1) // BLOCK,), (BLOCK,)


class Engine:
    """Potential energy and forces of a composite System on the GPU."""

    TERMS = ("bond", "angle", "torsion", "cmap", "lj", "coul", "tail")

    def __init__(self, system, cutoff=10.0, skin=2.0, dsf_alpha=0.2, lj_tail=True, device=0,
                 precision="mixed", cross_epsilon_scale=1.0, variant=None):
        if precision not in ("mixed", "double"):
            raise ValueError("precision must be 'mixed' or 'double'")
        cp.cuda.Device(device).use()
        real = "float" if precision == "mixed" else "double"
        module = cp.RawModule(code=_CUDA, options=(f"-DREAL={real}",))
        self.real = cp.float32 if precision == "mixed" else cp.float64
        self._k = {name: module.get_function(name)
                   for name in ("build_nlist", "nonbonded", "excluded", "bonds", "angles",
                               "torsions", "cmap")}
        self.n = system.n
        self.box = float(system.box)
        self.rc, self.skin, self.alpha = float(cutoff), float(skin), float(dsf_alpha)
        erfc_rc = math.erfc(self.alpha * self.rc)
        self.dsf_e = erfc_rc / self.rc
        self.dsf_f = (erfc_rc / self.rc ** 2
                      + 2.0 * self.alpha / math.sqrt(math.pi)
                      * math.exp(-(self.alpha * self.rc) ** 2) / self.rc)

        self.pos = cp.asarray(system.pos, dtype=cp.float64)
        self.type = cp.asarray(system.type, dtype=cp.int32)
        self.charge = cp.asarray(system.charge, dtype=cp.float64)
        self.mobile = cp.asarray(system.mobile, dtype=cp.int32)
        self.mobile_f = self.mobile.astype(cp.float64)[:, None]
        self.mol = cp.asarray(system.mol, dtype=cp.int32)
        self.mass = cp.asarray(system.mass, dtype=cp.float64)

        bond_par, angle_par, tors, tors_par = ff.bonded_parameters(
            system.type, system.bonds, system.angles, system.dihedrals, system.impropers,
            **({"variant": variant} if variant else {}))
        excl, p14 = exclusions(system.n, system.bonds, system.angles, system.dihedrals)
        as_i = lambda a: cp.asarray(np.ascontiguousarray(a), dtype=cp.int32)
        as_f = lambda a: cp.asarray(np.ascontiguousarray(a), dtype=cp.float64)
        self.bond_idx, self.bond_par = as_i(system.bonds), as_f(bond_par)
        self.angle_idx, self.angle_par = as_i(system.angles), as_f(angle_par)
        self.tors_idx, self.tors_par = as_i(tors), as_f(tors_par)
        self.excl, self.p14 = as_i(excl), as_i(p14)
        self.e_bond = cp.zeros(len(system.bonds))
        self.e_angle = cp.zeros(len(system.angles))
        self.e_tors = cp.zeros(len(tors))
        self.cmap_idx = as_i(np.asarray(system.cmaps).reshape(-1, 5))
        self.e_cmap = cp.zeros(len(self.cmap_idx))
        if len(self.cmap_idx):
            coef, self.cmap_d = ff.cmap_coefficients()
            self.cmap_coef, self.cmap_n = as_f(coef), int(round(2 * math.pi / self.cmap_d))
        ii, kk = np.nonzero(excl >= 0)
        pairs = np.stack([ii, excl[ii, kk]], axis=1)
        self.excl_idx = as_i(pairs[pairs[:, 0] < pairs[:, 1]])
        self.e_excl = cp.zeros(len(self.excl_idx))

        a, b, s = ff.lj_tables(self.rc, cross_epsilon_scale)
        self.ntype = a.shape[0]
        self.lj_a, self.lj_b, self.lj_s = as_f(a), as_f(b), as_f(s)
        # homogeneous long-range dispersion correction: E_tail = -tail_coeff / V
        n_mob = np.bincount(system.type[system.mobile], minlength=self.ntype).astype(float)
        n_frz = np.bincount(system.type[~system.mobile], minlength=self.ntype).astype(float)
        pair_sum = 0.5 * n_mob @ b @ n_mob + n_mob @ b @ n_frz
        self.tail_coeff = 4.0 * math.pi / (3.0 * self.rc ** 3) * pair_sum if lj_tail else 0.0

        self.f = cp.zeros((self.n, 3))
        self.e_lj = cp.zeros(self.n)
        self.e_coul = cp.zeros(self.n)
        self.maxn = 0
        self.nl = self.nn = self.nl_ref = None
        self.nl_box = 0.0
        self.n_builds = 0

    # ------------------------------------------------------------------ neighbour list
    def build_nlist(self):
        rlist = self.rc + self.skin
        if self.box < 2.0 * rlist:
            raise RuntimeError(f"box ({self.box:.2f} A) fell below twice the neighbour-list "
                               f"radius ({rlist:.1f} A); the system is too small for this cutoff")
        nc = int(self.box // rlist)
        nc = nc if nc >= 3 else 1
        frac = self.pos / self.box + 0.5
        frac -= cp.floor(frac)
        wrapped = ((frac - 0.5) * self.box).astype(self.real)
        ijk = cp.minimum((frac * nc).astype(cp.int32), nc - 1)
        cell = (ijk[:, 2] * nc + ijk[:, 1]) * nc + ijk[:, 0]
        order = cp.argsort(cell).astype(cp.int32)
        cell_start = cp.searchsorted(cell[order], cp.arange(nc ** 3 + 1)).astype(cp.int32)
        if self.maxn == 0:        # first guess from the mean density, with head-room
            self.maxn = int(1.6 * self.n / self.box ** 3 * 4.19 * rlist ** 3) + 64
        while True:
            if self.nl is None or self.nl.shape[1] != self.maxn:
                self.nl = cp.zeros((self.n, self.maxn), dtype=cp.int32)
                self.nn = cp.zeros(self.n, dtype=cp.int32)
            self._k["build_nlist"](*_grid(self.n), (
                wrapped, cell, order, cell_start, self.mobile, self.mol, self.excl, self.p14,
                np.int32(self.n), np.int32(nc), np.int32(self.excl.shape[1]),
                np.int32(self.p14.shape[1]), np.int32(self.maxn), np.float64(self.box),
                np.float64(rlist * rlist), self.nl, self.nn))
            needed = int(self.nn.max())
            if needed <= self.maxn:
                break
            self.maxn = int(needed * 1.25) + 16
        self.nl_ref = self.pos.copy()
        self.nl_box = self.box
        self.n_builds += 1

    def nlist_valid(self):
        if self.nl_ref is None or self.nl_box != self.box:
            return False
        d = self.pos - self.nl_ref
        return float(cp.max(cp.sum(d * d, axis=1))) < (0.5 * self.skin) ** 2

    # ------------------------------------------------------------------ energy / forces
    def compute(self):
        """Total potential energy [kcal/mol]; forces are left in self.f."""
        if not self.nlist_valid():
            self.build_nlist()
        k = self._k
        k["nonbonded"](*_grid(self.n), (
            self.pos, self.type, self.charge, self.mobile, self.nl, self.nn,
            self.lj_a, self.lj_b, self.lj_s, np.int32(self.n), np.int32(self.maxn),
            np.int32(self.ntype), np.float64(self.box), np.float64(self.rc),
            np.float64(self.alpha), np.float64(self.dsf_e), np.float64(self.dsf_f),
            np.float64(ff.COULOMB), np.float64(ff.LJ14_SCALE), np.float64(ff.COUL14_SCALE),
            self.f, self.e_lj, self.e_coul))
        if len(self.e_excl):
            k["excluded"](*_grid(len(self.e_excl)), (
                self.pos, self.charge, self.excl_idx, np.int32(len(self.e_excl)),
                np.float64(self.alpha), np.float64(self.dsf_e), np.float64(self.dsf_f),
                np.float64(self.rc), np.float64(ff.COULOMB), self.f, self.e_excl))
        for name, idx, par, e in (("bonds", self.bond_idx, self.bond_par, self.e_bond),
                                  ("angles", self.angle_idx, self.angle_par, self.e_angle),
                                  ("torsions", self.tors_idx, self.tors_par, self.e_tors)):
            if len(e):
                k[name](*_grid(len(e)), (self.pos, idx, par, np.int32(len(e)), self.f, e))
        if len(self.e_cmap):
            k["cmap"](*_grid(len(self.e_cmap)), (
                self.pos, self.cmap_idx, self.cmap_coef, np.int32(len(self.e_cmap)),
                np.int32(self.cmap_n), np.float64(self.cmap_d), self.f, self.e_cmap))
        self.f *= self.mobile_f
        sums = cp.stack([self.e_bond.sum(), self.e_angle.sum(), self.e_tors.sum(),
                         self.e_cmap.sum(), self.e_lj.sum(),
                         self.e_coul.sum() + self.e_excl.sum()]).get()
        self.terms = dict(zip(self.TERMS, [*sums.tolist(), -self.tail_coeff / self.box ** 3]))
        return sum(self.terms.values())

    # ------------------------------------------------------------------ state handling
    def snapshot(self):
        """Everything needed to return to the current state after a rejected move."""
        return (self.pos.copy(), self.f.copy(), self.box, dict(self.terms), self.nl, self.nn,
                self.nl_ref, self.nl_box, self.maxn, self.n_builds)

    def restore(self, snap):
        self.pos, self.f, self.box, self.terms = snap[:4]
        nl, nn, nl_ref, nl_box, maxn, n_builds = snap[4:]
        if nl is self.nl and n_builds != self.n_builds:
            self.nl_ref = None            # list buffer was rebuilt in place: rebuild lazily
        else:
            self.nl, self.nn, self.nl_ref, self.nl_box, self.maxn = nl, nn, nl_ref, nl_box, maxn

    def detach_nlist(self):
        """Force the next build into fresh buffers so a snapshot keeps the current list."""
        self.nl = self.nn = self.nl_ref = None
