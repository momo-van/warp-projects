"""
JAX reference implementation of 1-D compressible Euler.

Identical numerical scheme to WarpEuler1D:
  - WENO3 reconstruction on primitive variables (Jiang & Shu 1996)
  - HLLC Riemann solver (Toro 2009)
  - SSP-RK2 time integration

Array layout: (nvars=3, N) — conserved variables [rho, rho*u, E].
Ghost cells are rebuilt each step via BC padding (no persistent ghost array).
"""

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

jax.config.update("jax_enable_x64", False)   # float32 to match Warp

NG    = 2
GAMMA = 1.4
EPS   = 1.0e-6


# ──────────────────────────── primitives ──────────────────────────────────────

def _to_prim(Q, gamma):
    rho = Q[0]
    u   = Q[1] / rho
    p   = (gamma - 1.0) * (Q[2] - 0.5 * rho * u * u)
    return rho, u, p


def _to_cons(rho, u, p, gamma):
    E = p / (gamma - 1.0) + 0.5 * rho * u * u
    return jnp.stack([rho, rho * u, E])


# ──────────────────────────── WENO3 (element-wise) ────────────────────────────

def _weno3_left(qm1, q0, qp1):
    p0 = -0.5 * qm1 + 1.5 * q0
    p1 =  0.5 * q0  + 0.5 * qp1
    b0 = (q0  - qm1) ** 2
    b1 = (qp1 - q0 ) ** 2
    a0 = (1.0 / 3.0) / (EPS + b0) ** 2
    a1 = (2.0 / 3.0) / (EPS + b1) ** 2
    return (a0 * p0 + a1 * p1) / (a0 + a1)


def _weno3_right(q0, qp1, qp2):
    p0 =  1.5 * qp1 - 0.5 * qp2
    p1 =  0.5 * q0  + 0.5 * qp1
    b0 = (qp2 - qp1) ** 2
    b1 = (qp1 - q0 ) ** 2
    a0 = (1.0 / 3.0) / (EPS + b0) ** 2
    a1 = (2.0 / 3.0) / (EPS + b1) ** 2
    return (a0 * p0 + a1 * p1) / (a0 + a1)


# ──────────────────────────── HLLC (vectorised) ───────────────────────────────

def _hllc(rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma):
    """All inputs are 1-D arrays of length N+1. Returns flux (3, N+1)."""
    a_L = jnp.sqrt(gamma * p_L / rho_L)
    a_R = jnp.sqrt(gamma * p_R / rho_R)

    S_L    = jnp.minimum(u_L - a_L, u_R - a_R)
    S_R    = jnp.maximum(u_L + a_L, u_R + a_R)
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
             (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))

    F_L = jnp.stack([rho_L * u_L,
                     rho_L * u_L * u_L + p_L,
                     u_L * (E_L + p_L)])
    F_R = jnp.stack([rho_R * u_R,
                     rho_R * u_R * u_R + p_R,
                     u_R * (E_R + p_R)])

    # Left star state
    chi_L = rho_L * (S_L - u_L) / (S_L - S_star)
    Es_L  = chi_L * (E_L / rho_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L))))
    U_L   = jnp.stack([rho_L, rho_L * u_L, E_L])
    Us_L  = jnp.stack([chi_L, chi_L * S_star, Es_L])
    Fs_L  = F_L + S_L[None] * (Us_L - U_L)

    # Right star state
    chi_R = rho_R * (S_R - u_R) / (S_R - S_star)
    Es_R  = chi_R * (E_R / rho_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R))))
    U_R   = jnp.stack([rho_R, rho_R * u_R, E_R])
    Us_R  = jnp.stack([chi_R, chi_R * S_star, Es_R])
    Fs_R  = F_R + S_R[None] * (Us_R - U_R)

    F = jnp.where(S_L[None] >= 0.0,   F_L,
        jnp.where(S_star[None] >= 0.0, Fs_L,
        jnp.where(S_R[None] <= 0.0,   F_R, Fs_R)))
    return F


# ──────────────────────────── spatial operator ────────────────────────────────

def _spatial_op(Q, dx, gamma, periodic: bool):
    N = Q.shape[1]

    # Ghost cell padding
    if periodic:
        Q_ext = jnp.concatenate([Q[:, -NG:], Q, Q[:, :NG]], axis=1)
    else:
        Q_ext = jnp.concatenate([Q[:, :1], Q[:, :1], Q, Q[:, -1:], Q[:, -1:]], axis=1)

    rho_e, u_e, p_e = _to_prim(Q_ext, gamma)

    # WENO3: N+1 interfaces, stencil windows into Q_ext length N+4
    rho_L = _weno3_left( rho_e[0:N+1], rho_e[1:N+2], rho_e[2:N+3])
    u_L   = _weno3_left(   u_e[0:N+1],   u_e[1:N+2],   u_e[2:N+3])
    p_L   = _weno3_left(   p_e[0:N+1],   p_e[1:N+2],   p_e[2:N+3])

    rho_R = _weno3_right(rho_e[1:N+2], rho_e[2:N+3], rho_e[3:N+4])
    u_R   = _weno3_right(  u_e[1:N+2],   u_e[2:N+3],   u_e[3:N+4])
    p_R   = _weno3_right(  p_e[1:N+2],   p_e[2:N+3],   p_e[3:N+4])

    E_L = p_L / (gamma - 1.0) + 0.5 * rho_L * u_L * u_L
    E_R = p_R / (gamma - 1.0) + 0.5 * rho_R * u_R * u_R

    F = _hllc(rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma)

    return -(F[:, 1:] - F[:, :-1]) / dx


# JIT-compiled step for each BC type
@partial(jax.jit, static_argnums=(2, 3))
def _step_jit(Q, dt, gamma_int, periodic_bool, dx):
    # gamma passed as int-scaled to work with static_argnums; recover as float
    gamma = gamma_int / 1000.0
    L1 = _spatial_op(Q, dx, gamma, periodic_bool)
    Q1 = Q + dt * L1
    L2 = _spatial_op(Q1, dx, gamma, periodic_bool)
    return 0.5 * (Q + Q1 + dt * L2)


# ──────────────────────────── solver class ────────────────────────────────────

class JaxEuler1D:
    """
    1-D compressible Euler solver using JAX (CPU).
    Same scheme as WarpEuler1D for direct backend comparison.

    Parameters
    ----------
    N       : number of real cells
    dx      : cell width
    gamma   : ratio of specific heats
    bc      : "outflow" or "periodic"
    """

    def __init__(self, N: int, dx: float, gamma: float = 1.4, bc: str = "outflow"):
        self.N        = N
        self.dx       = float(dx)
        self._gamma   = float(gamma)
        self._gamma_k = int(round(gamma * 1000))   # static key for JIT
        self._per     = bc == "periodic"
        self._Q       = None
        self._t       = 0.0

    def initialize(self, Q0: np.ndarray) -> None:
        assert Q0.shape == (3, self.N)
        self._Q = jnp.array(Q0, dtype=jnp.float32)
        self._t = 0.0

    def step(self, dt: float) -> None:
        self._Q = _step_jit(self._Q, dt, self._gamma_k, self._per, self.dx)
        self._t += dt

    def compute_dt(self, cfl: float = 0.4) -> float:
        Q = np.asarray(self._Q)
        rho = Q[0];  u = Q[1] / rho
        p   = (self._gamma - 1.0) * (Q[2] - 0.5 * rho * u ** 2)
        a   = np.sqrt(np.abs(self._gamma * p / rho))
        return float(cfl * self.dx / np.max(np.abs(u) + a))

    def run(self, t_end: float, cfl: float = 0.4) -> int:
        n = 0
        while self._t < t_end:
            dt = min(self.compute_dt(cfl), t_end - self._t)
            self.step(dt)
            n += 1
        return n

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self._Q)

    @property
    def time(self) -> float:
        return self._t
