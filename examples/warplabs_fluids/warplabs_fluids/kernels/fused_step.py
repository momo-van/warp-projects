import warp as wp

from .reconstruct import weno3_left, weno3_right
from .riemann import hllc_flux_1d

# Fused bc+WENO3+HLLC+RK-stage kernel — 1 launch per RK stage (2 per step).
#
# Design vs the original 3-kernel-per-stage approach:
#   OLD: bc_kernel → compute_flux_1d → update_rk_1d   (3 launches, global F array)
#   NEW: fused_rk_stage_1d                             (1 launch, no global F array)
#
# Each thread owns one real cell. It computes the two interface fluxes that
# bound its cell (F_l at i-1/2, F_r at i+1/2) entirely in registers using a
# 5-cell stencil, then writes the RK-stage result directly to Q_out.
# Ghost-cell accesses are remapped inline via clamping (outflow) or modular
# arithmetic (periodic) — no separate bc launch needed.
#
# Memory traffic per RK stage vs old design:
#   OLD: Q read twice (flux + update) + F written + F read = 4 array passes
#   NEW: Q_in read once + Q_ref read once + Q_out written = 3 array passes
#
# RK update formula (same coefficients as update_rk_1d):
#   Q_out[i] = alpha*Q_ref[i] + beta*Q_in[i] + coeff*dt*(-(F_r - F_l)/dx)
#   Stage 1: alpha=1, beta=0, coeff=1   (Q_in = Q_ref = Q0 → Q1 = Q0 + dt*L(Q0))
#   Stage 2: alpha=0.5, beta=0.5, coeff=0.5  (Q_in=Q1, Q_ref=Q0 → Q_final)


@wp.func
def _flux_at_interface(
    rho_a: float, u_a: float, p_a: float,
    rho_b: float, u_b: float, p_b: float,
    rho_c: float, u_c: float, p_c: float,
    rho_d: float, u_d: float, p_d: float,
    gamma: float,
) -> wp.vec3f:
    """
    HLLC flux at interface b+1/2 from 4-cell stencil (a, b, c, d).
      Q_L = weno3_left (a, b, c)   left-biased reconstruction
      Q_R = weno3_right(b, c, d)   right-biased reconstruction
    All 12 primitive values stay in registers — no global writes.
    """
    rho_L = weno3_left(rho_a, rho_b, rho_c)
    u_L   = weno3_left(u_a,   u_b,   u_c)
    p_L   = weno3_left(p_a,   p_b,   p_c)
    rho_R = weno3_right(rho_b, rho_c, rho_d)
    u_R   = weno3_right(u_b,   u_c,   u_d)
    p_R   = weno3_right(p_b,   p_c,   p_d)
    E_L   = p_L / (gamma - 1.0) + 0.5 * rho_L * u_L * u_L
    E_R   = p_R / (gamma - 1.0) + 0.5 * rho_R * u_R * u_R
    return hllc_flux_1d(rho_L, u_L, p_L, E_L, rho_R, u_R, p_R, E_R, gamma)


@wp.kernel
def fused_rk_stage_1d_outflow(
    Q_in:  wp.array2d(dtype=float),   # spatial-op input   (3, N+2*ng)
    Q_ref: wp.array2d(dtype=float),   # RK reference state (3, N+2*ng)
    Q_out: wp.array2d(dtype=float),   # output             (3, N+2*ng)
    ng: int, N: int, gamma: float,
    dt: float, dx: float,
    alpha: float, beta: float, coeff: float,
):
    i  = wp.tid()   # real-cell index, 0 .. N-1
    i0 = ng + i

    # Stencil indices — out-of-bounds clamped to nearest real cell (outflow)
    ia  = wp.clamp(i0 - 2, ng, ng + N - 1)
    ib  = wp.clamp(i0 - 1, ng, ng + N - 1)
    # ic = i0 always in [ng, ng+N-1]
    id_ = wp.clamp(i0 + 1, ng, ng + N - 1)
    ie  = wp.clamp(i0 + 2, ng, ng + N - 1)

    # Conserved → primitives for 5 stencil cells (all in registers)
    rho_a_c = Q_in[0, ia];  rho_a = rho_a_c;  u_a = Q_in[1, ia] / rho_a;  p_a = (gamma - 1.0) * (Q_in[2, ia] - 0.5 * rho_a * u_a * u_a)
    rho_b_c = Q_in[0, ib];  rho_b = rho_b_c;  u_b = Q_in[1, ib] / rho_b;  p_b = (gamma - 1.0) * (Q_in[2, ib] - 0.5 * rho_b * u_b * u_b)
    rho_c_c = Q_in[0, i0];  rhou_c = Q_in[1, i0]; E_c = Q_in[2, i0]
    rho_c = rho_c_c;  u_c = rhou_c / rho_c;  p_c = (gamma - 1.0) * (E_c - 0.5 * rho_c * u_c * u_c)
    rho_d_c = Q_in[0, id_]; rho_d = rho_d_c; u_d = Q_in[1, id_] / rho_d; p_d = (gamma - 1.0) * (Q_in[2, id_] - 0.5 * rho_d * u_d * u_d)
    rho_e_c = Q_in[0, ie];  rho_e = rho_e_c; u_e = Q_in[1, ie] / rho_e;  p_e = (gamma - 1.0) * (Q_in[2, ie] - 0.5 * rho_e * u_e * u_e)

    # Interface fluxes — both computed from register-resident primitives, never written globally
    F_l = _flux_at_interface(rho_a, u_a, p_a, rho_b, u_b, p_b, rho_c, u_c, p_c, rho_d, u_d, p_d, gamma)
    F_r = _flux_at_interface(rho_b, u_b, p_b, rho_c, u_c, p_c, rho_d, u_d, p_d, rho_e, u_e, p_e, gamma)

    # SSP-RK stage update — one global write per variable
    inv_dx = 1.0 / dx
    Q_out[0, i0] = alpha * Q_ref[0, i0] + beta * rho_c_c  + coeff * dt * (-(F_r[0] - F_l[0]) * inv_dx)
    Q_out[1, i0] = alpha * Q_ref[1, i0] + beta * rhou_c   + coeff * dt * (-(F_r[1] - F_l[1]) * inv_dx)
    Q_out[2, i0] = alpha * Q_ref[2, i0] + beta * E_c      + coeff * dt * (-(F_r[2] - F_l[2]) * inv_dx)


@wp.kernel
def fused_rk_stage_1d_periodic(
    Q_in:  wp.array2d(dtype=float),
    Q_ref: wp.array2d(dtype=float),
    Q_out: wp.array2d(dtype=float),
    ng: int, N: int, gamma: float,
    dt: float, dx: float,
    alpha: float, beta: float, coeff: float,
):
    i  = wp.tid()
    i0 = ng + i

    # Periodic wrapping: Warp uses C-style %, so shift by N before taking % to avoid negative results
    ia  = ng + (i - 2 + N) % N
    ib  = ng + (i - 1 + N) % N
    id_ = ng + (i + 1) % N
    ie  = ng + (i + 2) % N

    rho_a_c = Q_in[0, ia];  rho_a = rho_a_c;  u_a = Q_in[1, ia] / rho_a;  p_a = (gamma - 1.0) * (Q_in[2, ia] - 0.5 * rho_a * u_a * u_a)
    rho_b_c = Q_in[0, ib];  rho_b = rho_b_c;  u_b = Q_in[1, ib] / rho_b;  p_b = (gamma - 1.0) * (Q_in[2, ib] - 0.5 * rho_b * u_b * u_b)
    rho_c_c = Q_in[0, i0];  rhou_c = Q_in[1, i0]; E_c = Q_in[2, i0]
    rho_c = rho_c_c;  u_c = rhou_c / rho_c;  p_c = (gamma - 1.0) * (E_c - 0.5 * rho_c * u_c * u_c)
    rho_d_c = Q_in[0, id_]; rho_d = rho_d_c; u_d = Q_in[1, id_] / rho_d; p_d = (gamma - 1.0) * (Q_in[2, id_] - 0.5 * rho_d * u_d * u_d)
    rho_e_c = Q_in[0, ie];  rho_e = rho_e_c; u_e = Q_in[1, ie] / rho_e;  p_e = (gamma - 1.0) * (Q_in[2, ie] - 0.5 * rho_e * u_e * u_e)

    F_l = _flux_at_interface(rho_a, u_a, p_a, rho_b, u_b, p_b, rho_c, u_c, p_c, rho_d, u_d, p_d, gamma)
    F_r = _flux_at_interface(rho_b, u_b, p_b, rho_c, u_c, p_c, rho_d, u_d, p_d, rho_e, u_e, p_e, gamma)

    inv_dx = 1.0 / dx
    Q_out[0, i0] = alpha * Q_ref[0, i0] + beta * rho_c_c  + coeff * dt * (-(F_r[0] - F_l[0]) * inv_dx)
    Q_out[1, i0] = alpha * Q_ref[1, i0] + beta * rhou_c   + coeff * dt * (-(F_r[1] - F_l[1]) * inv_dx)
    Q_out[2, i0] = alpha * Q_ref[2, i0] + beta * E_c      + coeff * dt * (-(F_r[2] - F_l[2]) * inv_dx)
