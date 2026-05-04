import numpy as np
import warp as wp

from .kernels import fused_rk_stage_1d_outflow, fused_rk_stage_1d_periodic

_FUSED_KERNELS = {
    "outflow":  fused_rk_stage_1d_outflow,
    "periodic": fused_rk_stage_1d_periodic,
}

NVARS_1D = 3   # [rho, rho*u, E]
NG = 2         # ghost cells required by WENO3


class WarpEuler1D:
    """
    1-D compressible Euler solver on a uniform Cartesian grid.

    Scheme: WENO3 reconstruction (primitive variables) + HLLC Riemann solver + SSP-RK2.
    Ghost cells (ng=2) are embedded in the state array; real cells live at [ng : ng+N].

    Kernel architecture (fused):
      step() fires 2 kernel launches per timestep (1 per RK stage).
      Each launch fuses: BC handling (inline) + WENO3 + HLLC + RK update.
      No intermediate global flux array — all interface fluxes live in registers.

    Parameters
    ----------
    N      : number of real cells
    dx     : cell width (m)
    gamma  : ratio of specific heats (default 1.4)
    bc     : boundary condition — "outflow" or "periodic"
    device : Warp device string ("cuda" or "cpu")
    """

    def __init__(
        self,
        N: int,
        dx: float,
        gamma: float = 1.4,
        bc: str = "outflow",
        device: str = "cuda",
    ):
        if bc not in _FUSED_KERNELS:
            raise ValueError(f"bc must be one of {list(_FUSED_KERNELS)}, got {bc!r}")

        self.N      = N
        self.dx     = float(dx)
        self.gamma  = float(gamma)
        self.bc     = bc
        self.device = device
        self._ng    = NG
        self._N_ext = N + 2 * NG
        self._t     = 0.0

        wp.init()

        shape = (NVARS_1D, self._N_ext)
        self._Q       = wp.zeros(shape, dtype=float, device=device)
        self._Q_stage = wp.zeros(shape, dtype=float, device=device)
        # Note: no _F array — fluxes are computed in registers inside the fused kernel

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self, Q0: np.ndarray) -> None:
        """Set initial condition from a (3, N) numpy array of conserved variables."""
        assert Q0.shape == (NVARS_1D, self.N), \
            f"Expected shape ({NVARS_1D}, {self.N}), got {Q0.shape}"

        Q_ext = np.zeros((NVARS_1D, self._N_ext), dtype=np.float32)
        Q_ext[:, NG : NG + self.N] = Q0.astype(np.float32)
        self._Q = wp.from_numpy(Q_ext, dtype=float, device=self.device)
        self._t = 0.0
        # Ghost cells are not pre-filled — the fused kernel handles BC inline.

    def step(self, dt: float) -> None:
        """Advance by one SSP-RK2 timestep (2 kernel launches)."""
        # Stage 1: Q_stage = Q + dt * L(Q)
        self._fused_stage(self._Q, self._Q, self._Q_stage, dt, 1.0, 0.0, 1.0)

        # Stage 2: Q = 0.5*Q + 0.5*Q_stage + 0.5*dt*L(Q_stage)
        # Q_ref = Q (start-of-step), Q_in = Q_stage, Q_out = Q (in-place safe: each thread
        # reads Q_ref[i] and writes Q_out[i] for the same i, no cross-thread hazard)
        self._fused_stage(self._Q_stage, self._Q, self._Q, dt, 0.5, 0.5, 0.5)

        self._t += dt

    def compute_dt(self, cfl: float = 0.4) -> float:
        """Return the largest stable dt satisfying the CFL condition."""
        Q = self._Q.numpy()[:, NG : NG + self.N]
        rho = Q[0]
        u   = Q[1] / rho
        p   = (self.gamma - 1.0) * (Q[2] - 0.5 * rho * u ** 2)
        a   = np.sqrt(np.abs(self.gamma * p / rho))
        return cfl * self.dx / float(np.max(np.abs(u) + a))

    def run(self, t_end: float, cfl: float = 0.4, n_max: int = 10_000_000) -> int:
        """Run to t_end using adaptive CFL timestepping. Returns step count."""
        n = 0
        while self._t < t_end and n < n_max:
            dt = min(self.compute_dt(cfl), t_end - self._t)
            self.step(dt)
            n += 1
        return n

    @property
    def state(self) -> np.ndarray:
        """Current conserved state as (3, N) numpy float32 array."""
        return self._Q.numpy()[:, NG : NG + self.N]

    @property
    def time(self) -> float:
        return self._t

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _fused_stage(
        self,
        Q_in:   wp.array,
        Q_ref:  wp.array,
        Q_out:  wp.array,
        dt:     float,
        alpha:  float,
        beta:   float,
        coeff:  float,
    ) -> None:
        wp.launch(
            _FUSED_KERNELS[self.bc],
            dim=self.N,
            inputs=[Q_in, Q_ref, Q_out,
                    self._ng, self.N, self.gamma,
                    float(dt), self.dx,
                    float(alpha), float(beta), float(coeff)],
            device=self.device,
        )
