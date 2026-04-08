"""Shared command buffer — thread-safe interface between command producers
(M2: slider; M3: SlowLoop/VLM) and the FastLoop consumer.

The buffer holds a single 3-tuple `(vx, vy, wz)` that producers overwrite via
`set()` and the FastLoop reads via `get()`. Semantics are *last-write-wins* —
no queue, no history. The FastLoop consumes whatever was last written, and old
values are silently dropped.

Why a lock if M2 is single-threaded?
    ipywidgets callbacks may be invoked from the Jupyter kernel's event loop
    thread, not the cell-execution thread. Even in the "single-thread" M2 design
    we treat the buffer as if it were truly shared, so the M3 transition
    (slider → SlowLoop) is a one-line swap with no semantic change.

The buffer also clamps inputs to the command ranges from `pbt_meta.config` —
producers cannot push out-of-distribution commands that the trained policy has
never seen. Out-of-range values are clipped silently and a count is kept for
inspection.
"""

import threading
from dataclasses import dataclass, field
from typing import Tuple

from pbt_meta.config import COMMAND_RANGES


@dataclass
class _Stats:
    """Internal stats tracked by the buffer."""
    n_writes: int = 0
    n_reads: int = 0
    n_clamps: int = 0


class SharedCommandBuffer:
    """Thread-safe last-write-wins buffer for (vx, vy, wz) commands.

    Example:
        buf = SharedCommandBuffer()
        buf.set(0.5, 0.0, 0.0)        # producer writes
        vx, vy, wz = buf.get()        # consumer reads -> (0.5, 0.0, 0.0)

    All values are floats. Out-of-range values are clamped silently to the
    bounds defined in `pbt_meta.config.COMMAND_RANGES`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Initial value is "stop"
        self._vx: float = 0.0
        self._vy: float = 0.0
        self._wz: float = 0.0
        self._stats = _Stats()

    def set(self, vx: float, vy: float, wz: float) -> None:
        """Write a new command (clamped to range). Last write wins."""
        vx_lo, vx_hi = COMMAND_RANGES["lin_vel_x"]
        vy_lo, vy_hi = COMMAND_RANGES["lin_vel_y"]
        wz_lo, wz_hi = COMMAND_RANGES["ang_vel_z"]

        # Coerce to float (in case caller passes numpy scalar / tensor)
        vx = float(vx)
        vy = float(vy)
        wz = float(wz)

        # Clamp + count clamps
        clamped = False
        if vx < vx_lo:
            vx = vx_lo; clamped = True
        elif vx > vx_hi:
            vx = vx_hi; clamped = True
        if vy < vy_lo:
            vy = vy_lo; clamped = True
        elif vy > vy_hi:
            vy = vy_hi; clamped = True
        if wz < wz_lo:
            wz = wz_lo; clamped = True
        elif wz > wz_hi:
            wz = wz_hi; clamped = True

        with self._lock:
            self._vx = vx
            self._vy = vy
            self._wz = wz
            self._stats.n_writes += 1
            if clamped:
                self._stats.n_clamps += 1

    def get(self) -> Tuple[float, float, float]:
        """Read the current command. Returns (vx, vy, wz)."""
        with self._lock:
            self._stats.n_reads += 1
            return (self._vx, self._vy, self._wz)

    def stop(self) -> None:
        """Convenience: set command to (0, 0, 0)."""
        self.set(0.0, 0.0, 0.0)

    def stats(self) -> dict:
        """Return read/write/clamp counts (snapshot)."""
        with self._lock:
            return {
                "n_writes": self._stats.n_writes,
                "n_reads": self._stats.n_reads,
                "n_clamps": self._stats.n_clamps,
            }
