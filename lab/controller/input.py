"""DualSense input, via raw hidapi report parsing.

Why not pygame/SDL: SDL's event pump must run on the process **main thread** on
macOS, but the MuJoCo viewer under ``mjpython`` already owns the main thread and
runs our loop on a worker thread -- so ``pygame.event.pump()`` crashes there
("nextEventMatchingMask should only be called from the Main Thread"). hidapi
reads a device handle directly, on any thread, so it coexists with the viewer.

This parses the DualSense input report (Sony VID ``0x054C``, PID ``0x0CE6``):
USB is report ``0x01`` (axes at byte 1); Bluetooth is ``0x31`` (shifted +1). It is
the thin, faithful reader: normalizes ranges, decodes the button/hat bytes, and
detects edges, leaving sign/semantics to the mapping layer above.

Conventions:
- sticks in [-1, 1] with a radial deadzone; raw axes are +x right, **+y down**.
- triggers (``l2``/``r2``) in [0, 1].
- ``pressed`` / ``released`` are button names whose state flipped since the
  previous :meth:`DualSense.poll` (rising / falling edges).

M0 entry point -- watch the normalized state and button edges:

    uv run python -m lab.controller.input
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import hid

VID = 0x054C
PIDS = (0x0CE6, 0x0DF2)      # DualSense, DualSense Edge
CENTER = 128.0
HALF = 127.0
DEADZONE = 0.08

# (byte, bitmask) for each button, relative to the first button byte (btn0).
# btn0: d-pad hat in the low nibble + the four face buttons; btn1: shoulders /
# menu / stick clicks; btn2: PS / touchpad / mic.
_BITS = {
    "x": (0, 0x10),      # square
    "a": (0, 0x20),      # cross
    "b": (0, 0x40),      # circle
    "y": (0, 0x80),      # triangle
    "l1": (1, 0x01),
    "r1": (1, 0x02),
    "back": (1, 0x10),   # Create
    "start": (1, 0x20),  # Options
    "l3": (1, 0x40),
    "r3": (1, 0x80),
    "guide": (2, 0x01),  # PS
}
# d-pad hat value (btn0 low nibble) -> which directions are held. 8 = neutral.
_HAT = {
    "up": (7, 0, 1),
    "right": (1, 2, 3),
    "down": (3, 4, 5),
    "left": (5, 6, 7),
}
BUTTONS = tuple(_BITS) + tuple(_HAT)


@dataclass
class ControllerState:
    """One normalized snapshot of the controller.

    Args:
        lx, ly: left stick in [-1, 1] (+x right, +y down), deadzoned.
        rx, ry: right stick, same convention.
        l2, r2: triggers in [0, 1].
        buttons: name -> bool, current state (keys are :data:`BUTTONS`).
        pressed: names that went down since the previous poll (rising edges).
        released: names that went up since the previous poll (falling edges).
    """

    lx: float = 0.0
    ly: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    l2: float = 0.0
    r2: float = 0.0
    buttons: dict = field(default_factory=dict)
    pressed: frozenset = frozenset()
    released: frozenset = frozenset()


def _deadzone(x: float, y: float, dz: float = DEADZONE):
    """Radial deadzone: zero inside ``dz``, rescaled to reach 1 at the rim."""
    r = math.hypot(x, y)
    if r < dz:
        return 0.0, 0.0
    scale = (r - dz) / (1.0 - dz) / r
    return x * scale, y * scale


def _axis(v: int) -> float:
    return max(-1.0, min(1.0, (v - CENTER) / HALF))


class DualSense:
    """A connected DualSense, read over hidapi.

    Args:
        index: which matching device to open (0 = first).

    Raises:
        RuntimeError: if no DualSense is found.
    """

    def __init__(self, index: int = 0):
        matches = [d for pid in PIDS for d in hid.enumerate(VID, pid)]
        if not matches:
            raise RuntimeError(
                "no DualSense found -- plug it in over USB (or pair over "
                "Bluetooth) and retry.")
        self.dev = hid.device()
        self.dev.open_path(matches[index]["path"])
        self.name = matches[index].get("product_string") or "DualSense"

        # One blocking read to detect USB (report 0x01) vs Bluetooth (0x31); the
        # axis block sits one byte later over Bluetooth.
        first = self.dev.read(64, 500)
        self._base = 2 if first and first[0] == 0x31 else 1
        self.dev.set_nonblocking(1)
        self._buf = first or None
        self._prev = {k: False for k in BUTTONS}

    def poll(self) -> ControllerState:
        """Drain to the latest report and return the current state."""
        # drain the nonblocking queue, keeping only the freshest report
        while True:
            r = self.dev.read(64)
            if not r:
                break
            self._buf = r
        if self._buf is None:
            return ControllerState(buttons={k: False for k in BUTTONS})

        b, r = self._base, self._buf
        lx, ly = _deadzone(_axis(r[b]), _axis(r[b + 1]))
        rx, ry = _deadzone(_axis(r[b + 2]), _axis(r[b + 3]))
        l2, r2 = r[b + 4] / 255.0, r[b + 5] / 255.0
        btn = (r[b + 7], r[b + 8], r[b + 9])                   # btn0, btn1, btn2
        hat = btn[0] & 0x0F

        cur = {k: bool(btn[i] & m) for k, (i, m) in _BITS.items()}
        cur.update({k: hat in vals for k, vals in _HAT.items()})
        pressed = frozenset(k for k in BUTTONS if cur[k] and not self._prev[k])
        released = frozenset(k for k in BUTTONS if not cur[k] and self._prev[k])
        self._prev = cur

        return ControllerState(lx, ly, rx, ry, l2, r2, cur, pressed, released)

    def close(self):
        self.dev.close()


def _watch(hz: float = 30.0):
    """M0: open the pad and stream normalized state + button edges to stdout."""
    pad = DualSense()
    print(f"connected: {pad.name}   (Ctrl-C to stop)")
    dt = 1.0 / hz
    try:
        while True:
            s = pad.poll()
            line = (f"L({s.lx:+.2f},{s.ly:+.2f}) R({s.rx:+.2f},{s.ry:+.2f}) "
                    f"L2 {s.l2:.2f} R2 {s.r2:.2f}")
            if s.pressed:
                line += f"   +{sorted(s.pressed)}"
            if s.released:
                line += f"   -{sorted(s.released)}"
            print(line, flush=True)
            time.sleep(dt)
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        pad.close()


if __name__ == "__main__":
    _watch()
