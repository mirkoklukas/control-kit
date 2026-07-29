"""Passively watch which Microsoft PIDs are enumerated, printing changes.

No open/claim/configuration calls — pure descriptor enumeration, so it
cannot itself disturb the device. Timestamps are seconds since start.

Run: uv run python lab/joystick/watch_pids.py [seconds, default 60]
"""

import sys
import time

import usb.core

from probe import get_backend, MICROSOFT_VID


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    backend = get_backend()
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < budget:
        pids = sorted(
            dev.idProduct
            for dev in usb.core.find(find_all=True, idVendor=MICROSOFT_VID, backend=backend)
        )
        if pids != last:
            names = ", ".join(f"0x{p:04x}" for p in pids) or "none"
            print(f"[{time.monotonic() - t0:7.2f}s] {names}", flush=True)
            last = pids
        time.sleep(0.1)
    print("done")


if __name__ == "__main__":
    main()
