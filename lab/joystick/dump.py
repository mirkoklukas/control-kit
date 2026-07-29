"""Phase 2: claim the GIP interface, send power-on, hexdump incoming frames.

Run: uv run python lab/joystick/dump.py [seconds]
Stops after `seconds` (default: run until Ctrl-C). Frames starting with 0x20
are input reports and should change when the stick moves.
"""

import sys
import time

import usb.core
import usb.util

from probe import get_backend, MICROSOFT_VID

ADAPTIVE_JOYSTICK_PID = 0x0B1A
GIP_INTERFACE = 0
EP_IN = 0x81
EP_OUT = 0x01

# GIP power-on: command 0x05, flags 0x20 (internal), sequence, length 1,
# payload 0x00 (= on). Per xone GIP_CMD_POWER; sequence must be nonzero.
POWER_ON = bytes([0x05, 0x20, 0x01, 0x01, 0x00])


def open_device(wait_s=60.0):
    """Find the joystick, claim the GIP interface, and return the device.

    Args:
        wait_s: how long to keep retrying enumeration (covers a replug).

    Returns:
        The claimed pyusb device.
    """
    deadline = time.monotonic() + wait_s
    last_err = None
    while time.monotonic() < deadline:
        dev = usb.core.find(idVendor=MICROSOFT_VID, idProduct=ADAPTIVE_JOYSTICK_PID,
                            backend=get_backend())
        if dev is None:
            time.sleep(0.5)
            continue
        try:
            try:
                dev.get_active_configuration()
            except usb.core.USBError:
                dev.set_configuration()
            usb.util.claim_interface(dev, GIP_INTERFACE)
            return dev
        except usb.core.USBError as e:
            # Stale entry from a replug, or a transient claim failure: retry.
            last_err = e
            usb.util.dispose_resources(dev)
            time.sleep(0.5)
    raise RuntimeError(
        f"could not open Xbox Adaptive Joystick (045e:0b1a) within {wait_s}s"
        + (f"; last error: {last_err}" if last_err else "")
    )


def main():
    deadline = time.monotonic() + float(sys.argv[1]) if len(sys.argv) > 1 else None
    dev = open_device()
    try:
        n = dev.write(EP_OUT, POWER_ON, timeout=1000)
        print(f"sent power-on ({n} bytes): {POWER_ON.hex(' ')}")
        count = 0
        while deadline is None or time.monotonic() < deadline:
            try:
                data = bytes(dev.read(EP_IN, 64, timeout=200))
            except usb.core.USBTimeoutError:
                continue
            count += 1
            tag = " <-- INPUT" if data[:1] == b"\x20" else ""
            print(f"[{count:4d}] len={len(data):2d}  {data.hex(' ')}{tag}")
    except KeyboardInterrupt:
        pass
    finally:
        usb.util.release_interface(dev, GIP_INTERFACE)
        usb.util.dispose_resources(dev)
        print("released interface")


if __name__ == "__main__":
    main()
