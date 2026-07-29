"""Xbox Adaptive Joystick support (macOS, USB-C).

The joystick enumerates as one of two USB devices:
  0x0b1a  GIP mode (Microsoft's vendor protocol, invisible to macOS) — its
          state after plug-in, forever, until a host pokes it.
  0x0b1c  HID mode (standard gamepad: 5-byte reports, 7 buttons, x/y int16).

Sending the GIP device a SET_CONFIGURATION request makes it re-enumerate as
the HID gamepad within ~2 s (empirically determined; descriptor reads alone
do nothing). `ensure_hid_mode` does exactly that, and is idempotent.

Usage:
    uv run python -m controlkit.joystick            # CLI
or
    from controlkit.joystick import ensure_hid_mode
    ensure_hid_mode()                               # raises on failure
"""

import time

import hid
import usb.core
import usb.util
from usb.backend import libusb1

VID = 0x045E
PID_GIP = 0x0B1A
PID_HID = 0x0B1C

BUTTON_MAP = {
    "x1": 4,
    "x2": 5,
    "x3": 2,
    "x4": 3,
    "x5": 1,
    "x6": 0,
    "stick": 6,
}

def _backend():
    backend = libusb1.get_backend() or libusb1.get_backend(
        find_library=lambda name: "/opt/homebrew/lib/libusb-1.0.dylib"
    )
    if backend is None:
        raise RuntimeError("libusb-1.0 not found (brew install libusb)")
    return backend


def hid_present():
    """Return True if the joystick is enumerated as a HID gamepad (0x0b1c)."""
    return bool(hid.enumerate(VID, PID_HID))


def ensure_hid_mode(timeout_s: float = 10.0, verbose: bool = False) -> float:
    """Make sure the joystick is in HID mode, poking it out of GIP mode if needed.

    Args:
        timeout_s: how long to wait for the HID device after the poke.
        verbose: print each step to stdout.

    Returns:
        Seconds it took (0.0 if the device was already in HID mode).

    Raises:
        RuntimeError: joystick not plugged in, or it never entered HID mode.
    """
    def say(msg):
        if verbose:
            print(msg, flush=True)

    say(f"checking for HID gamepad ({VID:04x}:{PID_HID:04x})...")
    if hid_present():
        say("  found — already in HID mode, nothing to do")
        return 0.0
    say("  not present")

    say(f"looking for GIP device ({VID:04x}:{PID_GIP:04x}) via libusb...")
    dev = usb.core.find(idVendor=VID, idProduct=PID_GIP, backend=_backend())
    if dev is None:
        raise RuntimeError(
            "Xbox Adaptive Joystick not found in either mode (0x0b1a/0x0b1c). "
            "Is it plugged in?"
        )
    say(f"  found on bus {dev.bus} address {dev.address}")

    # SET_CONFIGURATION is what makes the firmware re-enumerate as a HID
    # gamepad; the device may drop off the bus mid-request as it does so.
    say("sending SET_CONFIGURATION (triggers re-enumeration as HID gamepad)...")
    try:
        dev.set_configuration()
        say("  request accepted")
    except usb.core.USBError as e:
        say(f"  request errored ({e}) — expected if the device is already re-enumerating")
    finally:
        usb.util.dispose_resources(dev)

    say(f"waiting up to {timeout_s:.0f}s for the HID gamepad to appear...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if hid_present():
            took = time.monotonic() - t0
            say(f"  appeared after {took:.1f}s")
            return took
        time.sleep(0.2)
    raise RuntimeError(f"poked GIP device but no HID gamepad appeared within {timeout_s}s")


if __name__ == "__main__":
    took = ensure_hid_mode(verbose=True)
    print("already in HID mode" if took == 0.0 else f"switched to HID mode in {took:.1f}s")
