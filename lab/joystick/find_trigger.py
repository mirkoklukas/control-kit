"""Find which libusb action flips the joystick from GIP mode to HID mode.

Escalates through increasingly invasive USB operations on the GIP device
(0x0b1a), checking after each whether the HID gamepad (0x0b1c) appeared.
Stops at the first action that works and reports it.

Run: uv run python lab/joystick/find_trigger.py
"""

import time

import usb.core
import usb.util

from controlkit.joystick import VID, PID_GIP, _backend, hid_present


def flipped(wait_s=8.0):
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if hid_present():
            return time.monotonic() - t0
        time.sleep(0.2)
    return None


def find_gip():
    return usb.core.find(idVendor=VID, idProduct=PID_GIP, backend=_backend())


def step(name, action):
    """Run one candidate trigger against the GIP device; report and check."""
    dev = find_gip()
    if dev is None:
        print(f"[{name}] GIP device gone before step ran")
        return hid_present()
    try:
        action(dev)
        print(f"[{name}] action completed without error")
    except usb.core.USBError as e:
        print(f"[{name}] USBError: {e}")
    finally:
        try:
            usb.util.dispose_resources(dev)
        except Exception:
            pass
    took = flipped()
    if took is not None:
        print(f"[{name}] >>> FLIPPED to HID after {took:.1f}s <<<")
        return True
    print(f"[{name}] no flip")
    return False


def a_descriptors(dev):
    usb.util.get_string(dev, dev.iProduct)
    dev.get_active_configuration()


def b_set_configuration(dev):
    dev.set_configuration()


def c_claim_release(dev):
    try:
        dev.get_active_configuration()
    except usb.core.USBError:
        dev.set_configuration()
    usb.util.claim_interface(dev, 0)
    usb.util.release_interface(dev, 0)


def d_reset(dev):
    dev.reset()


def main():
    if hid_present():
        print("already in HID mode; replug first to test triggers")
        return
    for name, action in [
        ("A descriptor reads", a_descriptors),
        ("B set_configuration", b_set_configuration),
        ("C claim+release intf", c_claim_release),
        ("D device reset", d_reset),
    ]:
        if step(name, action):
            return
    print("no action flipped it; GIP handshake may be required after all")


if __name__ == "__main__":
    main()
