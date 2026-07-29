"""Watch the joystick's USB mode, then capture and decode HID reports live.

Phase A: poll enumeration once a second and log mode changes
         (GIP 0x0b1a vs HID-fallback 0x0b1c), so a replug shows how long
         the device stays in GIP mode before falling back to HID.
Phase B: as soon as the HID device is present, open it and stream decoded
         reports (buttons bitfield, x, y) until the time budget runs out.

Run: uv run python lab/joystick/hid_capture.py [seconds, default 45]
"""

import struct
import sys
import time

import hid

VID, PID_GIP, PID_HID = 0x045E, 0x0B1A, 0x0B1C


def current_mode():
    """Return 'GIP', 'HID', or None depending on which PID is enumerated.

    hid.enumerate only sees the HID device; for GIP presence we check via
    pyusb (vendor interface, invisible to the HID layer).

    Returns:
        'HID' | 'GIP' | None.
    """
    if any(d["product_id"] == PID_HID for d in hid.enumerate(VID)):
        return "HID"
    import usb.core
    from probe import get_backend
    if usb.core.find(idVendor=VID, idProduct=PID_GIP, backend=get_backend()) is not None:
        return "GIP"
    return None


def main():
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
    t0 = time.monotonic()
    mode = None
    while time.monotonic() - t0 < budget:
        m = current_mode()
        if m != mode:
            print(f"[{time.monotonic() - t0:6.1f}s] mode: {mode} -> {m}", flush=True)
            mode = m
        if m == "HID":
            break
        time.sleep(1.0)
    if mode != "HID":
        print("HID device never appeared; still in GIP mode or unplugged.")
        return

    dev = hid.device()
    dev.open(VID, PID_HID)
    dev.set_nonblocking(True)
    print("streaming (move stick / press buttons)...", flush=True)
    frames = 0
    last_line = None
    while time.monotonic() - t0 < budget:
        data = dev.read(64)
        if not data:
            time.sleep(0.002)
            continue
        frames += 1
        raw = bytes(data)
        buttons = raw[0]
        x, y = struct.unpack_from("<hh", raw, 1)
        line = f"raw={raw.hex(' ')}  buttons={buttons:08b} x={x:6d} y={y:6d}"
        if line != last_line:
            print(f"[{frames:5d}] {line}", flush=True)
            last_line = line
    dev.close()
    print(f"done, {frames} frames total")


if __name__ == "__main__":
    main()
