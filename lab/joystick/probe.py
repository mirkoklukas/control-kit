"""Phase 1: enumerate Microsoft USB devices and dump their interface/endpoint layout.

Looks for the Xbox Adaptive Joystick's GIP interface: vendor-specific class
0xff, subclass 0x47, protocol 0xd0, with one interrupt IN and one interrupt
OUT endpoint.

Run: uv run python lab/joystick/probe.py
"""

import usb.core
import usb.util
from usb.backend import libusb1

MICROSOFT_VID = 0x045E

EP_TYPES = {
    usb.util.ENDPOINT_TYPE_CTRL: "control",
    usb.util.ENDPOINT_TYPE_ISO: "isochronous",
    usb.util.ENDPOINT_TYPE_BULK: "bulk",
    usb.util.ENDPOINT_TYPE_INTR: "interrupt",
}


def get_backend():
    """Return a libusb1 backend, falling back to the Homebrew dylib path.

    Args:
        None.

    Returns:
        A pyusb backend object, or raises RuntimeError if none is found.
    """
    backend = libusb1.get_backend()
    if backend is None:
        backend = libusb1.get_backend(
            find_library=lambda name: "/opt/homebrew/lib/libusb-1.0.dylib"
        )
    if backend is None:
        raise RuntimeError("libusb-1.0 not found (brew install libusb)")
    return backend


def safe_string(dev, index):
    """Read a USB string descriptor, returning None on access errors.

    Args:
        dev: pyusb device.
        index: string descriptor index (0 means "no string").

    Returns:
        The descriptor string or None.
    """
    if not index:
        return None
    try:
        return usb.util.get_string(dev, index)
    except usb.core.USBError:
        return None


def main():
    backend = get_backend()
    devices = list(usb.core.find(find_all=True, idVendor=MICROSOFT_VID, backend=backend))
    if not devices:
        print(f"No devices with idVendor=0x{MICROSOFT_VID:04x} found.")
        return

    gip_candidates = []
    for dev in devices:
        product = safe_string(dev, dev.iProduct)
        print(f"\ndevice idVendor=0x{dev.idVendor:04x} idProduct=0x{dev.idProduct:04x} "
              f"product={product!r}")
        try:
            cfg = dev.get_active_configuration()
        except usb.core.USBError:
            dev.set_configuration()
            cfg = dev.get_active_configuration()
        for intf in cfg:
            print(f"  interface {intf.bInterfaceNumber} alt {intf.bAlternateSetting}: "
                  f"class=0x{intf.bInterfaceClass:02x} "
                  f"subclass=0x{intf.bInterfaceSubClass:02x} "
                  f"protocol=0x{intf.bInterfaceProtocol:02x}")
            ep_dirs = set()
            for ep in intf:
                direction = ("IN" if usb.util.endpoint_direction(ep.bEndpointAddress)
                             == usb.util.ENDPOINT_IN else "OUT")
                ep_type = EP_TYPES[usb.util.endpoint_type(ep.bmAttributes)]
                print(f"    endpoint 0x{ep.bEndpointAddress:02x} {direction} {ep_type} "
                      f"maxpacket={ep.wMaxPacketSize}")
                if ep_type == "interrupt":
                    ep_dirs.add(direction)
            if (intf.bInterfaceClass == 0xFF and intf.bAlternateSetting == 0
                    and {"IN", "OUT"} <= ep_dirs):
                gip_candidates.append((dev.idProduct, product, intf.bInterfaceNumber))

    print()
    if gip_candidates:
        for pid, product, intf_num in gip_candidates:
            print(f"GIP candidate: idProduct=0x{pid:04x} product={product!r} "
                  f"interface={intf_num} (vendor class 0xff, interrupt IN+OUT)")
    else:
        print("NO vendor-specific (0xff) interface with interrupt IN+OUT found. STOP.")


if __name__ == "__main__":
    main()
