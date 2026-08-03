"""Recording session and sink: where the logs go, and when.

Separate from drawing on purpose -- pick the sink once, then every ``log_*`` in
``draw``/``robot`` just writes to it.
"""

from pathlib import Path

import rerun as rr


def _grpc_url(connect) -> str:
    """gRPC url from a port int (localhost) or a full url string."""
    return connect if isinstance(connect, str) else f"rerun+http://127.0.0.1:{int(connect)}/proxy"


def init_viewer(app: str = "controlkit", *, spawn=True, save=None, connect=None):
    """Set up the rerun recording and its sink.

    First option set wins.

    Args:
        app: recording/application id.
        spawn: launch a viewer process.
        save: path to write a ``.rrd`` to (parents created).
        connect: port int or url of an already-running viewer.
    """
    rr.init(app)
    if connect is not None:
        rr.connect_grpc(_grpc_url(connect))
    elif save is not None:
        save = Path(save)
        save.parent.mkdir(parents=True, exist_ok=True)
        rr.save(save)
    elif spawn:
        rr.spawn()
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)   # z is up


def set_time(i, *, timeline="t"):
    """Place subsequent logs at sequence index ``i`` on ``timeline``.

    Wraps ``rr.set_time`` so callers can drive their own timeline without
    importing rerun.

    Args:
        i: sequence index.
        timeline: timeline name.
    """
    rr.set_time(timeline, sequence=int(i))
