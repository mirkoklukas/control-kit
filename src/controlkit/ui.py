"""Shared terminal UI helpers."""

from __future__ import annotations

from rich.console import Console, Group
from rich.padding import Padding
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich.pretty import pprint
import yaml as _yaml_mod


console = Console()

PADDING_LEFT = 2


def silence() -> None:
    """Suppress all UI output. Call once at startup for machine-readable (--json) mode.

    Uses Rich's built-in quiet console — no need to guard individual functions.
    TODO: replace with ui.set_mode("silent") when multiple modes are needed.
    """
    global console
    console = Console(quiet=True)


HEADER_TEXT = Text()
HEADER_TEXT.append("Con...Trol...Kit...!", style="bold")


# # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Header, title, lines ...
#
# # # # # # # # # # # # # # # # # # # # # # # # # # #


def header(path: list[str], descr=None) -> None:
    """Branded header using grouped padding (original version)."""
    title = " → ".join([*path[:-1], f"[bold]{path[-1]}[/bold ]"])


    group = [f"{HEADER_TEXT} {title}"]
    if descr is not None:
        group.append(f"[dim]{descr}[/dim]")

    console.print(
        Padding(
            Group(*group),
            (1, 0, 1, PADDING_LEFT),
            style="",
            expand=False,
        ),
        highlight=False,
    )


def done(msg: str, hint: str = "") -> None:
    """Success panel printed at the end of a command."""
    lines = f"[bold green]✓ {msg}[/bold green]"
    if hint:
        lines += f"\n[dim]{hint}[/dim]"
    console.print(Padding(lines, (1, PADDING_LEFT, 1, PADDING_LEFT), expand=False))


def title(text: str) -> None:
    """Simple title line with bold text."""
    console.print(
        Padding(f"⏵⏵ [bold]{text}[/bold]", (0, PADDING_LEFT), style="", expand=False)
    )


def print(text: str, padding_left=PADDING_LEFT, highlight=False) -> None:
    """Simple line of text."""
    console.print(Padding(text, (0, padding_left), expand=False), highlight=highlight)


def print_line(text: str, padding_left=PADDING_LEFT, highlight=True) -> None:
    """Simple line of text."""
    console.print(
        Padding(text, (0, padding_left), style="", expand=False), highlight=highlight
    )


def print_progress(
    text: str, end: str = "\n", overwrite: bool = False, padding_left=PADDING_LEFT
) -> None:
    """Print a progress line, optionally overwriting the current line."""
    if overwrite:
        padding = " " * padding_left
        with console.capture() as capture:
            console.print(text, highlight=False, end="")
        rendered = capture.get()
        console.file.write(f"\r{padding}{rendered}")
        console.file.flush()
    else:
        console.print(
            Padding(text, (0, padding_left), expand=False), highlight=False, end=end
        )


def ask(prompt: str, default: str = "", password: bool = False) -> str:
    """Styled user prompt."""
    from rich.prompt import Prompt

    return Prompt.ask(
        f"{' ' * PADDING_LEFT}{prompt}", default=default, password=password
    )


def ask_int(prompt: str, choices: list[int]) -> int:
    """Styled integer prompt with a restricted list of choices."""
    from rich.prompt import IntPrompt

    return IntPrompt.ask(
        f"{' ' * PADDING_LEFT}{prompt}",
        choices=[str(c) for c in choices],
    )


def confirm(prompt: str, default: bool = False) -> bool:
    """Styled yes/no prompt."""
    from rich.prompt import Confirm

    return Confirm.ask(f"{' ' * PADDING_LEFT}{prompt}", default=default)


def ok(msg: str) -> None:
    """Success line: green checkmark."""
    print_line(f"[green]✓[/green] {msg}")


def fail(msg: str) -> None:
    """Failure line: red cross."""
    print_line(f"[red]✗[/red] {msg}")


def error(msg: str) -> None:
    console.print(Padding(f"{msg}", (1, PADDING_LEFT, 1, PADDING_LEFT), expand=False))


def warn(msg: str) -> None:
    """Warning line: yellow triangle."""
    print_line(f"[yellow]⚠[/yellow] {msg}")


def stub(msg: str = "not yet implemented") -> None:
    """Placeholder for unimplemented steps."""
    print_line(f"[dim yellow]▢[/dim yellow] {msg}")


def status(msg: str) -> None:
    """Status line with custom psilia spinner."""
    return Live(
        Padding(
            Spinner("dots", text=msg, style="bright_magenta"),
            (0, PADDING_LEFT),
            expand=False,
        ),
        console=console,
        transient=True,
        refresh_per_second=12.5,
    )


def info(msg: str) -> None:
    """Indented descriptive text."""
    print_line(f"{msg}")


def fyi(msg: str) -> None:
    print_line(f"[dim](FYI: {msg})[/dim]")


def debug(msg: str, show=True) -> None:
    if not show:
        return
    print_line(f"[dim](DEBUG: {msg})[/dim]")


def item(msg: str) -> None:
    """List item with a dim arrow."""
    print_line(f"[dim]→[/dim]  {msg}", padding_left=PADDING_LEFT)


def detail(key: str, value: str) -> None:
    """Key/value config detail line."""
    print_line(f"[dim]{key}:[/dim]  {value}", padding_left=PADDING_LEFT)


def _dict(d: dict) -> None:
    """Pretty-print a dictionary with rich's pprint."""
    pprint(d, expand_all=True, console=console, indent_guides=False)


def _yaml2(d) -> None:
    console.print(
        Padding(
            _yaml_mod.dump(d, default_flow_style=False),
            (2, 2),
        )
    )


def build_tree(
    d: dict | list,
    label: str = "",
    key_style: str = "cyan",
    value_style: str = "normal",
    label_style: str = "magenta",
    guide_style: str = "dim cyan",
    collapse_flat_lists: bool = False,
    collapse_max: int = 4,
):
    """Build and return a Rich Tree from a nested dict (without printing)."""
    from rich.tree import Tree

    def _add_dict_leaf(node, key, value) -> None:
        node.add(
            f"[{key_style}]{key}:[/{key_style}] [{value_style}]{value}[/{value_style}]"
        )

    def _add_list_leaf(node, value) -> None:
        node.add(f"[{key_style}][/{key_style}][{value_style}]{value}[/{value_style}]")

    def _is_flat(lst: list) -> bool:
        return len(lst) <= collapse_max and all(
            not isinstance(v, (dict, list)) for v in lst
        )

    def _collapse(lst: list) -> str:
        return "[" + ", ".join(str(v) for v in lst) + "]"

    def _add(node, d: dict | list) -> None:
        if isinstance(d, dict):
            for key, value in d.items():
                if isinstance(value, dict) or isinstance(value, list):
                    if (
                        collapse_flat_lists
                        and isinstance(value, list)
                        and _is_flat(value)
                    ):
                        _add_dict_leaf(node, key, _collapse(value))
                    else:
                        branch = node.add(f"[{key_style}]{key}[/{key_style}]")
                        _add(branch, value)
                else:
                    _add_dict_leaf(node, key, value)
        elif isinstance(d, list):
            for value in d:
                if isinstance(value, dict) or isinstance(value, list):
                    if (
                        collapse_flat_lists
                        and isinstance(value, list)
                        and _is_flat(value)
                    ):
                        _add_list_leaf(node, _collapse(value))
                    else:
                        branch = node.add(f"[{key_style}]*[/{key_style}]")
                        _add(branch, value)
                else:
                    _add_list_leaf(node, value)

    # Check if dict of list has any entries.
    if len(d) > 0:
        if isinstance(d, list):
            tree = Tree(
                f"[{label_style}]{label}[/{label_style}]",
                guide_style=guide_style,
            )
        else:
            tree = Tree(
                f"[{label_style}]{label}[/{label_style}]",
                guide_style=guide_style,
            )
        _add(tree, d)
    else:
        tree = f"[{label_style}]{label}[/{label_style}]\n{d}"
    return Padding(tree, (0, 0, 0, PADDING_LEFT))


def print_tree(
    d: dict | list,
    label: str = "",
    key_style: str = "cyan",
    value_style: str = "normal",
    label_style: str = "magenta",
    guide_style: str = "dim cyan",
    collapse_flat_lists: bool = False,
    collapse_max: int = 4,
) -> None:
    """Display a nested dict as a Rich Tree."""
    console.print(
        build_tree(
            d,
            label=label,
            key_style=key_style,
            value_style=value_style,
            label_style=label_style,
            guide_style=guide_style,
            collapse_flat_lists=collapse_flat_lists,
            collapse_max=collapse_max,
        )
    )


def print_yaml(d: dict, key_color="") -> None:
    """YAML-like dict printer with bold keys (Rich markup)."""

    def _collect(d: dict, indent: int) -> list[str]:
        lines = []
        pad = "\t" * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{pad}[bold {key_color}]{key}:[/bold {key_color}]")
                lines.extend(_collect(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{pad}[bold {key_color}]{key}:[/bold {key_color}]")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f"{pad}  -")
                        lines.extend(_collect(item, indent + 2))
                    else:
                        lines.append(f"{pad}  - {item}")
            else:
                lines.append(
                    f"{pad}[bold {key_color}]{key}:[/bold {key_color}] {value}"
                )
        return lines

    console.print("\n".join(_collect(d, 0)))


# ── demo ─────────────────────────────────────────────────────────────────────


def _demo() -> None:
    """Print a fake wizard flow to preview all UI helpers."""

    header(
        ["Device Manager", "Pair with Device"],
        "Pair a Jetson: connect, generate SSH keypair, register device.",
    )

    title("A section header")
    print_line("a simple line of [bold]text[/bold]")
    print_line("another line")
    info("Connecting to the Jetson over SSH. (info)")
    info("Trying [bold]192.168.1.42[/bold]… (info)")
    stub("detect SSD, confirm mount point, configure /etc/fstab (stub)")
    import time

    with status("Connecting to 192.168.1.42… (status)"):
        time.sleep(0.5)
    ok("Connected to 192.168.1.42 (ok)")
    detail("connect with", "ssh psilia-jetson (detail)")
    fail("Failed to connect to 192.168.1.42 (fail)")
    item("Try again with the hotspot IP (item)")
    warn("No USB wifi dongle found — falling back to built-in wifi. (warn)")

    done(
        "psilia-jetson paired.",
        "Run [bold]psilia setup psilia-jetson[/bold] to bootstrap the runtime.",
    )

    config = {
        "device": {
            "name": "psilia-jetson",
            "host": "psilia-jetson.local",
            "user": "nvidia",
        },
        "status": "paired",
        "nested": [1, 2, {"x": [3, 4]}],
    }

    print_tree(config, label="Config")

    r = ask("hello enter", default="testing", password=True)
    print(r)
    print_line(str(type(r)))

    detail("check runtime status", "psilia runtime status [bold]device[/bold]")


    print_tree(
        ["a", "b", {"c": [1, 2, 3]}, "d"],
        label="A list with a dict inside",
    )


if __name__ == "__main__":
    _demo()