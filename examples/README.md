# Examples

## 01 — cartpole swing-up (MPPI)

```bash
# remote (GPU box)
uv run python examples/01_mpc_cartpole.py record --out runs/cartpole.npz


# locally (on Laptop)
# ~2 min, CPU
uv run python examples/01_mpc_cartpole.py record                       
# viewer
uv run mjpython examples/01_mpc_cartpole.py play runs/cartpole.npz
```

## 02 — hexapod forward walking (MPPI)

Same record/play split. The hexapod is much heavier per tick (18 position
servos + contacts), so the default H/N are GPU-shaped; `--steps/--samples/--horizon`
let you run a quick local smoke test.

```bash
# remote (GPU box)
uv run python examples/02_mpc_hexapod.py record --out runs/hexapod.npz

# locally: quick smoke test (small, just checks it runs)
uv run python examples/02_mpc_hexapod.py record --steps 8 --samples 16 --horizon 12

# viewer
uv run mjpython examples/02_mpc_hexapod.py play runs/hexapod.npz
```

## Pulling results back from the GPU box

`record` runs headless on the remote box and writes the `.npz`; `play` needs a
display, so it runs locally. Copy the trajectory down (assumes an `ssh lambda`
alias in `~/.ssh/config`; run from the local repo root so it lands in `runs/`):

```bash
# single file
scp lambda:~/control-kit/runs/hexapod.npz runs/hexapod.npz

# or sync the whole runs/ dir (skips unchanged bytes; good for repeated pulls)
rsync -avz lambda:~/control-kit/runs/ runs/
```

Adjust the remote path if the repo lives somewhere other than `~/control-kit`.
