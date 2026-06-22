# Examples

```bash
# remote (GPU box)
uv run python examples/01_mpc_cartpole.py record --out runs/cartpole.npz


# locally (on Laptop)
# ~2 min, CPU
uv run python examples/01_mpc_cartpole.py record                       
# viewer
uv run mjpython examples/01_mpc_cartpole.py play runs/cartpole.npz
```