
# Task & Env Design Proposals

- slow down the actuators. 
- applying random forces to body
- ensure stable body, upright
- randomize the slope of the ground, 0 - 180 degrees. Maybe hat to start with full magnetized feet. 
- start with a lighter body and robot in general. 
- start with less gravity.
- walk up to and transition onto a sloped ground. randomize the slope from 0-90 degrees. 
- Complete the design of the environment with magnetic feet. Ensure that only one foot can be moved, and 3 remain planted at all times. 
- termination on fewer than 3 legs planted?

# What the env needs to know

- foot contact array, (4,) boolean, anything touches the ground
	- per pad contacts, (4,N,N)
- foot forces array (4,3)
	- per pad forces, (4,N,N, 3)
- Trunk contact, bool
- foot positions, velocities


# Naming: sim step vs. control step

Two kinds of step, two words (use them in code, comments and talk):

- **sim step**: one `mj_step`, length `m.opt.timestep` (0.002 s). Only exists in simulation.
- **control step** (short: ctl / ctrl): one action of the controller, held for several sim
  steps (0.02 s at 50 Hz). Carries over to the real robot's control loop; a scripted
  controller has the same step. Not "env / gym / rl step".

Possible names that follow (not renamed yet): `sim_dt` (config `timestep`), `ctl_dt` (`dt`),
`ctl_hz` (`policy_hz`), `n_substeps` (`n_sub`, sim steps per control step),
`episode_steps` / `max_episode_steps` (`steps` / `max_steps`, in control steps).

Careful with `ctrl`: in MuJoCo `d.ctrl` is the actuator input array (servo targets,
magnet strength) -- `ctl` avoids the clash in names; `ctrl` fine in prose.

Training counts (`PolicyCfg.steps`, `N_STEPS`, SB3's `num_timesteps`) are control steps
too, summed over envs -- open whether to rename them.