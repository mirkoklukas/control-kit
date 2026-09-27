# Reinforcement Learning, locomotion policy in simulation

## TL;DR

- For now we want to use MuJoCo.
- For modeling the foot see `magnetic-foot-sim.md`.
- We got 3 main milestones/goals. 
    - locomotion on flat ground; avoiding obstacles. stepping over, and moving around. 
    - Transition to a wall and back; and climb up and down the wall.
    - Climb through an opening (rectangular, maybe oval) in the wall. 
- The main thing we need is the correct model including adhesion or dynamically welding the feet in place.  
- We need to think about domain randomization and a curriculum.
- For the flat-to-wall transition, maybe we could start with walking up onto a tilted surface first, then increase the tilt/angle till it is vertical? 

## Sim

For now we want to use MuJoCo. Note, locally we have a MacBook Pro M3 Max. Later we can run it on a remote machine with NVIDIA GPU and CUDA. 

A mujoco model contains two things, the robot and the environment geometry and so on. Can we separate that?

I think the first goal is to set up the RL environment. We should have some handwritten automatic movement for testing. Maybe a simple transition or two. And as a start just sitting on the ground and lifting up to a standing position. 

These two keyframes should have names. maybe home and standing, or rest and standing. or something like that. 