# Locomotion Strategies

# I. Discrete planning and pose Graph approach 

## TL;DR

Let's assume a quadruped, moving a single leg at a time, always maintaining a stable tripod stance. For the transition between two consequtive tripod stances all 4 legs must be planted. If we discretise (fix the number of) the potential contact sites (footholds) this defines a graph structure. Than one could search. It still remains to check whether two stances are in fact connected in configuration space without any self and environmental collisions.

There is also the dual version, where the fully planted stance is the node, and two of those are connected if they differ only by the placement of a single leg (the swing leg basically). The stance with the lifted leg needs to be stable. This might be the easier viewpoint. 

> Details moved to [[gait-graph]]

# II. Reinforcement Learning, locomotion policy in simulation

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

For now we want to use MuJoCo. The model contains two things, the robot and the environment geometry and so on.

We might 