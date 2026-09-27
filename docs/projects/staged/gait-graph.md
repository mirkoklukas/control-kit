# I. Discrete planning and pose Graph approach 

## TL;DR
Let's assume a quadruped, moving a single leg at a time, always maintaining a stable tripod stance. For the transition between two consequtive tripod stances all 4 legs must be planted. If we discretise (fix the number of) the potential contact sites (footholds) this defines a graph structure. Than one could search. It still remains to check whether two stances are in fact connected in configuration space without any self and environmental collisions.

There is also the dual version, where the fully planted stance is the node, and two of those are connected if they differ only by the placement of a single leg (the swing leg basically). The stance with the lifted leg needs to be stable. This might be the easier viewpoint. 

## Recipe

- Discretize footholds from environment geometry. At each time, given body pose we can filter footholds in rough neighbourhood of the robot, ensuring a reasonably sized (small) set of nearby footholds. 
- Given a full stance (all 4 legs are planted), pick the next swing leg, and sample its new placement. This is in face a two-step process: First pick a new body pose given the tripod stance. From the new body pose pick a reachable foothold. I'd say the more efficient that can be done the better. Also note that each choice (body and foot) should be informed by the given task, e.g. move in a target direction or move your body through a narrow hole. Obviously we need to check validity of the new fully planted pose.
- We can now roll out potential discrete graph trajectories (note we don't need to compute the graph in advance) which we can score and potentially improve. 
- Note that this is only half of the solution; we still need to ensure that the transitions can actually be executed.

## Utils

Incomplete list/draft of methods we need or which would be useful (Claude can complete that):

- Given a body pose and nearby footholds sample a full robot posture. Signature would be something like `key, robot, body, footholds --> ok, posture, support`. Question is if support should just contain the planted legs ids and site (foothold) ids (currently support contains the full site).
- Given a robot posture and nearby footholds re-sample a subset (maybe just one) of the legs. Signature would be something like `key, robot, posture, footholds, ids --> ok, posture, support`.
- Generating footholds from environment geometry. Nearby filter with respect to a position maybe.


## Sim Playground 

It would be great to have an environment with different elements to climb and walk on. And we want it to be an interactive playground: I would like to control a body position similar to how we do it in `lab.controller`. We want to accomodate two things:
1. Position the body where we want and sampling a full posture. 
2. Given a full body posture. Choose a swing leg, re-position the body (note we have to make sure the new body position is still supported by the remaining tripod position), re-sample the leg position. 

In both cases we need to (re-)compute the nearby and reachable footholds for sampling. 

