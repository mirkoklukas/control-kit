
**TL;DR** For testing it is very useful to have a policy that moves between two given tripod stances or between two adjacent full stances. Maybe this is two versions actually. 

There are different versions of that. 

1. Given two stances make it super quick to train a policy to just do the one transition. When ready (after training) run it and read out torques and so on and play the final transition in MuJoCo. 
2. Train a general policy that takes two given stances and moves from one to the other. This might require more thoughts on training data coverage.


**Dependency:** RL environment 