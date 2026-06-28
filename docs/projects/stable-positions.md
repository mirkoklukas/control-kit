# Stability

Find "all" stable tripod positions for a hexapod; maybe  define a rest positions for the remaining legs. Then the configuration space is 9-dimensional (3 legs with 3 DOF) for a fixed set of three free legs (there are 6 choose 3 free leg groups; 20); we "could" randomly position the 3 free legs, their feet define a plane. check if it is below the body, let that plane be the ground (xy-plane) and place the robot accordingly. We could then apply random forces etc. to see how stable it is. We can do the same with changing the gravity vector, testing for stable climbing positions. 

From here we can also explore, given a stable position what are the ways I can move my body given the stable foot placement constraint; e.g. can I pull myself up to place my free legs or reach/enable/prepare the next stable position. 

