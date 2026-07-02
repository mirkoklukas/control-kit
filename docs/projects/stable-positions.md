# Stability

Find "all" stable tripod positions for a hexapod; maybe  define a rest positions for the remaining legs. Then the configuration space is 9-dimensional (3 legs with 3 DOF) for a fixed set of three free legs (there are 6 choose 3 free leg groups; 20); we "could" randomly position the 3 free legs, their feet define a plane. check if it is below the body, let that plane be the ground (xy-plane) and place the robot accordingly. We could then apply random forces etc. to see how stable it is. We can do the same with changing the gravity vector, testing for stable climbing positions. 

From here we can also explore, given a stable position what are the ways I can move my body given the stable foot placement constraint; e.g. can I pull myself up to place my free legs or reach/enable/prepare the next stable position. 

One Idea would be to build an "alphabet" of stable positions. Then form a graph where the positions are the vertices and they are connected by an edge if you can stably move from one to the other; given some constraints, feet on a common floor plane or so. Then we can search in that space easily. 

What are ways to generate robot poses given some constraints (e.g. given planted feet); it should be quick. 

## Gait graph

**Strategy 1:**
Given: (Current) Foot Placement and body pose. 
Recipe:
- Sample potential foot placements of the free feet. (this could be conditioned on something)
- Sample potential body poses (could also be conditioned on something)
- Pick a new subset of placed feet. 
- Check each body pose is *valid* -- reachable **and** stable -- under both the old
  and the new placement. (reachable = every planted leg can connect to its foot;
  stable = CoM projects into the support polygon with margin.)

A body pose valid under both placements is an edge. The CoM-in-polygon stability
region is convex, so we can slide the body from the old pose to this one staying
stable on the old feet, then swap feet -- existence of a mutually-valid pose implies
a quasi-static transition, no path search needed. This is conservative (it also
requires stability under the old placement); a looser variant places the new feet
first and uses the union (all-down) support polygon as the corridor.

Future: also check the combined old+new configuration is physically feasible --
legs/body not passing through each other (self-collision).

## Footplacements

In our simulated world, let's cover the surface near the robot with tiny spheres at a given coarseness. The are imaginary foot targets, no physics apply to them. From those we can sample new foot placements. I'll call them foot grid, although it's not *really* a grid. A foot placement sampler can then specify a region and we'll sample from the intersection of that region with the grid; sort of. 

```
ok, here's what i want to generate potential foot placements. assume we have a current pose. In our simulated world, let's cover the surface near the robot with tiny spheres at a given coarseness. The are imaginary foot targets, no physics apply to them. From those we can sample new foot placements. I'll call them foot grid, although it's not really a grid. A foot placement sampler can then specify a region and we'll sample from the intersection of that region with the grid; sort of. let's make the world a floor and a big box of dimensions (xyz) 4x4x2 placed in front of the robot so its facing the yz wall of the box. does all this make sense?

ok, let's keep it "simple", first step is points on the surface mesh. we can consider only the part of the mesh that is within distance MESH_DISTANCE to the body. Then uniformly sample N positions on that mesh. place visual markers at these points. and let me look at it.
```

- next. for each leg the points by if they are reachable at all. so now each leg as a superset of candidates. we should have a boolean array (N,6,3) indicating for each leg if the points in the superset of candidates (N,3) are reachable.  