
# Robotics 101


## Terminology and Notation

### Velocities

- $(x,y,z)$ — Position
- $\theta = (\text{roll}, \text{pitch}, \text{yaw})$ —  Rotation in Euler angles  (not sure what convention intrinsic or extrinsic)
- $v = (\dot x, \dot y, \dot z)$ — Linear velocity
- $\omega = \dot \theta$ — Angular velocity
- ...
### Torque

- **Definition:** Torque is the rotational equivalent of linear force. It is the measure of the turning or twisting force that causes an object to rotate around an axis.
- **Calculation:** It is calculated as the cross product of the lever arm (distance) and the applied force:  
    τ = r × F
- **Units:** Newton-meters (Nm).
- **Role in Robotics:** Torque is what allows a robot's **joint actuators** or **motors** to move robotic arms or turn wheels. [1](https://community.robotshop.com/forum/t/help-on-robot-torque-math/36231), [2](https://www.youtube.com/watch?v=nGNf2jcfuUw), [3](https://www.youtube.com/watch?v=TM6UoHzZI40), [4](https://www.roboticsunveiled.com/robotics-wrenches/), [5](https://www.geeksforgeeks.org/physics/torque/)

### Wrench

- **Definition:** A wrench is a compact 6-dimensional vector used in spatial mechanics to represent all forces and torques acting on a rigid body simultaneously. [1](https://www.youtube.com/watch?v=0wsYPJPGtKE&t=9), [2](https://www.roboticsunveiled.com/robotics-wrenches/)]
- **Representation:** It is typically written as a column vector:  
	$$\mathcal{F} = \begin{bmatrix} \tau \\ F \end{bmatrix}$$
    Here, τ represents the 3-dimensional rotational torque (moments), and F represents the 3-dimensional linear force. [1](https://modernrobotics.northwestern.edu/nu-gm-book-resource/3-4-wrenches/), [2](https://www.roboticsunveiled.com/robotics-wrenches/)]
- **Role in Robotics:** Wrenches are used to calculate static equilibrium and dynamic stability. When a robot interacts with its environment—such as a robotic hand holding an object or pushing against a table—the sum of the forces and torques is mathematically modeled as an interaction **wrench**. [1](https://www.youtube.com/watch?v=0wsYPJPGtKE&t=9), [2](https://modernrobotics.northwestern.edu/nu-gm-book-resource/5-2-statics-of-open-chains/), [3](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2022.892916/full)]

### Twist
...

### Base / Base Link / Link in general
...