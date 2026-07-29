


# 4DoF Leg

Assuming the leg consists of 4 hinges in the following order: Hip-yaw, hip-roll, hip-pitch, knee-pitch. All links are prolonged segments along the x-axis. 

## One point in the flex plane
Let $\theta$ denote the hip-yaw and $\psi$  the hip-roll angles. Then the flexion plane is defined by the span of 
$$
A = \begin{pmatrix} \cos \theta \\ \sin \theta \\ 0\end{pmatrix}
$$
and
$$
B = \cos \psi \begin{pmatrix} -\sin \theta \\ \cos \theta \\ 0\end{pmatrix} +
\sin \psi \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}.
$$
The normal vector of the plane is given by
$$
C = -\sin \psi \begin{pmatrix} -\sin \theta \\ \cos \theta \\ 0\end{pmatrix} +
\cos \psi \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}
= \begin{pmatrix} \sin \psi \sin \theta \\ - \sin \psi \cos \theta \\ \cos \psi \end{pmatrix}.
$$
For a point $p=(x,y,z)$ to lie in the span of $\{A,B\}$ it must satisfy 
$$
\begin{align}
0 &= \langle p, C \rangle \\
  &= \cos \psi \cdot z + \sin \psi\cdot \underbrace{\big( \sin \theta \cdot x - \cos \theta \cdot y \big)}_{=: w = w(p,\theta)} \\
  &= \langle  \begin{pmatrix} \cos \psi \\ \sin \psi\end{pmatrix}, \begin{pmatrix} z \\ w \end{pmatrix} \rangle.
\end{align}
$$
Thus the angle $\psi$ is given by the angle of the point $\begin{pmatrix} z \\ w \end{pmatrix}$ rotated by $\pm \tfrac{\pi}{2}$, i.e. 
$$
\widetilde\psi_\mp = \widetilde\psi_\mp(p, \theta) = \arctan_2 \Big(w(p,\theta), z \Big) \mp \tfrac{\pi}{2}.
$$
Note that in our leg model the zero-roll flex plane is the $xz$-plane, not the $xy$-plane; because of the way the segments are linked. So our model-roll term is actually 
$$
\psi_\mp = \widetilde\psi_\mp - \tfrac{\pi}{2} = \arctan_2 \Big(w(p,\theta), z \Big) \mp \tfrac{\pi}{2} - \tfrac{\pi}{2}.
$$

**Note:** For $z = 0$ we have either $\psi = 0$ and $\theta$ free (flexion plane is the xy-plane), or $\theta = \arctan_2(y,x)$ and $\psi$ free (flexion plane spanned by $p$ and some orthogonal vector given by $\psi$).

--- 
If $z \neq 0$ we can also pack all info about $p$ into $w$ and set
$$
	\widetilde w := \tfrac{w}{z}.
$$
Then we have 
$$
\begin{align}
0 &= \langle p, C \rangle \\
  &= \Big\langle  \begin{pmatrix} \cos \psi \\ \sin \psi\end{pmatrix}, \begin{pmatrix} 1 \\ \widetilde w \end{pmatrix} \Big\rangle
\end{align}
$$
and
$$
\psi = \arctan_2 \Big(\widetilde w, 1 \Big) \pm \tfrac{\pi}{2}.
$$
---
## Two Points in the flex plane
The nice thing about this formulation is, that if we want a second point $p'$ to lie in the plane we have to find a $\theta$ such  that
$$
	\widetilde w (p, \theta) = \widetilde w (p', \theta).
$$
This translates to
$$
\Big\langle  \begin{pmatrix} \sin \theta \\ - \cos \theta \end{pmatrix}, \begin{pmatrix} \tfrac{x}{z} - \tfrac{x'}{z'} \\  \tfrac{y}{z} - \tfrac{y'}{z'}  \end{pmatrix} \Big\rangle = 0,
$$
which is solved by
$$
	\theta = \arctan_2 ( \tfrac{y}{z} - \tfrac{y'}{z'},  \tfrac{x}{z} - \tfrac{x'}{z'} ) \ \text{ and } \ \theta' = \theta + \pi
$$
---
If the z-coordinates of both points is zero, we're in the $\psi=0$ and $\theta$ free case (xy-plane).
If one of the z-coordinates is not zero (without any loss of generality we assume it
s the first point $p$), we set $\theta =\arctan_2(y,x)$. Then $\psi = \arctan_2 \Big(w(p',\theta), 0 \Big) \pm \tfrac{\pi}{2}$.

## Two Points in the flex plane: Simpler version

For two points $p,p'$ to lie in the flexion plane, their cross product $n = p \times p'$  hast to satisfy
$$
\langle A,n \rangle = 0 \ \text{ and } \ \langle B,n \rangle = 0
$$
From the first equation we get
$$
\Big\langle  \begin{pmatrix} \cos \theta \\ \sin \theta \end{pmatrix}, \begin{pmatrix} n_x \\ n_y \end{pmatrix} \Big\rangle = 0,
$$
and thus $\theta = \text{atan}_2(n_y, n_x) - \tfrac{\pi}{2} = \text{atan}_2(-n_x, n_y)$.  From the second we get
$$
\begin{align}
0 &= \cos \psi \cdot \Big\langle  \begin{pmatrix} - \sin \theta \\ \cos \theta \\ 0 \end{pmatrix}, n \Big\rangle + \sin \psi \cdot n_z \\
&= \cos \psi \cdot (\sqrt{n_x^2 + n_y^2}) + \sin \psi \cdot n_z \\
&= \Big\langle  \begin{pmatrix} \cos \psi \\ \sin \psi \end{pmatrix}, \begin{pmatrix} \sqrt{n_x^2 + n_y^2} \\ n_z \end{pmatrix} \Big\rangle
\end{align}
$$
and thus $\psi = \text{atan}_2(n_z, \sqrt{n_x^2 + n_y^2}) - \pi/2$. Note that in our model the zero-roll flex plane is the xz-plane, not the xy-plane. So our model-roll term is actually $\psi' = \psi + \tfrac{\pi}{2} = \text{atan}_2(n_z, \sqrt{n_x^2 + n_y^2})$.

## All branches

Note: $\text{atan}_2(y,x) - \tfrac{\pi}{2} = \text{atan}_2(-x,y)$,  $\text{atan}_2(y,x) + \tfrac{\pi}{2} = \text{atan}_2(x,-y)$

For two points $p,p'$ to lie in the flexion plane, their cross product $n = p \times p'$  hast to satisfy
$$
\langle A,n \rangle = 0 \ \text{ and } \ \langle B,n \rangle = 0.
$$
From the first equation we get
$$
\Big\langle  \begin{pmatrix} \cos \theta \\ \sin \theta \end{pmatrix}, \begin{pmatrix} n_x \\ n_y \end{pmatrix} \Big\rangle = 0,
$$
and thus 
$$
\theta_\mp = \text{atan}_2(n_y, n_x) \mp \tfrac{\pi}{2}.
$$From the second we get
$$
\begin{align}
0 &= \cos \psi \cdot \Big\langle  \begin{pmatrix} - \sin \theta_\mp \\ \cos \theta_\mp \\ 0 \end{pmatrix}, n \Big\rangle + \sin \psi \cdot n_z \\
&= \cos \psi \cdot (\pm \sqrt{n_x^2 + n_y^2}) + \sin \psi \cdot n_z \\
&= \Big\langle  \begin{pmatrix} \cos \psi \\ \sin \psi \end{pmatrix}, \begin{pmatrix} \pm \sqrt{n_x^2 + n_y^2} \\ n_z \end{pmatrix} \Big\rangle,
\end{align}
$$
and thus 
$$
\widetilde \psi (\theta_\mp)_\mp = \text{atan}_2(n_z, \pm \sqrt{n_x^2 + n_y^2}) \mp \tfrac{\pi}{2}.$$Note that in our leg model the zero-roll flex plane is the $xz$-plane, not the $xy$-plane; because of the way the segments are linked. So our model-roll term is actually 
$$
\psi = \widetilde \psi - \tfrac{\pi}{2} = \text{atan}_2(n_z, \pm \sqrt{n_x^2 + n_y^2}) \mp \tfrac{\pi}{2} - \tfrac{\pi}{2}.
$$
 