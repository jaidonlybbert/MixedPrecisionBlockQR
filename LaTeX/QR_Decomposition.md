---
author:
- Jaidon Lybbert
bibliography:
- refs.bib
date: 2023-01-24
title: Mixed-precision Block QR Decomposition on GPU
---

# QR Decomposition

#### 

The QR decomposition of an m-by-n matrix $A$ with $m>n$, is the matrix
product $A = QR$, where $Q$ is an m-by-n unitary matrix, and $R$ is
upper triangular.

## Matrix Q

#### 

The matrix $Q$ is a transformation which preserves inner products of
column vectors of $R$. If the inner product space is real, the matrix
$Q$ is equivalently orthogonal. One possibility of such a transformation
is a rotation.

#### 

Another possibility of such an orthogonal transformation is a
reflection. The matrix $Q$ in general is a combination of rotations and
reflections.

## Matrix R

#### 

The matrix $R$ is upper triangular, a form which has the following
useful properties: (I) the determinant is equal to the product of the
diagonal elements, (II) the eigenvalues are equal to the diagonal
elements, (III) given the linear system $Rx = b$ it is easy to solve for
$x$ by back substitution.

# Computation

#### 

In order to compute the decomposition of $A$, the matrix is iteratively
transformed by unitary matrices $\{U_i : 0 < i < k\}$ until the product
is upper triangular. This upper triangular matrix is the matrix $R$ in
$A = QR$ $$R = U_kU_{k-i} \dots U_1A.$$

#### 

It follows, that the matrix $Q$ is composed of the set of inverse
transformations $$Q = U_{1}^{T}U_{2}^{T} \dots U_{k}^{T}.$$

#### 

The key to solving for $R$ is to choose transformations $U_i$ which
produce zeros below the diagonal of the matrix product
$$P = U_{i} \dots U_1A,$$ and can iteratively be applied to converge to
$R$ as quickly as possible. Two choices for $U_i$ are Householder
reflections, and Givens rotations.

## Householder Reflections

#### 

The Householder reflection is a unitary transformation represented by a
matrix $H\in\mathbb{R}^{N\times{}N}$ which reflects a vector
$x\in\mathbb{R}^N$ across a hyperplane defined by its unit normal vector
$\{v\in\mathbb{R}^N: \|v\|=1\}$. The transformation matrix is given by
$$H = I - 2vv^T$$ where $I\in\mathbb{R}^{N\times{}N}$ is the identity
matrix. [@bhaskar86]

## Givens Rotations

#### 

A Givens rotation is a unitary transformation which rotates a vector $x$
counter-clockwise in a chosen plane. For example, possible Givens
rotation matrices in $\mathbb{R}^4$ include $$\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & c & -s & 0\\
0 & s & c & 0\\
0 & 0 & 0 & 1
\end{bmatrix},
\begin{bmatrix}
c & -s & 0 & 0\\
s & c & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & 1
\end{bmatrix}, or
\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & c & -s\\
0 & 0 & s & c
\end{bmatrix},$$ where $c = \cos{\theta}$ and $s = \sin{\theta}$. Each
of these examples have the effect of rotating the vector in different
planes.

#### 

The Givens rotation is parallelizable because it effects only two
dimensions of the input vector. For example, the second and last
transformations above could simultaneously be computed on the vector
$x$, then the results combined by selecting the first 2 dimensions of
the first result, and the second two dimensions of the second result.

#### 

A Givens rotation can easily be computed to introduce zeros in the
matrix $P$. The scalars $c$ and $s$ can be computed directly from
elements in $P$ in order to zero out targeted elements. For example, say
we want to zero out element $a_{21}$ in the matrix $$P = 
\begin{bmatrix}
a_{11} & a_{12} & a_{13}\\
a_{21} & a_{22} & a_{23}\\
a_{31} & a_{32} & a_{33}
\end{bmatrix}.$$

#### 

We target the second dimension of the column vector, so we rotate on the
plane spanned by the first two dimensions. We don't choose the plane
spanned by the second and third dimensions, because we would end up
losing the zero in the third row in the process. The Givens rotation to
rotate on this plane is of the form $$G = 
\begin{bmatrix}
c & -s & 0\\
s & c & 0\\
0 & 0 & 1
\end{bmatrix}$$ which will leave the third row of $P$ unmodified. We are
aligning the column vector with the axis of the first dimension, making
the component of the vector along the second dimension zero. Below is a
geometric illustration of the rotation.

<figure>
<img src="Givens1" style="width:75mm" />
<figcaption>Geometric illustration of the rotation of a vector in <span
class="math inline">ℝ<sup>3</sup></span> about the axis of basis vector
<span class="math inline"><em>x</em>3</span> to align with the basis
vector <span class="math inline"><em>x</em>1</span>. The result of this
transformation is that the component of the transformed vector in the
direction of the basis vector <span
class="math inline"><em>x</em>2</span> is zero, corresponding to a zero
introduced in the transformed matrix.</figcaption>
</figure>

#### 

The scalars $c$ and $s$ of matrix $G$ are computed directly from the
values in matrix P by the equations $$c = \frac{a_{11}}{r},$$
$$s = -\frac{a_{21}}{r},$$ where $$r=\sqrt{a_{11}^2 + a_{21}^2}$$ The
transformation to introduce the zero is then $$P = GP_{prior} =
\begin{bmatrix}
c & -s & 0\\
s & c & 0\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
a_{11} & a_{12} & a_{13}\\
a_{21} & a_{22} & a_{23}\\
a_{31} & a_{32} & a_{33}
\end{bmatrix}$$

$$P = GP_{prior} =
\begin{bmatrix}
a_{11}/r & a_{21}/r & 0\\
-a_{21}/r & a_{11}/r & 0\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
a_{11} & a_{12} & a_{13}\\
a_{21} & a_{22} & a_{23}\\
a_{31} & a_{32} & a_{33}
\end{bmatrix}$$ where . Multiplying through $$P = GP_{prior} =
\begin{bmatrix}
a_{11}/r & a_{21}/r & 0\\
-a_{21}/r & a_{11}/r & 0\\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
a_{11} & a_{12} & a_{13}\\
a_{21} & a_{22} & a_{23}\\
a_{31} & a_{32} & a_{33}
\end{bmatrix}
=
\begin{bmatrix}
\frac{a_{11}a_{11} + a_{21}a_{21}}{r} & \frac{a_{11}a_{12} + a_{21}a_{22}}{r} & \frac{a_{11}a_{13} + a_{21}a_{23}}{r}\\
-\frac{a_{21}a_{21}+ a_{11}a_{}{r} & -2.4 & 3.1\\
0 & 4 & 3
\end{bmatrix}$$ the zero is introduced in the desired location.
