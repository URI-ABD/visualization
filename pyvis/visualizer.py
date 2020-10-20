import time
from typing import Tuple, Set, Optional, Dict, List
import numpy as np
import streamlit as st
from matplotlib.pyplot import Figure, Axes
from pyclam import Manifold, Graph, Cluster
import sympy

from pyvis.utils import key


class Vector:
    def __init__(self, x: float, y: float):
        self._x, self._y = x, y

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    def __str__(self):
        return f'({self.x:.2f}, {self.y:.2f})'

    def __repr__(self):
        return str(self)
    
    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y)

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return self + (-other)

    def __mul__(self, scalar: float) -> 'Vector':
        """ multiply with scalar. """
        return Vector(self.x * scalar, self.y * scalar)

    @property
    def magnitude(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

    @property
    def perpendicular(self) -> 'Vector':
        """ Returns a Perpendicular Unit-Vector to self. """
        return Vector(self.y, -self.x) * (1 / self.magnitude)

    def dot(self, other: 'Vector') -> float:
        """ Dot-Product with other vector. """
        return self.x * other.x + self.y + other.y

    def draw(self, anchor: 'Point', ax: Axes):
        tip_x, tip_y = self.x - anchor.x, self.y - anchor.y
        ax.quiver(anchor.x, anchor.y, tip_x, tip_y,
                  scale_units='xy', angles='xy', scale=1, width=0.0025)
        return


class Point:
    def __init__(self, index: int, x: float, y: float, mass: float = 1):
        self.index: int = index
        self._x, self._y = x, y
        self.mass: float = mass

    def __eq__(self, other: 'Point'):
        return self.index == other.index

    def __str__(self):
        return f'({self.x:.2f}, {self.y:.2f})'

    def __repr__(self):
        return f'index: {self.index}, location: {str(self)}, mass: {self.mass}'

    def __hash__(self):
        return self.index

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def key(self) -> int:
        return self.index

    @property
    def location(self) -> Tuple[float, float]:
        return self.x, self.y

    def translate(self, vector: Vector) -> Vector:
        return Vector(self.x + vector.x, self.y + vector.y)

    def move(self, force: Vector) -> 'Point':
        self._x = self.x + force.x / self.mass
        self._y = self.y + force.y / self.mass
        return self

    def difference(self, other: 'Point') -> Tuple[float, float]:
        return self.x - other.x, self.y - other.y

    def distance(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def draw(self, ax: Axes):
        ax.scatter([self.x], [self.y])
        ax.annotate(self.index, (self.x, self.y))
        return


class Spring:
    def __init__(self, left: Point, right: Point, rest_length: float, stiffness: float = 1):
        if left == right:
            raise ValueError(f'left and right must be different points.')
        self.left, self.right = left, right
        self.rest_length: float = rest_length
        self.stiffness: float = stiffness

    def __str__(self):
        return f'{self.left}, {self.right}, {self.rest_length:.2f}, {self.stiffness}'

    def __repr__(self):
        return f'left: {self.left}, right: {self.right}, rest length: {self.rest_length}, stiffness: {self.stiffness}'

    def __hash__(self):
        return self.key()

    def key(self) -> int:
        return key(self.left.index, self.right.index)

    @property
    def length(self) -> float:
        return self.left.distance(self.right)

    def direction(self, anchor: str = 'left') -> Vector:
        """ Returns a unit direction vector from anchor. """
        if anchor not in ['left', 'right']:
            raise ValueError(f'Direction must be anchored at \'left\' or \'right\'. Got {anchor} instead.')
        left, right = Vector(*self.left.location), Vector(*self.right.location)
        direction = left - right
        return direction * (1 if anchor == 'left' else -1) * (1 / direction.magnitude)

    def force(self, anchor: str = 'left') -> Vector:
        if anchor not in ['left', 'right']:
            raise ValueError(f'Force must be anchored at \'left\' or \'right\'. Got {anchor} instead.')
        return self.direction(anchor) * -self.stiffness * (self.length - self.rest_length)

    def draw(self, ax: Axes):
        ax.plot([self.left.x, self.right.x], [self.left.y, self.right.y])
        self.left.draw(ax), self.right.draw(ax)
        self.force('left').draw(self.left, ax), self.force('right').draw(self.right, ax)
        return


class Visualizer:
    def __init__(self, manifold: Manifold):
        self.manifold: Manifold = manifold
        self.root: Cluster = manifold.root
        self.points: Dict[int, Point] = dict()
        self.springs: Dict[int, Spring] = dict()
        self.clusters: Set[Cluster] = set()

    def _springs_from(self, point: Point) -> List[Spring]:
        """ Returns a list of those springs that contain point as one end. """
        return [self.springs[k] for other in self.points if (k := key(point.index, other)) in self.springs]

    def _add_spring(self, spring: Spring):
        if spring.key() not in self.springs:
            self.springs[spring.key()] = spring
        return

    def _add_springs(self, cluster: Cluster):
        # This assumes that cluster is already in self.points
        [self._add_spring(Spring(self.points[cluster.argcenter], self.points[candidate.argcenter], distance))
         for candidate, distance in cluster.candidates.items()
         if candidate.argcenter in self.points and distance <= cluster.radius + candidate.radius]
        return

    def _add_triangle(self, cluster: Cluster, fig: Optional[Figure] = None):
        # cluster must already by in the visualized set
        if cluster.argcenter not in self.points:
            raise ValueError(f'when adding a triangle, the expanding cluster must already be in the visualized set.')

        # build triangle with vertices that are cluster.center, left.center, right.center
        left: Cluster
        right: Cluster
        [left, right] = list(cluster.children)

        # high-dim distances for triangle
        [cl, cr] = list(cluster.distance_from([left.argcenter, right.argcenter]))
        [rl] = list(right.distance_from([left.argcenter]))

        if (left.argcenter in self.points) and (right.argcenter in self.points):  # neither of the children provides a new point
            left_point, right_point = self.points[left.argcenter], self.points[right.argcenter]

            # add springs between trio of points
            self._add_spring(Spring(self.points[cluster.argcenter], left_point, cl))
            self._add_spring(Spring(self.points[cluster.argcenter], right_point, cr))
            self._add_spring(Spring(left_point, right_point, rl))

            [self._add_springs(child) for child in cluster.children]

            self.draw(fig, text='no new points added')

        elif (left.argcenter in self.points) or (right.argcenter in self.points):  # only one of left/right is a new point
            if right.argcenter in self.points:
                # right child is the designated new cluster, so swap clusters and distances
                left, right = right, left
                cl, cr = cr, cl

            neighbors: List[Tuple[Point, float]] = [
                (self.points[candidate.argcenter], distance) for candidate, distance in right.candidates.items()
                if candidate.argcenter in self.points and distance <= right.radius + candidate.radius
            ]

            if cluster.argcenter == right.argcenter:  # cluster center is the same as a child center
                # TODO: 2-d rotational dynamics of rigid bar rotating about one end
                # rigid rod connects cluster and right. solve dynamical system.
                # TODO: 3-d rotational dynamics of rigid bar rotating about one end
                pass

            triangle: List[float] = list(sorted((cl, cr, rl)))
            if triangle[2] < triangle[1] + triangle[0]:  # triangle inequality holds.
                # TODO: 3-D rotate triangle about axis

                # solve for two solutions for where to place new point
                (xc, yc), (xl, yl) = self.points[cluster.argcenter].location, self.points[left.argcenter].location
                x, y = sympy.symbols('x y')
                eq1 = sympy.Eq((x - xc) ** 2 + (y - yc) ** 2, cr ** 2)
                eq2 = sympy.Eq((x - xl) ** 2 + (y - yl) ** 2, rl ** 2)
                solutions: List[Tuple[float, float]] = [
                    (sx, sy) for sx, sy in sympy.solve((eq1, eq2), (x, y))
                    if not (isinstance(sx, sympy.Mul) or isinstance(sy, sympy.Mul))
                ]

                assert len(solutions) == 2, 'did not find exactly two real-valued solutions for triangle inequality case'

                # Find the solution which gives the least net force
                forces: List[float] = list()
                for xr, yr in solutions:
                    new_point = Point(right.argcenter, xr, yr)
                    springs: List[Spring] = [Spring(new_point, neighbor, distance) for neighbor, distance in neighbors]
                    forces.append(sum((spring.force('right') for spring in springs)).magnitude)

                xr, yr = solutions[int(np.argmin(forces))]
                self.points[right.argcenter] = Point(right.argcenter, xr, yr)
                [self._add_spring(Spring(self.points[right.argcenter], neighbor, distance)) for neighbor, distance in neighbors]

            else:  # triangle inequality is broken.
                # TODO: Find out whether right should be between cluster and left, or off to one side
                # Add new point somewhere along the axis of the pivots.
                # low-dim distances to each pivot should preserve ratio of high-dim distances to each pivot
                if cl > max(cr, rl):
                    axis: Spring = self.springs[key(cluster.argcenter, left.argcenter)]
                    location: Vector = axis.left.translate(axis.direction('left') * cl * cr / rl)
                    self.points[right.argcenter] = Point(right.argcenter, location.x, location.y)
                elif cr > max(cl, rl):
                    pass
                else:
                    pass

            self.draw(fig, text=f'one new point added: {self.points[right.argcenter]}')

        else:
            # left and right both provide new points
            pivot: Point = self.points[cluster.argcenter]
            # for now, solve 2-d rotational dynamics of triangle
            # triangle will rotate about pivot until net torque is zero
            # figure out translation of force from acting on left point to acting on the right point
            # rotate right point using the resultant force, then recover left point
            # loop until net torque after translation is nearly zero

            # TODO: 3-D solve rotational dynamics of rigid triangle anchored at one vertex
            # https://math.stackexchange.com/questions/871867/rotation-matrix-of-triangle-in-3d
            # https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
            self.draw(fig, text=f'two new points added: {left.argcenter}, {right.argcenter}')

        return

    def _local_optimization(self, cluster: Cluster, path_length: int, fig: Optional[Figure] = None):
        # find all points within path_length of children of cluster
        # these points are the active points
        # determine proper step_size for 1 unit of force
        # find and aggregate forces for each active point and move each active point by the resultant force
        # loop until all resultants are nearly zero
        pass

    def _global_optimize(self, fig: Optional[Figure] = None):
        # from each spring, aggregate the forces on each point.
        # determine proper step_size for 1 unit of force
        # then move all points by the resultant force
        # loop until resultants are nearly zero
        pass

    def force_direct(self, graph: Graph, *, fig: Optional[Figure] = None):
        # TODO: figure out need for step-sizez
        # TODO: think about any need for momentum and decay factor on momentum
        # breadth-first expansion from root, replacing each cluster with a triangle of cluster-left-right.
        self.clusters = {self.root}
        for depth in range(graph.depth):
            clusters: Set[Cluster] = {cluster for cluster in self.clusters if cluster not in graph}
            for cluster in clusters:
                # for each cluster, replace with triangle and solve local optimization
                self._add_triangle(cluster, fig)
                # return
                # after replacing each triangle, perform some local optimizations from replaced cluster.
                [self._local_optimization(cluster, d, fig) for d in range(1, depth + 1)]
            # after replacing each layer, perform global optimization for some number of steps.
            self._global_optimize(fig)
        pass

    def draw(self, fig: Figure, *, text: Optional[str] = None) -> Figure:
        fig.clf()
        ax = fig.add_subplot(111)
        [spring.draw(ax) for spring in self.springs.values()]
        with st.empty():
            if text is not None:
                st.write(text)
            st.write(fig)
            time.sleep(1)
        return fig
