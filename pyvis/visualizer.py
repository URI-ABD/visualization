import time
from typing import Tuple, Set, Optional, Dict, List

import streamlit as st
from matplotlib.pyplot import Figure, Axes
from pyclam import Manifold, Graph, Cluster

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
        pivot = self.points[cluster.argcenter]

        # build triangle with vertices that are cluster.center, left.center, right.center
        left_child: Cluster
        right_child: Cluster
        [left_child, right_child] = list(cluster.children)
        [pl_distance, pr_distance] = list(cluster.distance_from([left_child.argcenter, right_child.argcenter]))
        [lr_distance] = list(left_child.distance_from([right_child.argcenter]))

        if (left_child.argcenter in self.points) and (right_child.argcenter in self.points):
            # neither of the children provides a new point
            left_point, right_point = self.points[left_child.argcenter], self.points[right_child.argcenter]

            # add springs between trio of points
            self._add_spring(Spring(pivot, left_point, pl_distance))
            self._add_spring(Spring(pivot, right_point, pr_distance))
            self._add_spring(Spring(left_point, right_point, lr_distance))

            [self._add_springs(child) for child in cluster.children]

            self.draw(fig, text='no new points added')

        elif (left_child.argcenter in self.points) or (right_child.argcenter in self.points):
            # only one of left/right is a new point,
            left_pivot, right_pivot, new_point = pivot, self.points[left_child.argcenter], self.points[right_child.argcenter]
            if right_child.argcenter in self.points:
                right_pivot, new_point = new_point, right_pivot

            axis: Spring = self.springs[key(left_pivot.index, right_pivot.index)]
            triangle_sides = (axis.length, new_point.distance(left_pivot), new_point.distance(right_pivot))
            if triangle_sides[0] < triangle_sides[1] + triangle_sides[2]:
                # triangle inequality holds.
                # add new point and rotate triangle about axis
                pass
            else:
                # triangle inequality is broken.
                # Add new point somewhere along the axis in-between the pivots.
                pass
            self.draw(fig, text=f'one new point added: {new_point}')

        else:
            # left and right both provide new points
            # for now, solve 2-d rotational dynamics of triangle

            # later, solve rotational dynamics of 3-d rigid triangle anchored at one vertex
            # https://math.stackexchange.com/questions/871867/rotation-matrix-of-triangle-in-3d
            # https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula
            self.draw(fig, text=f'two new points added: {left_child.argcenter}, {right_child.argcenter}')

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
