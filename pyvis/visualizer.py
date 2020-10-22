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
        """ Returns the first counter-clockwise Perpendicular Unit-Vector to self. """
        return Vector(-self.y, self.x) * (1 / self.magnitude)

    def dot(self, other: 'Vector') -> float:
        """ Dot-Product with other vector. """
        return self.x * other.x + self.y + other.y

    def cross(self, other: 'Vector') -> float:
        """ Magnitude of cross-product between self and other. """
        return self.x * other.y - self.y * other.x

    def angle(self, other: 'Vector') -> float:
        """ returns the angle in the counter clockwise direction from other to self. """
        return float(np.arccos(self.dot(other) / (self.magnitude * other.magnitude)))

    def draw(self, anchor: 'Point', ax: Axes):
        tip_x, tip_y = self.x - anchor.x, self.y - anchor.y
        ax.quiver(anchor.x, anchor.y, tip_x, tip_y,
                  scale_units='xy', angles='xy', scale=1, width=0.0025)
        return


class Point:
    def __init__(self, index: int, x: float, y: float, mass: float = 1):
        self._index: int = index
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
    def index(self) -> int:
        return self._index

    @property
    def location(self) -> Tuple[float, float]:
        return self.x, self.y

    @property
    def vector(self) -> Vector:
        return Vector(self.x, self.y)

    def translate(self, vector: Vector) -> Vector:
        return Vector(self.x + vector.x, self.y + vector.y)

    def move(self, force: Vector) -> 'Point':
        self._x = self.x + force.x / self.mass
        self._y = self.y + force.y / self.mass
        return self

    def rotate(self, pivot: 'Point', force: Vector) -> 'Point':
        """ Rotate self around pivot by given force. """
        direction: Vector = self.vector - pivot.vector
        # force = mass * radius * angular_acceleration
        alpha: float = force.magnitude / (self.mass * direction.magnitude)
        # determine direction of force. whether clockwise or counter-clockwise
        alpha *= (1 if direction.cross(force) > 0 else -1)
        theta_0: float = direction.angle(Vector(1, 0))
        theta: float = theta_0 + alpha / 2
        self._x = pivot.x + direction.magnitude * np.cos(theta)
        self._y = pivot.y + direction.magnitude * np.sin(theta)
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

    def direction(self, anchor: int) -> Vector:
        """ Returns a unit direction vector from anchor. """
        if anchor not in [self.left.index, self.right.index]:
            raise ValueError(f'Direction must be anchored at one end of the spring. Got {anchor} instead.')
        left, right = Vector(*self.left.location), Vector(*self.right.location)
        direction = left - right
        return direction * (1 if anchor == self.left.index else -1) * (1 / direction.magnitude)

    def force(self, anchor: int) -> Vector:
        if anchor not in [self.left.index, self.right.index]:
            raise ValueError(f'Direction must be anchored at one end of the spring. Got {anchor} instead.')
        return self.direction(anchor) * -self.stiffness * (self.length - self.rest_length)

    def draw(self, ax: Axes):
        ax.plot([self.left.x, self.right.x], [self.left.y, self.right.y])
        self.left.draw(ax), self.right.draw(ax)
        self.force(self.left.index).draw(self.left, ax), self.force(self.right.index).draw(self.right, ax)
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
        pivot: Point = self.points[cluster.argcenter]

        # high-dim distances for triangle
        [cl, cr] = list(cluster.distance_from([left.argcenter, right.argcenter]))
        [rl] = list(right.distance_from([left.argcenter]))

        if right.argcenter in self.points:
            # Right is the designated new cluster, so swap clusters and distances
            left, right = right, left
            cl, cr = cr, cl
        else:
            pass

        def _add_along_axis():
            # This preserves ratio of high-dimensional distances
            _center: Point = self.points[cluster.argcenter]
            _axis: Spring = self.springs[key(cluster.argcenter, left.argcenter)]
            # Right may be placed between Cluster and Left, or to the right of Cluster, or to the left of Left.
            _ratio: float = cr / ((cr + rl) if cl > max(cr, rl) else (rl - cr) if cr < cl < rl else (cr - rl))
            # translate center in the direction of axis, by the proper ratio of length
            _location: Vector = _center.translate(_axis.direction(_center.index) * _axis.length * _ratio)
            self.points[right.argcenter] = Point(right.argcenter, _location.x, _location.y)
            return

        if (left.argcenter in self.points) and (right.argcenter in self.points):  # neither of the children provides a new point
            left_point, right_point = self.points[left.argcenter], self.points[right.argcenter]

            # add springs between trio of points
            self._add_spring(Spring(self.points[cluster.argcenter], left_point, cl))
            self._add_spring(Spring(self.points[cluster.argcenter], right_point, cr))
            self._add_spring(Spring(left_point, right_point, rl))
            self._add_springs(left), self._add_springs(right)

            self.draw(fig, text='no new points added')
            return
        elif (left.argcenter in self.points) or (right.argcenter in self.points):  # only one of left/right is a new point
            if cluster.argcenter == right.argcenter:  # cluster center is the same as a Right center
                # rigid rod connects cluster and right and rotates about cluster. solve dynamical system.
                # pick starting point for Right
                point = self.points[right.argcenter] = Point(right.argcenter, pivot.x, pivot.y + cr)
                self._add_springs(left), self._add_springs(right)
                # add Springs from Right
                springs: List[Spring] = self._springs_from(point)
                rod: Spring = self.springs[key(pivot.index, point.index)]
                while True and point.mass < 1e3:  # rotate rod until moment is zero
                    force: Vector = sum((spring.force(point.index) for spring in springs))
                    if rod.direction(rod.left.index).cross(force) < 1e-3:
                        # a force that is parallel or anti-parallel to the rod will produce no rotation
                        break
                    else:
                        # rotate by some angle due to moment
                        point.rotate(pivot, force)
                        # increase mass to dampen next rotation magnitude
                        point.mass += 1
                point.mass = 1

                self.draw(fig, text=f'added and rotated rod with pivot {pivot.index} and point {right.argcenter}')
                # TODO: 3-d rotational dynamics of rigid bar rotating about one end
                return
            else:  # a child has same center as a non-parent ancestor
                if 2 * max(cl, cr, rl) < sum((cl, cr, rl)):  # triangle inequality holds.
                    # find all points that would imply springs from right
                    neighbors: List[Tuple[Point, float]] = [
                        (self.points[candidate.argcenter], distance) for candidate, distance in right.candidates.items()
                        if candidate.argcenter in self.points and distance <= right.radius + candidate.radius
                    ]

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
                        forces.append(sum((spring.force(spring.right.index) for spring in springs)).magnitude)

                    xr, yr = solutions[int(np.argmin(forces))]

                    self.points[right.argcenter] = Point(right.argcenter, xr, yr)
                    self._add_springs(left), self._add_springs(right)

                    self.draw(fig, text=f'added new point {right.argcenter}, triangle inequality holds')
                    # TODO: 3-D rotate triangle about axis
                    return
                else:  # triangle inequality is broken.
                    # Add Right somewhere along the axis of the pivots.
                    # low-dim distances to each pivot should preserve ratio of high-dim distances to each pivot
                    _add_along_axis()
                    self._add_springs(left), self._add_springs(right)
                    self.draw(fig, text=f'one new point added: {right.argcenter}, triangle inequality was broken.')
                    return
        else:
            # left and right both provide new points
            # for now, solve 2-d rotational dynamics of triangle

            if rl == cl + cr:  # triangle inequality was broken
                _add_along_axis()
                self.draw(fig, text=f'new triangle added: {right.argcenter}, triangle inequality was broken.')
            else:  # triangle inequality holds
                # triangle will rotate about pivot until net torque is zero

                # add new points
                cos_theta: float = (cl ** 2 + cr ** 2 - rl ** 2) / (2 * cl * cr)
                sin_theta: float = (1 - cos_theta ** 2) ** 0.5
                right_point: Point = Point(right.argcenter, pivot.x + cr, pivot.y)
                left_point: Point = Point(left.argcenter, pivot.x + cl * cos_theta, pivot.y + cl * sin_theta)
                self.points.update({left.argcenter: left_point, right.argcenter: right_point})
                self._add_springs(left), self._add_springs(right)

                while True:
                    cl_vec: Vector = left_point.vector - pivot.vector
                    force_left: Vector = sum((spring.force(left_point.index) for spring in self._springs_from(left_point)))
                    moment_left: float = cl_vec.cross(force_left)

                    cr_vec: Vector = right_point.vector - pivot.vector
                    force_right: Vector = sum((spring.force(right_point.index) for spring in self._springs_from(right_point)))
                    moment_right: float = cr_vec.cross(force_right)

                    if moment_left + moment_right <= 0 or right_point.mass < 1e3:
                        break
                    else:
                        if __name__ == '__main__':
                            force: float = (moment_left + moment_right) / cr_vec.magnitude
                            exit(1)

                # figure out translation of force from acting on left point to acting on the right point
                # rotate right point using the resultant force, then recover left point
                # loop until net torque after translation is nearly zero
                self.draw(fig, text=f'two new points added: {left.argcenter}, {right.argcenter}. Triangle inequality holds')

            # TODO: 3-D solve rotational dynamics of rigid triangle anchored at one vertex
            # https://math.stackexchange.com/questions/871867/rotation-matrix-of-triangle-in-3d
            # https://en.wikipedia.org/wiki/Rodrigues'_rotation_formula

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
        # loop until resultants are nearly zro
        pass

    def force_direct(self, graph: Graph, *, fig: Optional[Figure] = None):
        # TODO: figure out need for step-size
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
