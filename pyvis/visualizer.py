from typing import Dict, Set, Union

import numpy as np
from matplotlib import pyplot as plt
from pyclam import Cluster, Graph
from sympy import symbols, Eq, solve
from sympy.core.mul import Mul


class Vector:
    def __init__(self, x: float, y: float):
        self._x, self._y = x, y
        self.cache: Dict[str, any] = dict()

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Vector') -> 'Vector':
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scale: float) -> 'Vector':
        return Vector(self.x * scale, self.y * scale)

    def __neg__(self) -> 'Vector':
        return Vector(-self.x, -self.y)

    @property
    def magnitude(self) -> float:
        if 'magnitude' not in self.cache:
            self.cache['magnitude'] = (self.x ** 2 + self.y ** 2) ** 0.5
        return self.cache['magnitude']

    def unit(self) -> 'Vector':
        return Vector(self.x / self.magnitude, self.y / self.magnitude) if self.magnitude > 0 else Vector(0, 0)


class Spring:
    def __init__(self, source: Cluster, destination: Cluster, length: float, constant: float = 1.):
        self._source, self._destination = source, destination
        self._length, self._constant = length, constant

    @property
    def source(self) -> Cluster:
        return self._source

    @property
    def destination(self) -> Cluster:
        return self._destination

    @property
    def length(self) -> float:
        return self._length

    @property
    def constant(self) -> float:
        return self._constant

    def __contains__(self, item: Cluster) -> bool:
        return item.argcenter == self.source

    def force(self, left: Vector, right: Vector) -> Vector:
        return (left - right).unit() * ((left - right).magnitude - self.length) * self.constant


class Visualizer:
    def __init__(self, root: Cluster, graph: Graph):
        self.root: Cluster = root
        self.graph: Graph = graph

        self.vertices: Set[Cluster] = {root}
        self.locations: Dict[int, Vector] = {root.argcenter: Vector(0, 0)}
        self.springs: Set[Spring] = set()

        self._replaced: Set[Cluster] = set()

    def draw(self, filename: str):
        xy = np.asarray([(self.locations[cluster.argcenter].x, self.locations[cluster.argcenter].y) for cluster in self.graph], dtype=float)

        plt.clf()
        fig = plt.figure(figsize=(16, 10), dpi=200)
        fig.add_subplot(111)
        plt.scatter(xy[:, 0], xy[:, 1], s=4)

        plt.title('getting started')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
        plt.close('all')
        return

    def _replace_with_children(self, pivot: Cluster):
        # TODO: cascade change of Spring to take cluster instead of index of cluster-center
        # TODO: find some other way of managing different clusters having the same center
        [left, right] = list(pivot.children)
        px, py = self.locations[pivot.argcenter].x, self.locations[pivot.argcenter].y
        [pl, pr] = list(pivot.distance_from([left.argcenter, right.argcenter]))
        [lr] = list(left.distance_from(right.argcenter))
        self.springs.add(Spring(left.argcenter, right.argcenter, lr, 1))

        if left.argcenter in self.locations:
            rx, ry = px + pr, py
            self.locations[right.argcenter] = Vector(rx, ry)
            self.springs.add(Spring(pivot.argcenter, right.argcenter, pr, 1))
            self.springs.add(Spring(right.argcenter, pivot.argcenter, pr, 1))
        elif right.argcenter in self.locations:
            lx, ly = px + pl, py
            self.locations[left.argcenter] = Vector(lx, ly)
            self.springs.add(Spring(pivot.argcenter, left.argcenter, pl, 1))
            self.springs.add(Spring(left.argcenter, pivot.argcenter, pl, 1))
        elif (left.argcenter in self.locations) and (right.argcenter in self.locations):
            pass
        else:
            lx, ly = px + pl, py

            x, y = symbols('x'), symbols('y')
            eq1 = Eq((x - px) ** 2 + (y - py) ** 2, pr ** 2)
            eq2 = Eq((x - lx) ** 2 + (y - ly) ** 2, lr ** 2)
            solutions = [(sx, sy) for sx, sy in solve((eq1, eq2), (x, y)) if not (isinstance(sx, Mul) or isinstance(sy, Mul))]

            rx, ry = solutions[0]

            self.locations[left.argcenter] = Vector(lx, ly)
            self.springs.add(Spring(pivot.argcenter, left.argcenter, pl, 1))
            self.springs.add(Spring(left.argcenter, pivot.argcenter, pl, 1))

            self.locations[right.argcenter] = Vector(rx, ry)
            self.springs.add(Spring(pivot.argcenter, right.argcenter, pr, 1))
            self.springs.add(Spring(right.argcenter, pivot.argcenter, pr, 1))

        self.vertices.update({left, right})
        return

    def _optimize(self, movers: Union[Set[Cluster], str] = 'all'):
        if isinstance(movers, str):
            if movers == 'all':
                movers: Set[Cluster] = {vertex for vertex in self.vertices}
            else:
                raise ValueError(f'movers must be a set of Clusters or must be the string \'all\'. Got {movers} instead.')

        forces: Dict[Cluster, Vector] = {cluster: Vector(0, 0) for cluster in movers}
        for cluster in movers:
            springs: Set[Spring] = {spring for spring in self.springs if cluster in spring}
            for spring in springs:
                pass
            pass
        return

    def force_direct(self):
        replacements: Set[Cluster] = {cluster for cluster in self.vertices if (cluster not in self.graph) and (cluster not in self._replaced)}

        while replacements:
            for cluster in replacements:
                self._replace_with_children(cluster)
            self._replaced.update(replacements)
            replacements = {cluster for cluster in self.vertices if (cluster not in self.graph) and (cluster not in self._replaced)}
        [print(spring) for spring in self.springs]
        return
