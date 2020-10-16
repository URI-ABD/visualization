from collections import namedtuple
from operator import itemgetter
from typing import Dict, Set

import numpy as np
from matplotlib import pyplot as plt
from pyclam import Cluster, Graph
from sympy import symbols, Eq, solve
from sympy.core.mul import Mul

Location = namedtuple('Location', 'x y')
Spring = namedtuple('Spring', 'source destination length constant')


class Visualizer:
    def __init__(self, root: Cluster, graph: Graph):
        self.root: Cluster = root
        self.graph: Graph = graph

        self.vertices: Set[Cluster] = {root}
        self.locations: Dict[int, Location] = {root.argcenter: Location(0, 0)}
        self.springs: Set[Spring] = set()

        self._replaced: Set[Cluster] = set()

    def draw(self, filename: str):
        xy = np.asarray([self.locations[cluster.argcenter] for cluster in self.graph], dtype=float)

        plt.clf()
        fig = plt.figure(figsize=(16, 10), dpi=200)
        fig.add_subplot(111)
        plt.scatter(xy[:, 0], xy[:, 1], s=4)

        plt.title('getting started')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.25)
        plt.close('all')
        return

    def _replace_with_children(self, pivot: Cluster):
        [left, right] = list(pivot.children)
        px, py = self.locations[pivot.argcenter]
        [pl, pr] = list(pivot.distance_from([left.argcenter, right.argcenter]))
        [lr] = list(left.distance_from(right.argcenter))

        if left.argcenter in self.locations:
            rx, ry = px + pr, py
            self.locations[right.argcenter] = Location(rx, ry)
            self.springs.add(Spring(pivot.argcenter, right.argcenter, pr, 1))
            self.springs.add(Spring(right.argcenter, pivot.argcenter, pr, 1))
        elif right.argcenter in self.locations:
            lx, ly = px + pl, py
            self.locations[left.argcenter] = Location(lx, ly)
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

            self.locations[left.argcenter] = Location(lx, ly)
            self.springs.add(Spring(pivot.argcenter, left.argcenter, pl, 1))
            self.springs.add(Spring(left.argcenter, pivot.argcenter, pl, 1))

            self.locations[right.argcenter] = Location(rx, ry)
            self.springs.add(Spring(pivot.argcenter, right.argcenter, pr, 1))
            self.springs.add(Spring(right.argcenter, pivot.argcenter, pr, 1))

        self.vertices.update({left, right})
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
