from typing import Tuple

from pyclam import Manifold, Graph, Cluster


class Force:
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
    
    def __neg__(self) -> 'Force':
        return Force(-self.x, -self.y)

    def __add__(self, other: 'Force') -> 'Force':
        return Force(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Force') -> 'Force':
        return self + (-other)


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
    def location(self) -> Tuple[float, float]:
        return self.x, self.y

    def move(self, force: Force) -> 'Point':
        self._x = self.x + force.x / self.mass
        self._y = self.y + force.y / self.mass
        return self

    def difference(self, other: 'Point') -> Tuple[float, float]:
        return self.x - other.x, self.y - other.y

    def distance(self, other: 'Point') -> float:
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5


class Spring:
    def __init__(self, left: Point, right: Point, rest_length: float, stiffness: float = 1):
        if left == right:
            raise ValueError(f'left and right must be different points.')
        elif left.index < right.index:
            self.left, self.right = left, right
        else:
            self.left, self.right = right, left
        self.rest_length: float = rest_length
        self.stiffness: float = stiffness

    def __str__(self):
        return f'{self.left}, {self.right}, {self.rest_length:.2f}, {self.stiffness}'

    def __repr__(self):
        return f'left: {self.left}, right: {self.right}, rest length: {self.rest_length}, stiffness: {self.stiffness}'

    def __hash__(self):
        i, j = self.right.index, self.left.index
        return (i * (i - 1)) // 2 + j + 1

    @property
    def length(self) -> float:
        return self.left.distance(self.right)

    def direction(self, anchor: str = 'left') -> Tuple[float, float]:
        if anchor not in ['left', 'right']:
            raise ValueError(f'Direction must be anchored at \'left\' or \'right\'. Got {anchor} instead.')
        direction: Tuple[float, float] = self.left.difference(self.right)
        distance: float = self.left.distance(self.right)
        sign = 1 if anchor == 'left' else -1
        return sign * direction[0] / distance, sign * direction[1] / distance

    def force(self, anchor: str = 'left') -> Force:
        if anchor not in ['left', 'right']:
            raise ValueError(f'Force must be anchored at \'left\' or \'right\'. Got {anchor} instead.')
        magnitude: float = -self.stiffness * (self.length - self.rest_length)
        direction: Tuple[float, float] = self.direction(anchor)
        return Force(magnitude * direction[0], magnitude * direction[1])


class Visualizer:
    def __init__(self, manifold: Manifold):
        self.manifold: Manifold = manifold
        self.root: Cluster = manifold.root

    def force_direct(self, graph: Graph):
        pass

    def draw(self, filename: str):
        pass
