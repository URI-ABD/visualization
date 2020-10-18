import numpy as np
import os
from pyclam import Manifold, criterion

from pyvis.visualizer import Visualizer

PLOTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots'))


def main():
    filename = os.path.join(PLOTS_PATH, 'test.png')
    data = np.asarray([[1, 1], [2, 2], [1, 2], [2, 1]], dtype=float)
    manifold: Manifold = Manifold(data, 'euclidean').build(criterion.MaxDepth(10))
    vis = Visualizer(manifold)
    vis.force_direct(manifold.layers[-1])
    vis.draw(filename)
    return


if __name__ == '__main__':
    os.makedirs(PLOTS_PATH, exist_ok=True)
    main()
