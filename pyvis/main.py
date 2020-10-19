import os

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from pyclam import Manifold, criterion

from pyvis.visualizer import Visualizer

PLOTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plots'))


def main():
    data = np.asarray([[1, 1], [2, 2], [1, 2], [2, 1]], dtype=float)
    manifold: Manifold = Manifold(data, 'euclidean').build(criterion.MaxDepth(10))
    vis = Visualizer(manifold)

    st.title('Force Directed Layout')
    fig = plt.figure(figsize=(8, 8), dpi=200)
    vis.force_direct(manifold.layers[-1], fig=fig)

    st.write('Final Layout:')
    st.write(fig)
    return


if __name__ == '__main__':
    os.makedirs(PLOTS_PATH, exist_ok=True)
    main()
