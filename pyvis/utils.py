def key(i: int, j: int) -> int:
    """ Transforms from (col, row) to index of cell in lower-triangular matrix, without the diagonal. """
    if i < j:
        i, j = j, i
    return (i * (i - 1)) // 2 + j + 1
