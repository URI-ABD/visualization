def key(i: int, j: int) -> int:
    if i < j:
        i, j = j, i
    return (i * (i - 1)) // 2 + j + 1
