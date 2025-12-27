import numpy as np


def quantize_items(items, ticks=120):
    if not items:
        return []
    grids = np.arange(0, items[-1].start + ticks, ticks, dtype=int)
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end = item.end + shift if item.end is not None else None
    return items
