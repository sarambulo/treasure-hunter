import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

color_map = {-1: 'grey', 0: 'white', 1: 'blue', 2: 'black'}
custom_cmap = ListedColormap([color_map[i] for i in sorted(color_map.keys())])
value_to_cmap_index = {k: i for i, k in enumerate(sorted(color_map.keys()))}

def color_grid(categorical_grid: np.ndarray) -> np.ndarray:
    index_grid = categorical_grid - categorical_grid.min() # 0 indexing
    rgba_float_grid = custom_cmap(index_grid)
    rgb_grid = (rgba_float_grid[..., :3] * 255).astype(np.uint8)
    return rgb_grid