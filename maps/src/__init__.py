from .map_types.i_shape      import create_i_map
from .map_types.square       import create_square_map
from .map_types.bottleneck   import create_bottleneck_1x5_map
from .map_types.random_field import create_random_field_5x5_map
from .map_types.maze         import create_maze_5x5_map

__all__ = [
    "create_i_map",
    "create_square_map",
    "create_bottleneck_1x5_map",
    "create_random_field_5x5_map",
    "create_maze_5x5_map",
]
