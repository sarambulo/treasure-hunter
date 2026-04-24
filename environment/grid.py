from math import ceil, floor, cos, sin, pi
import numpy as np
from .shapes import Circle, Rectangle
from aim_fsm import WallObj, BarrelObj, WorldObject, DoorwayObj
from aim_fsm import geometry
from .translate import object_to_shape
from .cell_states import encode_cell, decode_cell, cell_state_decoding
from typing import Sequence

class GridWorld():    
    def __init__(self, square_size: float = 5, grid_shape: tuple[int, int] = (150, 150), fill_value: int = 0):
        self.cell_size = square_size  # in mm
        self.grid_shape = grid_shape  # array shape
        self.grid = np.full(self.grid_shape, fill_value=fill_value, dtype=np.int32)
        self.one_hot = self.as_one_hot()

    def get_unique_cell_values(self) -> np.ndarray:
        values = sorted([encode_cell("empty"), encode_cell(WallObj), encode_cell(BarrelObj)])
        return np.array(values)
    
    def coords_to_grid(self, xcoord, ycoord):
        """
            Convert world map coordinates to grid row and column
            World map coordinate frame is: x forward, y left, z up
            World map origin is bottom right corner of room
        """
        H, W = self.grid_shape # Height and width
        row_cells = floor(xcoord / self.cell_size)
        column_cells = floor(ycoord / self.cell_size)
        row_index = H - 1 - row_cells
        column_index = W - 1 - column_cells
        if row_index >= 0 and row_index < H and \
           column_index >= 0 and column_index < W:
            return (row_index, column_index)
        else:
            return (None, None)

    def grid_to_coords(self, row, column):
        """
            Convert grid row and column to world map coordinates
            World map coordinate frame is: x forward, y left, z up
            World map origin is bottom right corner of room
        """
        if (row < 0 or row >= self.grid_shape[0] or 
            column < 0 or column >= self.grid_shape[1]):
            return (None, None)
        H, W = self.grid_shape # Height and width
        # We use the center coord of each cell
        x = (H - row - 0.5) * self.cell_size
        y = (W - column - 0.5) * self.cell_size
        return (x, y)

    def add_objects(self, objects: Sequence[WallObj | BarrelObj]):
        labels = [encode_cell(type(obj)) for obj in objects]
        shapes = [object_to_shape(obj) for obj in objects]
        self.add_shapes(shapes, labels)
        self.one_hot = self.as_one_hot() # Update one-hot representation
            
    def add_shapes(self, shapes: list | Rectangle | Circle, labels: list[int] | int):
        if isinstance(shapes, (Rectangle, Circle)):
            shapes_list = [shapes]
        else:
            shapes_list = shapes
        if isinstance(labels, int):
            labels_list = [labels]
        else:
            labels_list = labels
        assert len(labels_list) == len(shapes_list)
        for shape, label in zip(shapes_list, labels_list):
            if isinstance(shape, Circle):
                self.add_circle(shape, label)
            elif isinstance(shape, Rectangle):
                self.add_rectangle(shape, label)
            elif isinstance(shape, list):
                self.add_shapes(shape, label)
            else:
                raise Exception(f"%s has no add method defined for %s." % (self, shape))

    def add_circle(self, circle, label):
        center_x, center_y = circle.center[0,0], circle.center[1,0]
        radius = circle.radius
        for theta in range(0,360,5):
            for r in range(7):
                new_x = center_x + (radius-r) * cos(theta/180*pi)
                new_y = center_y + (radius-r) * sin(theta/180*pi)
                self.set_cell(new_x, new_y, label)

    def add_rectangle(self, rectangle, label):
        centerX, centerY = rectangle.center[0,0], rectangle.center[1,0]
        height = rectangle.dimensions[0]
        width = rectangle.dimensions[1]
        theta = geometry.wrap_angle(rectangle.orient)
        for x in range(floor(centerX-height/2), ceil(centerX+height/2), int(self.cell_size/2)):
            for y in range(floor(centerY-width/2), ceil(centerY+width/2), int(self.cell_size/2)):
                new_x = ((x - centerX) * cos(theta) - (y - centerY) * sin(theta)) + centerX
                new_y = ((x - centerX) * sin(theta) + (y - centerY) * cos(theta)) + centerY
                row, column = self.coords_to_grid(new_x, new_y)
                if row is not None and column is not None:
                    self.set_cell(row=row, column=column, label=label)

    def add_cone(self, cone, label):
        center_x, center_y = cone.center
        radius = cone.radius
        start_theta = cone.orientation_degrees - cone.angle_degrees / 2
        end_theta = start_theta + cone.angle_degrees
        # Ray casting
        CONE_DEGREE_DELTA = 3
        for theta in range(int(start_theta), int(end_theta), CONE_DEGREE_DELTA):
            for r in range(0, radius, int(self.cell_size / 2)):
                coord_x = center_x + r * cos(theta/180*pi)
                coord_y = center_y + r * sin(theta/180*pi)
                # Check if ray hits wall
                row, column = self.coords_to_grid(coord_x, coord_y)
                if row is not None and self.grid[row, column] != encode_cell(WallObj):
                    self.set_cell(row=row, column=column, label=label)
                else:
                    # Move to the next ray
                    break
    
    def set_cell(self, label: int, row = None, column = None, x = None, y = None):
        coords_as_input = False
        if x is not None and y is not None:
            row, column = self.coords_to_grid(x, y)
            coords_as_input = True
        if row is not None and column is not None:
            self.grid[row, column] = label
        else:
            raise ValueError(
                f"Invalid inputs: {(x, y) if coords_as_input else [row, column]}"
            )
        
    def as_one_hot(self):
        # Get all posible combinations of row and column indices
        H, W = self.grid_shape
        row_idx, col_idx = np.indices((H, W))
        unique_cell_values = self.get_unique_cell_values()
        C = len(unique_cell_values)
        # Initialize the output as all zeros
        one_hot = np.zeros((C, H, W), dtype=np.uint8)
        zero_index_grid = self.grid - min(unique_cell_values) # zero indexed
        one_hot[zero_index_grid, row_idx, col_idx] = 1
        return one_hot