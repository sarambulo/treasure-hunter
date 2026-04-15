from math import ceil, floor, cos, sin, pi
import numpy as np
from .shapes import Circle, Rectangle
from aim_fsm import WallObj, BarrelObj, WorldObject, DoorwayObj
from aim_fsm import geometry
from .translate import object_to_shape
from .cell_states import encode_cell, decode_cell, cell_state_decoding

class GridWorld():    
    def __init__(self, square_size: float = 5, bbox=None, grid_shape=(150,150), grid_margin=0):
        self.cell_size = square_size  # in mm
        self.grid_margin = grid_margin  # in mm
        self.grid_shape = grid_shape  # array shape
        if bbox is None:
            self.bbox = ((0,0), (self.grid_shape[0]*self.cell_size, self.grid_shape[1]*self.cell_size))
        else:
            self.bbox = bbox  # in mm
        self.initialize_grid(bbox=bbox)

    def possible_cell_values(self):
        return list(cell_state_decoding.keys())

    def initialize_grid(self, bbox=None):
        if bbox:
            self.bbox = bbox
            self.grid_shape = (ceil((bbox[1][0] - bbox[0][0] + 2*self.grid_margin)/self.cell_size),
                               ceil((bbox[1][1] - bbox[0][1] + 2*self.grid_margin)/self.cell_size))
        self.grid = np.zeros(self.grid_shape, dtype=np.int32)
        self.maxdist = 1

    def coords_to_grid(self, xcoord,ycoord):
        "Convert world map coordinates to grid subscripts."
        x = floor((xcoord-self.bbox[0][0]+self.grid_margin) / self.cell_size)
        y = floor((ycoord-self.bbox[0][1]+self.grid_margin) / self.cell_size)
        if x >= 0 and x < self.grid_shape[0] and \
           y >= 0 and y < self.grid_shape[1]:
            return (x,y)
        else:
            return (None,None)

    def grid_to_coords(self,gridx,gridy):
        if (gridx < 0 or gridx >= self.grid_shape[0] or 
            gridy < 0 or gridy >= self.grid_shape[1]):
            return (None, None)
        xmin = self.bbox[0][0]
        ymin = self.bbox[0][1]
        x = gridx*self.cell_size + xmin - self.grid_margin
        y = gridy*self.cell_size + ymin - self.grid_margin
        return (x,y)

    def add_objects(self, objects: list[WallObj | BarrelObj]):
        labels = [encode_cell(type(obj)) for obj in objects]
        shapes = [object_to_shape(obj) for obj in objects]
        self.add_shapes(shapes, labels)
            
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
        width = rectangle.dimensions[0]
        height = rectangle.dimensions[1]
        theta = geometry.wrap_angle(rectangle.orient)
        for x in range(floor(centerX-width/2), ceil(centerX+width/2), int(self.cell_size/2)):
            for y in range(floor(centerY-height/2), ceil(centerY+height/2), int(self.cell_size/2)):
                new_x = ((x - centerX) * cos(theta) - (y - centerY) * sin(theta)) + centerX
                new_y = ((x - centerX) * sin(theta) + (y - centerY) * cos(theta)) + centerY
                self.set_cell(new_x, new_y, label)

    def add_cone(self, cone, label):
        center_x, center_y = cone.center
        radius = cone.radius
        start_theta = cone.orientation_degrees
        end_theta = start_theta + cone.angle_degrees
        # Ray casting
        CONE_DEGREE_DELTA = 3
        for theta in range(int(start_theta), int(end_theta), CONE_DEGREE_DELTA):
            for r in range(0, radius, int(self.cell_size / 2)):
                coord_x = center_x - r * sin(theta/180*pi)
                coord_y = center_y + r * cos(theta/180*pi)
                # Check if ray hits wall
                grid_x, grid_y = self.coords_to_grid(coord_x, coord_y)
                if grid_x is not None and self.grid[grid_x, grid_y] != encode_cell(WallObj):
                    self.grid[grid_x, grid_y] = label
                else:
                    # Move to the next ray
                    break
    
    def set_cell(self, xcoord, ycoord, type_label):
        (x,y) = self.coords_to_grid(xcoord,ycoord)
        if x is not None:
            self.grid[x,y] = type_label

    def set_empty_cell(self,xcoord,ycoord):
        self.set_cell_contents(xcoord, ycoord, 0)

    def set_cell_contents(self,xcoord,ycoord,contents):
        (x,y) = self.coords_to_grid(xcoord,ycoord)
        if x is not None:
            self.grid[x,y] = contents
        else:
            print('**** bbox=', self.bbox, '  grid_shape=', self.grid_shape,
                  '  x,y=', (x,y), '  xcoord,ycoord=', (xcoord,ycoord))
            print(ValueError('Coordinates (%s, %s) are outside the wavefront grid' % ((xcoord,ycoord))))