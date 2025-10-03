from .primitives import Segment

def build_cup(x_left=-0.6, x_right=0.6, y_bottom=0.0, height=0.8):
    left   = Segment([x_left,  y_bottom], [x_left,  y_bottom+height], [ +1.0, 0.0])
    right  = Segment([x_right, y_bottom], [x_right, y_bottom+height], [ -1.0, 0.0])
    bottom = Segment([x_left,  y_bottom], [x_right, y_bottom],        [  0.0, +1.0])
    return [left, bottom, right]

def build_walls(x_lim=(0, 100), y_lim=(0, 100)):
    left   = Segment([x_lim[0],  y_lim[0]], [x_lim[0],  y_lim[1]], [+1.0, 0.0])
    right  = Segment([x_lim[1], y_lim[0]], [x_lim[1], y_lim[1]], [-1.0, 0.0])
    bottom = Segment([x_lim[0],  y_lim[0]], [x_lim[1], y_lim[0]],[0.0, +1.0])
    top    = Segment([x_lim[0],  y_lim[1]], [x_lim[1], y_lim[1]],[0.0, -1.0])
    return [left, bottom, right, top]