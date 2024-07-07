import math
from dataclasses import replace

from ochra.plane import Point
from ochra.marker import Marker
from ochra.rect import AxisAlignedRectangle
from ochra.poly import Polygon, Polyline
from ochra.conic import Circle
from ochra.style import Fill
from ochra.style.stroke import LineJoin, Stroke
from ochra.util.functions import deg_to_rad

