from typing import Collection, Tuple

from ochra.plot import Plot


class HeatMap(Plot[float, float]):
    """
    Represents a heatmap.
    """
    def __init__(self,
                 data: Collection[Tuple[Tuple[float, float], Tuple[float, float], float, ...]],
                 ):
        self.data = data
