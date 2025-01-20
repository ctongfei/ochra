import ochra as ox
import ochra.plot as oxp
from ochra.palettes import nord
from ochra.style import Colormap

import numpy as np

data = np.random.rand(16, 16)


plot = oxp.Chart(
    size=(128, 128),
    x_axis=oxp.DiscreteAxis("x", range(16)),
    y_axis=oxp.DiscreteAxis("y", range(16)),
    plots=[
        oxp.HeatMap(data, colormap=Colormap.viridis, palette=nord)
    ],
    palette=nord
)

c = ox.Canvas(elements=[plot.draw()])

ox.svg.save_svg(c, "test.svg")

