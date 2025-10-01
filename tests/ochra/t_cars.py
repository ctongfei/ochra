import numpy as np
from vega_datasets import data

import ochra as ox
import ochra.plot as oxp
import ochra.style as oxs
from ochra.style import Palette

cars = data.cars()
names = ["USA", "Europe", "Japan"]
# filter cars where origin is USA then select only the columns Horsepower and Miles_per_Gallon
series = [
    list(cars[cars.Origin == origin][["Horsepower", "Miles_per_Gallon"]].itertuples(index=False)) for origin in names
]

nord = Palette.nord

markers = [
    ox.Marker.plus(stroke=oxs.Stroke(nord.aurora[0])),
    ox.Marker.cross(stroke=oxs.Stroke(nord.aurora[1])),
    ox.Marker.polygon(3, stroke=oxs.Stroke(nord.aurora[2])),
]

plot = oxp.Chart(
    size=(200, 200),
    x_axis=oxp.ContinuousAxis("Horsepower", (0, 240), major_ticks=np.linspace(0, 240, 7)),
    y_axis=oxp.ContinuousAxis("Miles per gallon", (0, 50)),
    plots=[
        oxp.ScatterPlot(name, data, marker=marker)
        for name, data, color, marker in zip(names, series, nord.colors, markers)
    ],
    palette=nord,
)

c = ox.Canvas([plot.draw()])

ox.save_svg(c, "test.svg", horizontal_padding=5, vertical_padding=5)
