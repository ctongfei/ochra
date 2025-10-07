import io
import xml.etree.ElementTree as ET
from typing import Literal

from PIL import Image
import resvg_py

import numpy as np
import jax.numpy as jnp
from jaxtyping import Float

import ochra as ox
import ochra.svg as svg


def rasterize(c: ox.Canvas) -> Float[np.ndarray, "h w"]:
    """Rasterizes the canvas to a numpy array using ReSVG."""
    svg_str = ET.tostring(svg.to_svg(c), encoding='unicode')
    png_bytes = resvg_py.svg_to_bytes(svg_string=svg_str)
    img = Image.open(io.BytesIO(png_bytes))
    alpha_channel = np.asarray(img, dtype=np.float32)[:, :, 3] / 255
    return alpha_channel


global_viewport = ox.AxisAlignedRectangle((0, 0), (100, 100))

def check_approx(
        e: ox.Parametric,
        approx_method: Literal["polyline", "hermite"] = "hermite",
) -> bool:
    """
    Compares the rasterization of the element and its linear approximation,
    and returns True if the PSNR is above 30.
    """
    approximator = {
        "polyline": lambda x: x.approx_as_polyline(),
        "hermite": lambda x: x.approx_as_hermite_spline(),
    }[approx_method]
    gold_canvas, approx_canvas = [
        ox.Canvas([elems], viewport=global_viewport)
        for elems in (e, approximator(e))
    ]
    gold_img, approx_img = [rasterize(c) for c in (gold_canvas, approx_canvas)]
    approx_img = np.clip(approx_img / np.sum(approx_img) * np.sum(gold_img), 0, 1)  # Match brightness

    diff = np.abs(gold_img - approx_img)
    mse = np.mean(diff * diff)
    psnr = 10 * np.log10(1.0 / mse)
    print(f"PSNR = {psnr:.4f} for {e}")
    return psnr > 30  # 30dB is roughly the threshold for human perception


def test_line_segment():
    assert check_approx(ox.LineSegment((1, 1), (99, 99)))
    assert check_approx(ox.LineSegment((0, 80), (80, 10)))
    assert check_approx(ox.LineSegment((0, 10), (100, 80)))

def test_circle():
    assert check_approx(ox.Circle(10, (20, 20)))
    assert check_approx(ox.Circle(40, (50, 50)))

def test_ellipse():
    assert check_approx(ox.Ellipse.from_foci_and_major_axis((10, 10), (40, 40), 55))

