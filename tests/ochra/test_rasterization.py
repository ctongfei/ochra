import io
import xml.etree.ElementTree as ET
from typing import Literal
import pytest

from PIL import Image
import resvg_py

import numpy as np
from jaxtyping import Float

import ochra as ox
import ochra.svg as svg

from .test_common import global_viewport, finite_parametric_test_cases


def rasterize(c: ox.Canvas) -> Float[np.ndarray, "h w"]:
    """Rasterizes the canvas to a numpy array using ReSVG."""
    svg_str = ET.tostring(svg.to_svg_element(c), encoding='unicode')
    png_bytes = resvg_py.svg_to_bytes(svg_string=svg_str)
    img = Image.open(io.BytesIO(png_bytes))
    alpha_channel = np.asarray(img, dtype=np.float32)[:, :, 3] / 255
    return alpha_channel

def check_approx(
        e: ox.Parametric,
        approx_method: Literal["polyline", "hermite"] = "hermite",
) -> bool:
    """
    Compares the rasterization of the element and its linear / cubic approximation,
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


@pytest.mark.parametrize("approx_method", ["polyline", "hermite"])
@pytest.mark.parametrize("elem_type", finite_parametric_test_cases.keys())
def test_rasterization(elem_type: type, approx_method: Literal["polyline", "hermite"]):
    """
    Checks that the rasterization of the element and its approximation match.
    """
    for e in finite_parametric_test_cases[elem_type]:
        assert check_approx(e, approx_method), f"Approximation with {approx_method} failed for {e}"
