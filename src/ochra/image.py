from __future__ import annotations

import io
from functools import cached_property
from pathlib import Path

from PIL import Image as PILImage
import jax
import jax.numpy as jnp
from jaxtyping import Real

from ochra.geometry import Point, PointI, Vector
from ochra.core import Element, AxisAlignedRectangle, Rectangle


class Image(Element):
    """
    Embeds an image in the canvas.
    """
    def __init__(self, image: Real[jax.Array, "h w c"], bottom_left: PointI):
        self.image = image
        self.height, self.width, _ = image.shape
        self.bottom_left = Point.mk(bottom_left)

    @cached_property
    def rotated_bbox(self) -> Rectangle:
        top_right = self.bottom_left + Vector.mk((self.width, self.height))
        return Rectangle(self.bottom_left, top_right)

    def to_png_bytes(self) -> bytes:
        """Converts the image to PNG bytes."""
        buf = io.BytesIO()
        PILImage.fromarray(self.image.__array__()).save(buf, format="png")
        return buf.getvalue()

    @classmethod
    def from_pil_image(cls, image: PILImage.Image, bottom_left: PointI) -> Image:
        image = jnp.array(image)
        return cls(image, bottom_left)

    @classmethod
    def from_file(cls, path: Path | str, bottom_left: PointI) -> Image:
        image = PILImage.open(path)
        return cls.from_pil_image(image, bottom_left)

    def aabb(self) -> AxisAlignedRectangle:
        return self.rotated_bbox.aabb()
