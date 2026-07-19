import base64
import xml.etree.ElementTree as ET
from typing import Dict, ContextManager

from ochra.util import f2s
from ochra.functions import rad_to_deg
from ochra.core import *  # noqa: F403
from ochra.style import *  # noqa: F403
from ochra.mark import Marker, MarkerConfig, Mark
from ochra.text import Text
from ochra.image import Image


class _ReflectionX(Reflection, Scaling):
    def __init__(self):
        Reflection.__init__(self, Line.y_axis)
        Scaling.__init__(self, (1.0, -1.0))


refl_x = _ReflectionX()  # Ochra <-> SVG coordinate system


class Session(ContextManager):
    def __init__(self):
        pass

    def __enter__(self):
        Marker.all_named_markers = {}
        Marker.all_named_symbols = {}

    def __exit__(self, __exc_type, __exc_value, __traceback):
        Marker.all_named_markers = {}
        Marker.all_named_symbols = {}


def dash_to_css(dash: Dash) -> dict[str, str]:
    style = {"stroke-dasharray": " ".join(str(d) for d in dash.array)}
    offset = {} if dash.offset is None else {"stroke-dashoffset": str(dash.offset)}
    return {**style, **offset}


def marker_config_to_css(marker: MarkerConfig) -> dict[str, str]:
    if marker is None:
        return {}
    start = {} if marker.start is None else {"marker-start": f"url(#{marker.start.name})"}
    mid = {} if marker.mid is None else {"marker-mid": f"url(#{marker.mid.name})"}
    end = {} if marker.end is None else {"marker-end": f"url(#{marker.end.name})"}
    return {**start, **mid, **end}


def stroke_to_css(s: Stroke) -> dict[str, str]:
    if s is None:
        s = Stroke()
    stroke = {"stroke": s.color.hex}
    dash = {} if s.dash is None else dash_to_css(s.dash)
    line_cap = {} if s.line_cap is None else {"stroke-linecap": s.line_cap.value}
    line_join = {} if s.line_join is None else {"stroke-linejoin": s.line_join.value}
    miter_limit = {} if s.miter_limit is None else {"stroke-miterlimit": f2s(s.miter_limit)}
    opacity = {} if s.opacity is None else {"stroke-opacity": f2s(s.opacity)}
    width = {} if s.width is None else {"stroke-width": f2s(s.width)}
    return {**stroke, **dash, **width, **line_cap, **line_join, **miter_limit, **opacity, **width}


def fill_to_css(f: Fill) -> dict[str, str]:
    if f is None:
        f = Fill()
    fill = {"fill": f.color.hex}
    fill_rule = {} if f.rule is None else {"fill-rule": f.rule.value}
    opacity = {} if f.opacity is None else {"fill-opacity": f2s(f.opacity)}
    return {**fill, **fill_rule, **opacity}


def font_to_css(font: Font) -> Dict[str, str]:
    family = {"font-family": font.family}
    size = {} if font.size is None else {"font-size": str(font.size)}
    weight = {} if font.weight is None else {"font-weight": f2s(font.weight.weight)}
    style = {} if font.style is None else {"font-style": str(font.style)}
    return {**family, **size, **weight, **style}


def affine_transformation_to_css(t: AffineTransformation) -> dict[str, str]:
    """
    Converts an affine transformation to a CSS transform string.
    """
    m = t.matrix / t.matrix[2, 2]
    [a, c, tx, b, d, ty] = m[:2, :].flatten().tolist()
    s = " ".join(f2s(x) for x in [a, b, c, d, tx, ty])
    return {"transform": f"matrix({s})"}


def contour_to_svg_path_data(contour: Contour) -> str:
    """Lowers exactly representable contour segments to SVG path data."""
    start = contour.segments[0].at(0)
    commands = [f"M {f2s(start.x)} {f2s(start.y)}"]

    def append_segment(segment: Parametric):
        if isinstance(segment, LineSegment):
            commands.append(f"L {f2s(segment.p1.x)} {f2s(segment.p1.y)}")
        elif isinstance(segment, QuadraticBezierCurve):
            commands.append(
                f"Q {f2s(segment.p1.x)} {f2s(segment.p1.y)} {f2s(segment.p2.x)} {f2s(segment.p2.y)}"
            )
        elif isinstance(segment, CubicBezierCurve):
            commands.append(
                f"C {f2s(segment.p1.x)} {f2s(segment.p1.y)} {f2s(segment.p2.x)} {f2s(segment.p2.y)} "
                f"{f2s(segment.p3.x)} {f2s(segment.p3.y)}"
            )
        elif isinstance(segment, HermiteCurve):
            append_segment(segment.as_cubic_bezier_curve())
        elif isinstance(segment, Polyline | QuadraticBezierSpline | CubicBezierSpline | HermiteSpline):
            for child in segment.segments:
                append_segment(child)
        else:
            raise NotImplementedError(f"Exact SVG contour lowering is not implemented for {type(segment)}.")

    for segment in contour.segments:
        append_segment(segment)
    commands.append("Z")
    return " ".join(commands)


def element_to_svg(c: Canvas, e: Element) -> ET.Element:
    match e:
        case Group():
            g = ET.Element(
                "g",
            )
            g.extend([element_to_svg(c, child) for child in e.elements])
            return g
        case AnyAffinelyTransformed():
            g = ET.Element(
                "g",
                **affine_transformation_to_css(e.transformation),
            )
            g.append(element_to_svg(c, e.element))
            return g
        case EmbeddedCanvas():
            # TODO: clipping
            d = e.left_bottom - c.viewport.bottom_left
            return element_to_svg(c, e.canvas.translate(d.x, d.y))
        case Image():
            return ET.Element(
                "image",
                x=f2s(e.bottom_left.x),
                y=f2s(e.bottom_left.y),
                width=f2s(e.width),
                height=f2s(e.height),
                href=f"data:image/png;base64,{base64.b64encode(e.to_png_bytes()).decode('utf-8')}",
            )
        case Text():
            if not jnp.isclose(e.angle, 0.0):
                return element_to_svg(
                    c,
                    AnyAffinelyTransformed(
                        Text(e.text, e.bottom_left, angle=0.0, font=e.font),
                        Rotation.centered(-e.angle, e.bottom_left),
                    ),
                )
            else:
                t = ET.Element(
                    "text",
                    x=f2s(e.visual_bbox().bottom_left.x),
                    y=f2s(e.visual_bbox().bottom_left.y),
                    **font_to_css(e.font),
                )
                t.text = e.text
                return t
        case Mark():
            return ET.Element(
                "use",
                x=f2s(e.point.x + e.marker.viewport.bottom_left.x),
                y=f2s(e.point.y + e.marker.viewport.bottom_left.y),
                href=f"#symbol-{e.marker.name}",
            )  # conform to SVG 1.1 standard, can't use refX, refY
        case Line():
            segment = clip_line_aabb(e, c.viewport)
            if segment is None:
                return ET.Element("group")
            else:
                return element_to_svg(c, segment)
        case Ray():
            segment = clip_ray_aabb(e, c.viewport)
            if segment is None:
                return ET.Element("group")
            else:
                return element_to_svg(c, segment)
        case LineSegment():
            return ET.Element(
                "line",
                x1=f2s(e.p0.x),
                y1=f2s(e.p0.y),
                x2=f2s(e.p1.x),
                y2=f2s(e.p1.y),
                **stroke_to_css(e.get_style(Stroke)),
                **marker_config_to_css(e.get_style(MarkerConfig)),
            )
        case Contour():
            return ET.Element(
                "path",
                d=contour_to_svg_path_data(e),
                fill="none",
                **stroke_to_css(e.get_style(Stroke)),
                **marker_config_to_css(e.get_style(MarkerConfig)),
            )
        case Region():
            css = fill_to_css(e.get_style(Fill))
            css["fill-rule"] = e.fill_rule.value
            return ET.Element(
                "path",
                d=" ".join(contour_to_svg_path_data(contour) for contour in e.contours),
                **stroke_to_css(e.get_style(Stroke)),
                **css,
                **marker_config_to_css(e.get_style(MarkerConfig)),
            )
        case Polyline():
            return ET.Element(
                "polyline",
                points=" ".join(f"{f2s(p.x)},{f2s(p.y)}" for p in e.knots),
                fill="none",
                **stroke_to_css(e.get_style(Stroke)),
                **marker_config_to_css(e.get_style(MarkerConfig)),
            )
        case Polygon():
            return ET.Element(
                "polygon",
                points=" ".join(f"{f2s(p.x)},{f2s(p.y)}" for p in e.vertices),
                **stroke_to_css(e.get_style(Stroke)),
                **fill_to_css(e.get_style(Fill)),
                **marker_config_to_css(e.get_style(MarkerConfig)),
            )
        case Circle():
            return ET.Element(
                "circle",
                cx=f2s(e.center.x),
                cy=f2s(e.center.y),
                r=f2s(e.radius),
                **stroke_to_css(e.get_style(Stroke)),
                **fill_to_css(e.get_style(Fill)),
            )
        case Ellipse():
            rot = (
                {"transform": f"rotate({f2s(rad_to_deg(e.angle))} {f2s(e.center.x)} {f2s(e.center.y)})"}
                if e.angle != 0
                else {}
            )
            return ET.Element(
                "ellipse",
                cx=f2s(e.center.x),
                cy=f2s(e.center.y),
                rx=f2s(e._a),
                ry=f2s(e._b),
                **rot,
                **stroke_to_css(e.get_style(Stroke)),
                **fill_to_css(e.get_style(Fill)),
            )
        case Parabola():
            segment = clip_parabola_aabb(e, c.viewport)
            if segment is None:
                return ET.Element("group")
            else:
                return element_to_svg(c, segment)
        case QuadraticBezierCurve():
            return ET.Element(
                "path",
                d=f"M {f2s(e.p0.x)} {f2s(e.p0.y)} Q {f2s(e.p1.x)} {f2s(e.p1.y)}, {f2s(e.p2.x)} {f2s(e.p2.y)}",
                fill="none",
                **stroke_to_css(e.get_style(Stroke)),
            )
        case QuadraticBezierSpline():
            parts = [
                f"Q {f2s(e.points[2 * i + 1].x)} {f2s(e.points[2 * i + 1].y)}, {f2s(e.points[2 * i + 2].x)} {f2s(e.points[2 * i + 2].y)}"
                for i in range(e.num_segments)
            ]
            return ET.Element(
                "path",
                d=f"M {f2s(e.points[0].x)} {f2s(e.points[0].y)} {' '.join(parts)}",
                fill="none",
                **stroke_to_css(e.get_style(Stroke)),
                **marker_config_to_css(e.get_style(MarkerConfig)),
            )
        case CubicBezierCurve():
            return ET.Element(
                "path",
                d=f"M {f2s(e.p0.x)} {f2s(e.p0.y)} C {f2s(e.p1.x)} {f2s(e.p1.y)}, {f2s(e.p2.x)} {f2s(e.p2.y)}, {f2s(e.p3.x)} {f2s(e.p3.y)}",
                fill="none",
                **stroke_to_css(e.get_style(Stroke)),
            )
        case CubicBezierSpline():
            parts = [
                f"C {f2s(e.points[3 * i + 1].x)} {f2s(e.points[3 * i + 1].y)}, {f2s(e.points[3 * i + 2].x)} {f2s(e.points[3 * i + 2].y)}, {f2s(e.points[3 * i + 3].x)} {f2s(e.points[3 * i + 3].y)}"
                for i in range(e.num_segments)
            ]
            return ET.Element(
                "path",
                d=f"M {f2s(e.points[0].x)} {f2s(e.points[0].y)} {' '.join(parts)}",
                fill="none",
                **stroke_to_css(e.get_style(Stroke)),
                **marker_config_to_css(e.get_style(MarkerConfig)),
            )
        case HermiteCurve():
            return element_to_svg(c, e.as_cubic_bezier_curve())
        case HermiteSpline():
            return element_to_svg(c, e.as_cubic_bezier_spline())
        case Parametric():
            return element_to_svg(c, e.approx_as_hermite_spline())
        case Annotation():  # Materialize under the Ochra coordinate system
            return element_to_svg(c, e.transform(refl_x).materialize().transform(refl_x))
        case _:
            raise NotImplementedError(f"Unsupported element type: {type(e)}")


def marker_to_svg_def(c: Canvas, m: Marker) -> ET.Element:
    v = m.viewport
    marker = ET.Element(
        "marker",
        id=m.name,
        orient="auto",  # TODO: implement orientation
        markerUnits=m.units.value,
        markerWidth=f2s(v.width),
        markerHeight=f2s(v.height),
        refX="0",
        refY="0",
        viewBox=f"{v.bottom_left.x} {-v.bottom_left.y - v.height} {v.width} {v.height}",
    )
    marker.extend([element_to_svg(c, e.transform(refl_x)) for e in m.elements])
    return marker


def marker_to_svg_symbol(c: Canvas, m: Marker) -> ET.Element:
    v = m.viewport
    symbol = ET.Element(
        "symbol", id=f"symbol-{m.name}", viewBox=f"0 0 {v.width} {v.height}", width=str(v.width), height=str(v.height)
    )
    symbol.extend(
        [element_to_svg(c, e.transform(refl_x).translate(-v.bottom_left.x, -v.bottom_left.y)) for e in m.elements]
    )  # Conform to SVG 1.1 standard, can't use refX, refY -- so have to move the elements
    return symbol


def to_svg_coord[E: Element](e: E) -> E:
    if isinstance(e, Text):
        return Text(e.text, refl_x(e.bottom_left), e.angle, e.font)
    elif isinstance(e, Image):
        return Image(e.image, refl_x(e.aabb().top_left))
    elif isinstance(e, Group):
        return Group([to_svg_coord(e) for e in e.elements])
    else:
        return e.transform(refl_x)


def materialize[E: Element](e: E) -> E:
    if isinstance(e, Annotation):
        return e.materialize()
    elif isinstance(e, Canvas):
        return Canvas([materialize(e) for e in e.elements], e.viewport)
    elif isinstance(e, Group):
        return Group([materialize(e) for e in e.elements])
    else:
        return e


def to_svg_element(c: Canvas, horizontal_padding: float = 0.0, vertical_padding: float = 0.0) -> ET.Element:
    hp, vp = horizontal_padding, vertical_padding
    # Materialize annotations
    c = materialize(c)
    all_elements = [
        element_to_svg(c, to_svg_coord(e))  # to SVG coordinate system
        for e in c.elements
    ]
    all_markers = [marker_to_svg_def(c, m) for m in Marker.all_named_markers.values()]
    all_symbols = [marker_to_svg_symbol(c, m) for m in Marker.all_named_symbols.values()]
    root = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(c.viewport.width + 2 * hp),
        height=str(c.viewport.height + 2 * vp),
        viewBox=f"{c.viewport.bottom_left.x - hp} {-c.viewport.bottom_left.y - c.viewport.height - vp} {c.viewport.width + 2 * hp} {c.viewport.height + 2 * vp}",
    )
    defs = ET.Element("defs")
    defs.extend(all_markers)
    defs.extend(all_symbols)
    root.append(defs)
    root.extend(all_elements)
    return root


def to_svg_str(c: Canvas, horizontal_padding: float = 0.0, vertical_padding: float = 0.0) -> str:
    """Converts the canvas to an SVG string."""
    e = to_svg_element(c, horizontal_padding, vertical_padding)
    tree = ET.ElementTree(e)
    ET.indent(tree)
    return ET.tostring(e, encoding="unicode")


def save_svg(c: Canvas, path: str, horizontal_padding: float = 0.0, vertical_padding: float = 0.0):
    """Saves the canvas to an SVG file."""
    e = to_svg_element(c, horizontal_padding, vertical_padding)
    tree = ET.ElementTree(e)
    ET.indent(tree)
    tree.write(path, encoding="utf-8")
