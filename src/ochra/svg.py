import xml.etree.ElementTree as ET
from typing import Dict, ContextManager

from ochra.core import *
from ochra.functions import *
from ochra.style import *
from ochra.mark import *
from ochra.text import Text


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


def transformation_to_css(t: Transformation) -> dict[str, str]:
    m = t.matrix / t.matrix[2, 2]
    [a, c, e, b, d, f] = m[:2, :].flatten()
    s = ' '.join(f2s(x.item()) for x in [a, b, c, d, e, f])
    return {
        "transform": f"matrix({s})"
    }


def element_to_svg(c: Canvas, e: Element) -> ET.Element:
    if isinstance(e, Group):
        g = ET.Element(
            "g",
        )
        g.extend([element_to_svg(c, e) for e in e.elements])
        return g
    if isinstance(e, AnyTransformed):
        g = ET.Element(
            "g",
            **transformation_to_css(e.transformation),
        )
        g.append(element_to_svg(c, e.element))
        return g
    elif isinstance(e, EmbeddedCanvas):
        # TODO: clipping
        d = e.left_bottom - c.viewport.bottom_left
        return element_to_svg(c, e.canvas.translate(d.x, d.y))
    elif isinstance(e, Text):
        if e.angle != 0.0:
            flip = Scaling((1, -1))
            return element_to_svg(
                c,
                AnyTransformed(
                    Text(e.text, e.bottom_left, angle=0.0, font=e.font),
                    Rotation.centered(-e.angle, flip(e.bottom_left))
                )
            )
        else:
            t = ET.Element(
                "text.py",
                x=f2s(e.visual_bbox().bottom_left.x),
                y=f2s(-e.visual_bbox().bottom_left.y),
                **font_to_css(e.font),
            )
            t.text = e.text
            return t
    elif isinstance(e, Mark):
        return ET.Element(
            "use",
            x=f2s(e.point.x + e.marker.viewport.bottom_left.x),
            y=f2s(e.point.y + e.marker.viewport.bottom_left.y),
            href=f"#symbol-{e.marker.name}",
        )  # conform to SVG 1.1 standard, can't use refX, refY
    elif isinstance(e, Line):
        intersection = intersect_line_aabb(e, c.viewport)
        if intersection is None:
            return ET.Element("group")
        elif isinstance(intersection, Point):
            # TODO: draw a dot?
            pass
        elif isinstance(intersection, list):
            p0, p1 = intersection
            θ = LineSegment(p0, p1).angle
            d = Vector.unit(θ) * (e.stroke.width or 1.0)  # should * 0.5, but be conservative
            if (p1 - p0).dot(d) < 0:
                d = -d
            return element_to_svg(c, LineSegment(p0 + (-d), p1 + d, stroke=e.stroke))
    elif isinstance(e, LineSegment):
        return ET.Element(
            "line",
            x1=f2s(e.p0.x),
            y1=f2s(e.p0.y),
            x2=f2s(e.p1.x),
            y2=f2s(e.p1.y),
            **stroke_to_css(e.stroke),
            **marker_config_to_css(MarkerConfig(e.marker_start, None, e.marker_end)),
        )
    elif isinstance(e, Polyline):
        return ET.Element(
            "polyline",
            points=" ".join(f"{f2s(p.x)},{f2s(p.y)}" for p in e.vertices),
            fill="none",
            **stroke_to_css(e.stroke),
            **marker_config_to_css(MarkerConfig(e.marker_start, e.marker_mid, e.marker_end)),
        )
    elif isinstance(e, Polygon):
        return ET.Element(
            "polygon",
            points=" ".join(f"{f2s(p.x)},{f2s(p.y)}" for p in e.vertices),
            **stroke_to_css(e.stroke),
            **fill_to_css(e.fill),
            **marker_config_to_css(MarkerConfig(None, e.marker, None)),
        )
    elif isinstance(e, Circle):
        return ET.Element(
            "circle",
            cx=f2s(e.center.x),
            cy=f2s(e.center.y),
            r=f2s(e.radius),
            **stroke_to_css(e.stroke),
            **fill_to_css(e.fill),
        )
    elif isinstance(e, Ellipse):
        rot = {
            "transform": f"rotate({f2s(rad_to_deg(e.angle))} {f2s(e.center.x)} {f2s(e.center.y)})"
        } if e.angle != 0 else {}
        return ET.Element(
            "ellipse",
            cx=f2s(e.center.x),
            cy=f2s(e.center.y),
            rx=f2s(e.a),
            ry=f2s(e.b),
            **rot,
            **stroke_to_css(e.stroke),
            **fill_to_css(e.fill),
        )
    elif isinstance(e, Parabola):
        ts = [t for s in c.viewport.edges for t in intersect_segment_conic_param(s, e)]
        if len(ts) == 0:
            return ET.Element("group")
        tmin, tmax = min(ts), max(ts)
        p0 = e.at(tmin)
        p1 = e.at(tmax)
        t0 = e.tangent_vector_at(tmin)
        t1 = e.tangent_vector_at(tmax)
        c = get_quadratic_bezier_curve_control_point_by_tangent(p0, t0, p1, t1)
        return element_to_svg(c, QuadraticBezierCurve.from_points(p0, c, p1, stroke=e.stroke))
        
    elif isinstance(e, QuadraticBezierCurve):
        return ET.Element(
            "path",
            d=f"M {f2s(e.p0.x)} {f2s(e.p0.y)} Q {f2s(e.p1.x)} {f2s(e.p1.y)}, {f2s(e.p2.x)} {f2s(e.p2.y)}",
            fill="none",
            **stroke_to_css(e.stroke),
        )
    elif isinstance(e, QuadraticBezierPath):
        parts = [
            f"Q {f2s(e.mat[2*i+1, 0])} {f2s(e.mat[2*i+1, 1])}, {f2s(e.mat[2*i+2, 0])} {f2s(e.mat[2*i+2, 1])}"
            for i in range(e.num_segments)
        ]
        return ET.Element(
            "path",
            d=f"M {f2s(e.mat[0, 0])} {f2s(e.mat[0, 1])} {' '.join(parts)}",
            fill="none",
            **stroke_to_css(e.stroke),
            **marker_config_to_css(e.markers)
        )
    elif isinstance(e, Parametric):
        return element_to_svg(c, e.approx_as_polyline())
    elif isinstance(e, Annotation):  # Materialize under the Ochra coordinate system
        return element_to_svg(c, e.scale(1, -1).materialize().scale(1, -1))
    else:
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
    marker.extend([element_to_svg(c, e.scale(1, -1)) for e in m.elements])
    return marker


def marker_to_svg_symbol(c: Canvas, m: Marker) -> ET.Element:
    v = m.viewport
    symbol = ET.Element(
        "symbol",
        id=f"symbol-{m.name}",
        viewBox=f"0 0 {v.width} {v.height}",
        width=str(v.width),
        height=str(v.height)
    )
    symbol.extend([
        element_to_svg(c, e.scale(1, -1).translate(-v.bottom_left.x, -v.bottom_left.y))
        for e in m.elements
    ])  # Conform to SVG 1.1 standard, can't use refX, refY -- so have to move the elements
    return symbol


def to_svg(c: Canvas, horizontal_padding: float = 0.0, vertical_padding: float = 0.0) -> ET.Element:
    hp, vp = horizontal_padding, vertical_padding
    all = [
        element_to_svg(c, e.scale(1, -1))  # to SVG coordinate system
        for e in c.elements
    ]
    all_markers = [
        marker_to_svg_def(c, m)
        for m in Marker.all_named_markers.values()
    ]
    all_symbols = [
        marker_to_svg_symbol(c, m)
        for m in Marker.all_named_symbols.values()
    ]
    root = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        width=str(c.viewport.width + 2 * hp),
        height=str(c.viewport.height + 2 * vp),
        viewBox=f"{c.viewport.bottom_left.x - hp} {-c.viewport.bottom_left.y - c.viewport.height - vp} {c.viewport.width + 2 * hp} {c.viewport.height + 2 * vp}"
    )
    defs = ET.Element("defs")
    defs.extend(all_markers)
    defs.extend(all_symbols)
    root.append(defs)
    root.extend(all)
    return root


def save_svg(c: Canvas, path: str, horizontal_padding: float = 0.0, vertical_padding: float = 0.0):
    e = to_svg(c, horizontal_padding, vertical_padding)
    tree = ET.ElementTree(e)
    ET.indent(tree)
    tree.write(path, encoding="utf-8")
