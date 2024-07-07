import xml.etree.ElementTree as ET
from typing import Dict

from ochra import Transformation
from ochra.canvas import Canvas, EmbeddedCanvas
from ochra.mark import Mark
from ochra.group import Group
from ochra.conic import Ellipse, Circle, Arc
from ochra.element import Element, AnyTransformed
from ochra.marker import MarkerConfig, Marker
from ochra.parametric import Parametric
from ochra.poly import Polyline, Polygon
from ochra.segment import LineSegment
from ochra.text import Text
from ochra.style.font import Font
from ochra.style.stroke import Stroke, Dash
from ochra.style.fill import Fill
from ochra.util.functions import f2s, rad_to_deg


def dash_to_css(dash: Dash) -> Dict[str, str]:
    style = {"stroke-dasharray": " ".join(str(d) for d in dash.array)}
    offset = {} if dash.offset is None else {"stroke-dashoffset": str(dash.offset)}
    return {**style, **offset}


def marker_config_to_css(marker: MarkerConfig) -> Dict[str, str]:
    start = {} if marker.start is None else {"marker-start": f"url(#{marker.start.name})"}
    mid = {} if marker.mid is None else {"marker-mid": f"url(#{marker.mid.name})"}
    end = {} if marker.end is None else {"marker-end": f"url(#{marker.end.name})"}
    return {**start, **mid, **end}


def stroke_to_css(s: Stroke) -> Dict[str, str]:
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


def fill_to_css(f: Fill) -> Dict[str, str]:
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
    style = {} if font.style is None else {"font-style": font.style}
    return {**family, **size, **weight, **style}


def transformation_to_css(t: Transformation) -> Dict[str, str]:
    s = ' '.join(f2s(x.item()) for x in t.matrix[:2, 2].flat)
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
        d = e.left_bottom - c.viewport.left_bottom
        return element_to_svg(c, e.canvas.translate(d.x, d.y))
    elif isinstance(e, Text):
        t = ET.Element(
            "text",
            x=f2s(e.left_bottom.x),
            y=f2s(e.left_bottom.y),
            **font_to_css(e.font),
        )
        t.text = e.text
        return t
    elif isinstance(e, Mark):
        return ET.Element(
            "use",
            x=f2s(e.point.x),
            y=f2s(e.point.y),
            href=f"#symbol-{e.marker.name}",
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
            "transform": f"rotate({f2s(rad_to_deg(e.major_axis_angle))} {f2s(e.center.x)} {f2s(e.center.y)})"
        } if e.major_axis_angle != 0 else {}
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
    elif isinstance(e, Parametric):
        return element_to_svg(c, e.approx_as_polyline())
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
        viewBox=f"{v.left_bottom.x} {-v.left_bottom.y - v.height} {v.width} {v.height}",
    )
    marker.extend([element_to_svg(c, e) for e in m.elements])
    return marker


def marker_to_svg_symbol(c: Canvas, m: Marker) -> ET.Element:
    v = m.viewport
    symbol = ET.Element(
        "symbol",
        id=f"symbol-{m.name}",
        viewBox=f"{v.left_bottom.x} {-v.left_bottom.y - v.height} {v.width} {v.height}",
        width=str(v.width),
        height=str(v.height)
    )
    symbol.extend([element_to_svg(c, e) for e in m.elements])
    return symbol


def to_svg(c: Canvas) -> ET.Element:
    all = [
        element_to_svg(c, e.scale(1, -1))
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
        width=str(c.viewport.width),
        height=str(c.viewport.height),
        viewBox=f"{c.viewport.left_bottom.x} {-c.viewport.left_bottom.y - c.viewport.height} {c.viewport.width} {c.viewport.height}"
    )
    defs = ET.Element("defs")
    defs.extend(all_markers)
    defs.extend(all_symbols)
    root.append(defs)
    root.extend(all)
    return root


def to_svg_file(c: Canvas, path: str):
    e = to_svg(c)
    tree = ET.ElementTree(e)
    ET.indent(tree)
    tree.write(path, encoding="utf-8")
