# Object-oriented type design

The design of Ochra leans heavily on geometric properties, and these are encoded as relations between types.

## Geometric foundations

Ochra's geometric model is organized as a hierarchy of progressively richer structures:

```text
RP²: projective incidence geometry
 └── A²: a selected affine chart
      └── E²: Euclidean metric geometry
```

The layers have distinct responsibilities:

- **The projective plane `RP²`** provides homogeneous points, projective lines, conics, incidence, and projective
  transformations. A projective line may be the line at infinity. Operations such as incidence remain meaningful there,
  while affine or metric properties need not.
- **The affine plane `A²`** selects a line at infinity and therefore distinguishes finite points. It provides vectors,
  parallelism, affine combinations, ordinary segments and rays, Bezier curves, contours, and axis-aligned bounds.
- **The Euclidean plane `E²`** adds a metric and orientation to the affine plane. It provides lengths, angles,
  perpendicularity, circles, rectangles, stroke widths, and other visual measurements.

Most drawable declarations inhabit `A²` or `E²`, where concepts such as betweenness, boundedness, contour orientation,
and winding are well-defined. Projective geometry supplies the more general incidence model and determines how these
declarations behave under projective transformations.

A transformation may move a declaration to a less structured, more general layer. The result type should express that
widening exactly. For example:

```text
Circle      --affine----> Ellipse
Ellipse     --projective-> Conic
LineSegment --projective-> generic Parametric, pending ProjectiveLineSegment
```

An affine `LineSegment` has finite endpoints and a conventional notion of betweenness. Its projective image may cross
the line at infinity, so it cannot always remain an affine segment. A projective segment must retain enough homogeneous
data to identify the chosen arc on its projective line; its two projective endpoints alone are insufficient. Until such
a representation exists, affine segments, rays, polylines, polygons, and contours declare only affine closure. The
generic `Parametric` fallback does not claim that their affine geometric type has been preserved.

Operations are available only at the layer whose structure supports them. A projective `Line` always supports incidence,
but slope requires an affine chart and closest-point projection requires a Euclidean metric. Such operations are partial
for a type that can represent the line at infinity and must reject inputs for which they are undefined.

SVG is not another geometric layer in this hierarchy. `ochra.svg` lowers Ochra declarations from a selected affine chart
to SVG's device coordinate system. If projective geometry crosses the chart's line at infinity, lowering may split the
visible finite geometry, omit geometry outside the chart, or report that a declaration cannot be represented faithfully.
SVG path commands and other serialization details must not define or leak into Ochra's geometric semantics.

## Curves, contours, and regions

Ochra distinguishes three related concepts rather than deriving them from SVG paths:

- A parametric curve is a map from a parameter interval into the affine plane. It may be open, closed, bounded, or
  unbounded.
- `PiecewiseParametric[C]` supplies equal-width parameterization, continuity partitions, and bounds for an indexed
  sequence of segments by default, without imposing knots or closure semantics. Delegated slices may provide explicit
  unequal parameter breaks so that slicing remains pointwise exact.
- A `Spline` describes a piecewise-polynomial representation. Splines are generally open and therefore do not subclass
  `Contour`.
- A `Contour` is an oriented closed boundary assembled from an ordered tuple of exact parametric segments. Its traversal
  determines signed area and winding direction.
- A `ClosedSpline` is a contour made from a cyclic homogeneous sequence of segments. Unlike an open `Spline`, it stores
  `n` cyclic knots for `n` segments; the first knot is not duplicated. `to_spline()` cuts the cycle at its first knot and
  duplicates that knot only in the resulting open spline.
- A `Polygon` is a `ClosedSpline[LineSegment]`; `to_polyline()` is its concrete open-spline conversion.
- A `Region` is a filled planar set bounded by one or more contours. Multiple contours represent disconnected components
  or holes, interpreted using the region's nonzero or even-odd fill rule.

This separation allows a rounded rectangle to be a contour, a ring to be a region with two contours, and an open Hermite
spline to remain only a curve. `JoinedParametric` composes parameterizations but does not create unified boundary, stroke,
or fill semantics. During SVG lowering, a contour becomes one closed `<path>` and a region becomes one `<path>` with
multiple closed subpaths; those SVG details are consequences of the model rather than its definition.

The generic `Sliceable[R]` protocol records the exact result type of slicing. Primitive lines, Bezier curves, and Hermite
curves preserve their concrete types. A generic piecewise curve returns a `SplineSlice` that delegates complete interior
segments and slices only its boundary segments. Closed-spline slicing proceeds forward and wraps through parameter zero
when the end parameter precedes the start. `Polygon` refines this result to `Polyline`; its locus is exact, while the
result uses the polyline's conventional equal-width segment parameterization.

Relations between types of elements are encoded as subtyping relations. For example,

 - `Circle <: Ellipse <: Conic <: Implicit <: Element`
 - `AxisAlignedRectangle <: Rectangle <: Polygon <: Parametric <: Element`

Transformations also have their own type hierarchy. For example,

 - `Translation <: RigidTransformation <: AffineTransformation <: ProjectiveTransformation`

Each element `e: E` can be transformed by a transformation `f: F`. It should return the most specific type possible:

 - for `E₀ <: E₁`, `transform(E₀, F) <: transform(E₁, F)`
 - for `F₀ <: F₁`, `transform(E, F₀) <: transform(E, F₁)`

Take `Circle` as an example.

 - Under a translation, it remains a `Circle`. 
 - Under a rotation, it remains a `Circle`. 
 - Under a scaling, it becomes an `Ellipse`. 
 - Under a general affine transformation, it becomes an `Ellipse`. 
 - Under a projective transformation, it may even become a `Conic`. 

Or taking `Rectangle` as an example,

 - Under a translation, it remains a `Rectangle`. 
 - Under a rotation, it becomes a `Polygon`. 
 - Under a scaling, it remains a `Rectangle`. 
 - Under a general affine transformation, it becomes a `Polygon` (still quadrilateral, but not necessarily a rectangle) 
 - Under a general projective transformation, it widens to a generic parametrization until Ochra has an appropriate
   projective boundary type.

If transforming a shape always produces a particular result type, we encode that closure relation as a protocol.

Conceptually, this is a multi-parameter typeclass relating an element type and a transformation type to an associated
result type: `Transform(E, F) -> R`. The result must remain exact so that subsequent operations and transformations can
be type-checked when chained. Python has neither typeclasses nor associated types, so Ochra approximates this relation
with generic closure ABCs, multiple inheritance, overloads on `self`, and matching runtime dispatch. This encoding is
more verbose and less complete than the underlying mathematics: generic arguments are unavailable to runtime dispatch,
overlap between transformation classes must be ordered manually, and some relationships cannot be inferred without
explicit overloads. These are deliberate concessions to Python's type system rather than geometric semantics; the model
still requires each transformation to return the most specific mathematically valid type.

* `Circle` should be
    * `ClosedUnderRigidTransformations[Circle]`
    * `ClosedUnderAffineTransformations[Ellipse]`
    * `ClosedUnderProjectiveTransformations[Conic]`
  
- `AxisAlignedRectangle` should be
    - `ClosedUnderTranslations[AxisAlignedRectangle]`
    - `ClosedUnderRigidTransformations[Rectangle]`
    - `ClosedUnderAffineTransformations[Polygon]`
