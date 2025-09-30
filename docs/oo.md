# Object-oriented type design

The design of Ochra leans heavily on geometric properties, and these are encoded as relations between types.

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
 - Under a projective transformation, it becomes a `Polygon`. 

If a shape is invariant under a transformation type (i.e., return shape has the same type), we encode this as a protocol.

* `Circle` should be
    * `RigidInvariant[Circle]`
    * `AffineInvariant[Ellipse]`
    * `ProjectiveInvariant[Conic]`
  
- `AxisAlignedRectangle` should be
    - `TranslationalInvariant[AxisAlignedRectangle]`
    - `RigidInvariant[Rectangle]`
    - `AffineInvariant[Polygon]`
    - `ProjectiveInvariant[Polygon]`
