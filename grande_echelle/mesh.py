import gmsh
import numpy as np
from pathlib import Path


filename = "mesh/coque"
SHELL_CELL_TAG = 1


# Geometry scales for a Titanic-like half-hull shell surface
# (order of magnitude anchors: length ~269 m, beam ~28.2 m, draft ~10.5 m)
L = 269.0   # ship length [m] -> x
B_HALF = 14.1  # half-beam [m] -> y
DRAFT = 10.5   # draft below waterline [m]
FREEBOARD_TO_DECK = 7.8  # approx keel->deck total height ~18.3 m
Z_BOTTOM = -DRAFT
Z_TOP = FREEBOARD_TO_DECK

# Half-hull surface: y is the half-breadth (outboard side).
# Shape cues (from Titanic hull lines) approximated here:
# - rounded bilge
# - broad parallel mid-body
# - tapered bow/stern
# - slight tumblehome toward upper deck (very mild)
Y_KEEL = 0.0
Y_BILGE_BASE = 12.2
Y_WATERLINE_BASE = 13.9
Y_DECK_BASE = 13.5
Y_MIDSHIP_FULLNESS = 0.06
Y_LONGITUDINAL_WARP = 0.08

# Longitudinal curvature in z (very mild):
# - near-waterline sheer-like variation stronger than near keel
# - weak global sag/hog over the modeled segment
Z_SHEER_AMPLITUDE = 0.45
Z_KEEL_SAG_AMPLITUDE = 0.10
Z_END_RISE_AMPLITUDE = 0.18

# Longitudinal planform narrowing (half-breadth effect): long parallel mid-body and smoother ends
MIDBODY_U0 = 0.14
MIDBODY_U1 = 0.86
END_TAPER_MIN = 0.16

# Loft sections (couples) along x
N_SECTIONS_X = 11
N_SECTION_PTS = 17

# Iceberg trajectory (used only to define a mesh-refinement band)
ICEBERG_CENTER_Y = -10.8    # m, starboard side (sign flipped)
ICEBERG_CENTER_Z = -6.3     # m, below waterline (z=0)
# Longitudinal contact/damage zone (order of magnitude ~300 ft ~ 91 m)
ICEBERG_X_START = 177.0
ICEBERG_X_END = 268.0

# Mesh size field (refine around the iceberg trajectory band)
SIZE_MIN = 0.22
SIZE_MAX = 3.00
DIST_MIN = 0.8
DIST_MAX = 6.5

# Local refinement to resolve homogenized rivet bands represented as vertical
# strips (directed along z) and distributed regularly in x within the iceberg
# impact zone.
N_RIVET_STRIPS_X = 8
RIVET_STRIP_PHYSICAL_WIDTH_X = 0.30
RIVET_STRIP_Z_MIN = -10.2
RIVET_STRIP_Z_MAX = 0.2
# Stronger refinement than the baseline is needed for 0.30 m strips to appear
# as continuous bands on the shell surface (and not isolated CG1 spots).
RIVET_STRIP_SIZE_MIN = 0.05
RIVET_STRIP_SIZE_MAX = 1.20
RIVET_STRIP_MARGIN_X = 0.60
RIVET_STRIP_MARGIN_Y = 1.20


def _smoothstep(a: float, b: float, x: float) -> float:
    if x <= a:
        return 0.0
    if x >= b:
        return 1.0
    t = (x - a) / (b - a)
    return t * t * (3.0 - 2.0 * t)


def _midbody_fullness_factor(u: float) -> float:
    # 1.0 in the parallel middle body, smoothly tapered near both ends.
    left = _smoothstep(0.0, MIDBODY_U0, u)
    right = 1.0 - _smoothstep(MIDBODY_U1, 1.0, u)
    plateau = min(left, right)
    return END_TAPER_MIN + (1.0 - END_TAPER_MIN) * plateau


def hull_xyz(u: float, v: float) -> tuple[float, float, float]:
    """Reference hull-side geometry used for diagnostics and control-point generation."""
    x = L * u
    xu = 2.0 * u - 1.0
    s = 0.5 * (v + 1.0)  # 0 = keel, 1 = upper deck line
    fullness_x = _midbody_fullness_factor(u)
    endness = 1.0 - fullness_x
    s_water = (0.0 - Z_BOTTOM) / (Z_TOP - Z_BOTTOM)

    # z = vertical coordinate, with mild longitudinal curvature (sag + sheer-like effect)
    z = Z_BOTTOM + (Z_TOP - Z_BOTTOM) * s
    z += Z_KEEL_SAG_AMPLITUDE * (xu * xu)
    z += Z_SHEER_AMPLITUDE * (max(s - s_water, 0.0) / max(1.0 - s_water, 1e-9)) ** 1.8 * (xu * xu)
    # Extra end rise/sheer near bow/stern, stronger toward the upper hull
    z += Z_END_RISE_AMPLITUDE * (s ** 2.2) * (endness ** 1.6)

    # Section profile y(z): bilge + flare, with mild longitudinal fullness.
    # Fuller near midship, slimmer near the ends.
    fullness = 1.0 + Y_MIDSHIP_FULLNESS * (1.0 - xu * xu)
    y_bilge = Y_BILGE_BASE * fullness * fullness_x
    y_waterline = Y_WATERLINE_BASE * fullness * fullness_x
    y_deck = Y_DECK_BASE * fullness * fullness_x

    # Piecewise smooth half-breadth profile from keel to deck
    # Lower bilge transition + sharper turn gives a more vertical flank over a
    # larger height before the hull rounds out.
    s_bilge = 0.08
    if s <= s_bilge:
        t = s / s_bilge
        y_section = Y_KEEL + (y_bilge - Y_KEEL) * (t ** 1.45)
    elif s <= s_water:
        t = (s - s_bilge) / max(s_water - s_bilge, 1e-9)
        # Slightly convex flank-to-waterline transition (avoid visible concavity)
        y_section = y_bilge + (y_waterline - y_bilge) * (t ** 0.88)
    else:
        t = (s - s_water) / max(1.0 - s_water, 1e-9)
        # very mild tumblehome toward the deck
        y_section = y_waterline + (y_deck - y_waterline) * (t ** 1.15)
    y = -(y_section + Y_LONGITUDINAL_WARP * xu * (2.0 * s - 1.0))
    return x, y, z


def _build_smooth_hull_surface(occ) -> tuple[int, dict[str, list[int]]]:
    # Build a smooth hull patch by lofting several smooth transverse sections (couples).
    # This gives a more ship-like geometry than a single control-net B-spline surface.
    section_wires: list[int] = []
    for i_sec in range(N_SECTIONS_X):
        u = i_sec / (N_SECTIONS_X - 1)
        pts: list[int] = []
        for j in range(N_SECTION_PTS):
            v = -1.0 + 2.0 * j / (N_SECTION_PTS - 1)
            x, y, z = hull_xyz(u, v)
            pts.append(occ.addPoint(x, y, z))
        spline = occ.addSpline(pts)
        wire = occ.addWire([spline], checkClosed=False)
        section_wires.append(wire)

    loft_out = occ.addThruSections(
        section_wires,
        makeSolid=False,
        makeRuled=False,
        maxDegree=5,
    )
    surf_tags = [tag for dim, tag in loft_out if dim == 2]
    if len(surf_tags) != 1:
        raise RuntimeError(f"Expected one lofted surface, got: {loft_out}")
    surf = surf_tags[0]
    occ.synchronize()

    # Identify boundary curves geometrically to preserve BC tags used by the solver.
    boundary = gmsh.model.getBoundary([(2, surf)], oriented=False, recursive=False)
    edge_map = {"left": [], "right": [], "bottom": [], "top": []}
    remaining: list[tuple[int, float]] = []
    for dim, tag in boundary:
        if dim != 1:
            continue
        xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(1, tag)
        xmid = 0.5 * (xmin + xmax)
        zmid = 0.5 * (zmin + zmax)
        if abs(xmid - 0.0) < 1e-3 * L:
            edge_map["left"].append(tag)
        elif abs(xmid - L) < 1e-3 * L:
            edge_map["right"].append(tag)
        else:
            remaining.append((tag, zmid))

    if len(remaining) == 2:
        remaining.sort(key=lambda item: item[1])
        edge_map["bottom"].append(remaining[0][0])  # low z
        edge_map["top"].append(remaining[1][0])     # high z
    elif len(remaining) != 0:
        raise RuntimeError(f"Unexpected number of non-x boundary curves: {len(remaining)}")

    return surf, edge_map


def _add_mesh_size_field(occ) -> None:
    # Refinement trajectory follows the shell more closely than a straight line:
    # sample a 3D path at the impact depth across x (x+ -> x-).
    s_impact = max(0.0, min(1.0, (ICEBERG_CENTER_Z - Z_BOTTOM) / (Z_TOP - Z_BOTTOM)))
    v_impact = 2.0 * s_impact - 1.0
    x_start = max(0.0, min(L, ICEBERG_X_START))
    x_end = max(0.0, min(L, ICEBERG_X_END))
    if x_end < x_start:
        x_start, x_end = x_end, x_start
    u_start = x_start / L
    u_end = x_end / L
    _, y_shell_ref, _ = hull_xyz(0.5, v_impact)
    y_shift = ICEBERG_CENTER_Y - y_shell_ref

    path_points = []
    for k in range(41):
        # Sample only the longitudinal zone where the iceberg is assumed to act.
        u = u_end + (u_start - u_end) * (k / 40.0)
        x, y_shell, z = hull_xyz(u, v_impact)
        # Decalage lateral pour centrer la bande de raffinement sur la trajectoire cible.
        y = y_shell + y_shift
        path_points.append(occ.addPoint(x, y, z))
    traj = occ.addSpline(path_points)
    occ.synchronize()

    field = gmsh.model.mesh.field

    # 1) Refinement along the iceberg trajectory
    f_dist_traj = field.add("Distance")
    field.setNumbers(f_dist_traj, "CurvesList", [traj])
    field.setNumber(f_dist_traj, "Sampling", 200)

    f_th_traj = field.add("Threshold")
    field.setNumber(f_th_traj, "InField", f_dist_traj)
    field.setNumber(f_th_traj, "SizeMin", SIZE_MIN)
    field.setNumber(f_th_traj, "SizeMax", SIZE_MAX)
    field.setNumber(f_th_traj, "DistMin", DIST_MIN)
    field.setNumber(f_th_traj, "DistMax", DIST_MAX)

    # 2) Refinement around the 8 vertical homogenized rivet strips (same x-centers
    # as grande_echelle/main.py default bands) in the iceberg passage zone.
    rivet_band_boxes = []
    # For vertical rivet strips, keep the x-window narrow but cover the whole
    # side-shell width in y so the refinement is visible from top to bottom.
    y_extent_rivet = 1.2 * max(
        abs(Y_WATERLINE_BASE * (1.0 + Y_MIDSHIP_FULLNESS)),
        abs(Y_DECK_BASE * (1.0 + Y_MIDSHIP_FULLNESS)),
        abs(ICEBERG_CENTER_Y),
    ) + RIVET_STRIP_MARGIN_Y
    x_margin = 0.5 * RIVET_STRIP_PHYSICAL_WIDTH_X
    x_centers = np.linspace(
        x_start + x_margin,
        x_end - x_margin,
        N_RIVET_STRIPS_X,
    )
    for x_center in x_centers:
        f_box = field.add("Box")
        field.setNumber(f_box, "VIn", RIVET_STRIP_SIZE_MIN)
        field.setNumber(f_box, "VOut", RIVET_STRIP_SIZE_MAX)
        field.setNumber(f_box, "XMin", x_center - 0.5 * RIVET_STRIP_PHYSICAL_WIDTH_X - RIVET_STRIP_MARGIN_X)
        field.setNumber(f_box, "XMax", x_center + 0.5 * RIVET_STRIP_PHYSICAL_WIDTH_X + RIVET_STRIP_MARGIN_X)
        field.setNumber(f_box, "YMin", -y_extent_rivet)
        field.setNumber(f_box, "YMax", y_extent_rivet)
        field.setNumber(f_box, "ZMin", RIVET_STRIP_Z_MIN)
        field.setNumber(f_box, "ZMax", RIVET_STRIP_Z_MAX)
        rivet_band_boxes.append(f_box)

    # 3) Combine trajectory + rivet-band boxes
    f_min = field.add("Min")
    field.setNumbers(f_min, "FieldsList", [f_th_traj, *rivet_band_boxes])

    field.setAsBackgroundMesh(f_min)

    # Let the background field drive the mesh size instead of point-wise defaults.
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)


def main():
    gmsh.initialize()
    gmsh.model.add("titanic_hull_segment")
    occ = gmsh.model.occ

    # 1) Geometrie coque
    surf, edges = _build_smooth_hull_surface(occ)

    # 2) Groupes physiques (tags utilises ensuite par le solveur)
    gmsh.model.addPhysicalGroup(2, [surf], SHELL_CELL_TAG)
    gmsh.model.setPhysicalName(2, SHELL_CELL_TAG, "HullPlate")

    # Boundary tags expected by the solver
    if edges["left"]:
        gmsh.model.addPhysicalGroup(1, edges["left"], 1)
        gmsh.model.setPhysicalName(1, 1, "Left")
    if edges["right"]:
        gmsh.model.addPhysicalGroup(1, edges["right"], 2)
        gmsh.model.setPhysicalName(1, 2, "Right")
    if edges["bottom"]:
        gmsh.model.addPhysicalGroup(1, edges["bottom"], 3)
        gmsh.model.setPhysicalName(1, 3, "Bottom")
    if edges["top"]:
        gmsh.model.addPhysicalGroup(1, edges["top"], 4)
        gmsh.model.setPhysicalName(1, 4, "Top")

    # 3) Champ de taille de maille (raffinement pres de la trajectoire iceberg)
    _add_mesh_size_field(occ)

    # 4) Generation / export du maillage surfacique
    gmsh.model.mesh.generate(2)
    out_path = (Path(__file__).resolve().parent / f"{filename}.msh").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(out_path))
    print(f"Maillage ecrit dans : {out_path}")
    gmsh.finalize()


if __name__ == "__main__":
    main()
