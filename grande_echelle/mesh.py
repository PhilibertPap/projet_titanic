import gmsh

filename = "mesh/coque"
SHELL_CELL_TAG = 1
RIVET_CELL_TAG = 2

gmsh.initialize()
geom = gmsh.model.geo

# Segment coque "Titanic impact" (ordre de grandeur)
L = 100.0   # longueur de segment [m]
B = 14.1    # demi-largeur [m] -> axe y (transversal)
T = 10.5    # tirant d'eau [m] -> axe z (vertical)
a = 0.25    # courbure longitudinale
c = 0.08    # courbure verticale (reduite)
H = 0.02

# Maillage structuré non uniforme:
# - iceberg mobile sur toute la longueur x -> pas de raffinement local en x
# - raffinement dans la bande transversale (paramètre v) de contact probable
# - plus grossier ailleurs pour accélérer les calculs
# Mode aperçu rapide (ParaView / itérations de mise au point)
# Réduire fortement la taille pour accélérer la génération + le solveur.
Nu = 36
Nv = 6
rivet_u_centers = [0.2, 0.4, 0.6, 0.8]
# Rivet Titanic: diametre nominal ~0.875 in (22.2 mm), pas longitudinal ~3 in (76.2 mm).
# La bande materiau est homogenisee. Avec Nu=320, une colonne vaut L/Nu=0.3125 m.
rivet_band_half_width_u = 0.0018

# Zone probable de rupture (règle géométrique simple en espace paramétrique)
# u ~ position longitudinale (x), v ~ position transversale (y = B v)
# L'iceberg se déplace sur toute la longueur -> on raffine surtout selon v.
impact_v_center = 0.00
impact_v_half_width = 0.18


def _piecewise_refined_nodes(n, center, half_width, fine_fraction=0.55):
    """
    Build a monotone partition of [0, 1] with a denser central band [center-half_width, center+half_width].
    The total number of intervals is exactly `n`.
    """
    x0 = max(0.0, center - half_width)
    x1 = min(1.0, center + half_width)
    left_len = x0
    mid_len = max(x1 - x0, 1e-12)
    right_len = 1.0 - x1

    n_mid = max(4, int(round(fine_fraction * n)))
    n_mid = min(n - 2, n_mid) if n >= 6 else max(1, n - 2)
    n_outer = n - n_mid
    if n_outer <= 0:
        n_mid = n
        n_left = 0
        n_right = 0
    else:
        if left_len + right_len < 1e-12:
            n_left = n_outer // 2
        else:
            n_left = int(round(n_outer * left_len / (left_len + right_len)))
        n_left = min(max(n_left, 0), n_outer)
        n_right = n_outer - n_left

    # Guarantee all non-zero-length segments receive at least one interval when possible.
    if left_len > 1e-12 and n_left == 0 and n_mid > 1:
        n_left = 1
        n_mid -= 1
    if right_len > 1e-12 and n_right == 0 and n_mid > 1:
        n_right = 1
        n_mid -= 1

    nodes = [0.0]

    def append_uniform_segment(a_seg, b_seg, n_seg):
        if n_seg <= 0 or b_seg <= a_seg:
            return
        h_seg = (b_seg - a_seg) / n_seg
        for k in range(1, n_seg + 1):
            nodes.append(a_seg + k * h_seg)

    append_uniform_segment(0.0, x0, n_left)
    append_uniform_segment(x0, x1, n_mid)
    append_uniform_segment(x1, 1.0, n_right)

    # Numerical cleanup and exact endpoint
    nodes[0] = 0.0
    if nodes[-1] != 1.0:
        nodes[-1] = 1.0
    return nodes

def hull_xyz(u, v):
    x = L * u
    z_keel = -T + a * (u - 0.5)**2
    y = B * v
    # Partie immergée + franc-bord
    z = z_keel + c * v**2 + H * v**4
    return x, y, z

# Grosse maille de fond pour éviter une explosion du nombre de triangles.
# (Les surfaces sont triangulées par Gmsh; `lcar` pilote le raffinement effectif.)
lcar = 1.2
pts = [[0]*(Nv+1) for _ in range(Nu+1)]
u_nodes = [i / Nu for i in range(Nu + 1)]
v01_nodes = _piecewise_refined_nodes(Nv, 0.5 * (impact_v_center + 1.0), 0.5 * impact_v_half_width)
v_nodes = [-1.0 + 2.0 * s for s in v01_nodes]

for i in range(Nu+1):
    u = u_nodes[i]
    for j in range(Nv+1):
        v = v_nodes[j]
        x, y, z = hull_xyz(u, v)
        pts[i][j] = geom.addPoint(x, y, z, lcar)

# Lignes longitudinales (u = const)
long_lines = []
for i in range(Nu+1):
    row = []
    for j in range(Nv):
        l = geom.addLine(pts[i][j], pts[i][j+1])
        row.append(l)
    long_lines.append(row)

# Lignes de niveau (v = const)
vert_lines = []
for j in range(Nv+1):
    col = []
    for i in range(Nu):
        l = geom.addLine(pts[i][j], pts[i+1][j])
        col.append(l)
    vert_lines.append(col)

# Surfaces quadrilatères
shell_surfaces = []
rivet_surfaces = []
for i in range(Nu):
    for j in range(Nv):
        # les 4 lignes autour de la "cellule" (i,j)
        l1 = long_lines[i][j]       # u = i,   v: j -> j+1
        l3 = long_lines[i+1][j]     # u = i+1, v: j -> j+1
        l2 = vert_lines[j+1][i]     # v = j+1, u: i -> i+1
        l4 = vert_lines[j][i]       # v = j,   u: i -> i+1

        cl = geom.addCurveLoop([l1, l2, -l3, -l4])
        s  = geom.addSurfaceFilling([cl])
        u_center = (i + 0.5) / Nu
        is_rivet = any(abs(u_center - u0) <= rivet_band_half_width_u for u0 in rivet_u_centers)
        if is_rivet:
            rivet_surfaces.append(s)
        else:
            shell_surfaces.append(s)

geom.synchronize()

# --- Définition des lignes de bord pour les conditions limites ---

# Bord gauche (u = 0) : toutes les lignes long_lines[0][j] pour j=0..Nv-1
left_lines = [long_lines[0][j] for j in range(Nv)]

# Bord droit (u = 1) : toutes les lignes long_lines[Nu][j]
right_lines = [long_lines[Nu][j] for j in range(Nv)]

# (optionnel) Bord bas (v = 0) et haut (v = Nv) si tu en as besoin
bottom_lines = [vert_lines[0][i]  for i in range(Nu)]
top_lines    = [vert_lines[Nv][i] for i in range(Nu)]

# Coque: matériau standard et zones rivetées
gmsh.model.addPhysicalGroup(2, shell_surfaces, SHELL_CELL_TAG)
gmsh.model.setPhysicalName(2, SHELL_CELL_TAG, "HullPlate")
if rivet_surfaces:
    gmsh.model.addPhysicalGroup(2, rivet_surfaces, RIVET_CELL_TAG)
    gmsh.model.setPhysicalName(2, RIVET_CELL_TAG, "HullRivets")

# Bords pour BC :
gmsh.model.addPhysicalGroup(1, left_lines, 1)
gmsh.model.setPhysicalName(1, 1, "Left")
gmsh.model.addPhysicalGroup(1, right_lines, 2)
gmsh.model.setPhysicalName(1, 2, "Right")
gmsh.model.addPhysicalGroup(1, bottom_lines, 3)
gmsh.model.setPhysicalName(1, 3, "Bottom")
gmsh.model.addPhysicalGroup(1, top_lines, 4)
gmsh.model.setPhysicalName(1, 4, "Top")

for i in range(Nu+1):
    for j in range(Nv):
        geom.mesh.setTransfiniteCurve(long_lines[i][j], 2)
for j in range(Nv+1):
    for i in range(Nu):
        geom.mesh.setTransfiniteCurve(vert_lines[j][i], 2)

gmsh.model.mesh.generate(dim=2)
gmsh.write(filename + ".msh")
gmsh.finalize()
