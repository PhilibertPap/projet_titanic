import gmsh
import dolfinx.io.gmsh
from dolfinx import fem, mesh, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem # Conservé pour l'import mais on utilise PETSc manuel
import dolfinx.nls.petsc
from mpi4py import MPI
import ufl
import numpy as np
import petsc4py.PETSc as PETSc
import os

# =============================================================================
# 1. GÉOMÉTRIE ET MAILLAGE
# =============================================================================
def create_titanic_rivet_mesh(L=0.5, W=0.2, H=0.01, r=0.02, lc_fine=0.003, lc_coarse=0.01):
    gmsh.initialize()
    model = gmsh.model()
    model.add("Plaque_Rivet")

    factory = model.occ
    box = factory.addBox(-L/2, -W/2, -H/2, L, W, H)
    cylinder = factory.addCylinder(0, 0, -H, 0, 0, 2*H, r)
    
    # Découpe
    factory.cut([(3, box)], [(3, cylinder)])
    factory.synchronize()

    # --- 1. IDENTIFICATION DES COURBES DU TROU ---
    curves = model.getEntities(1)
    hole_curves = []
    
    for c in curves:
        com = model.occ.getCenterOfMass(1, c[1])
        dist_from_center = np.sqrt(com[0]**2 + com[1]**2)
        if dist_from_center < r * 1.5:
            hole_curves.append(c[1])

    # --- 2. CONFIGURATION DU RAFFINEMENT (FIELDS) ---
    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "CurvesList", hole_curves)
    model.mesh.field.setNumber(1, "Sampling", 100)

    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "SizeMin", lc_fine)
    model.mesh.field.setNumber(2, "SizeMax", lc_coarse)
    
    dist_min = r
    dist_max = r + 0.05
    
    model.mesh.field.setNumber(2, "DistMin", dist_min)
    model.mesh.field.setNumber(2, "DistMax", dist_max)
    model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    # --- 3. TAGS PHYSIQUES ---
    eps = 1e-6
    surfaces = model.getEntities(2)
    left_tags = []
    right_tags = []
    
    for s in surfaces:
        com = model.occ.getCenterOfMass(2, s[1])
        if np.abs(com[0] + L/2) < eps:
            left_tags.append(s[1])
        elif np.abs(com[0] - L/2) < eps:
            right_tags.append(s[1])

    volumes = model.getEntities(3)
    model.addPhysicalGroup(3, [v[1] for v in volumes], tag=1, name="Plaque")
    
    if left_tags:
        model.addPhysicalGroup(2, left_tags, tag=2, name="Gauche")
    if right_tags:
        model.addPhysicalGroup(2, right_tags, tag=3, name="Droite")

    model.mesh.generate(3)

    mdata = dolfinx.io.gmsh.model_to_mesh(model, MPI.COMM_WORLD, 0, gdim=3)
    gmsh.finalize()
    return mdata

# =============================================================================
# 2. PARAMÈTRES PHYSIQUES
# =============================================================================
E = 200e9  
nu = 0.3   
Gc = 3.0e3 
l0 = 0.005 

# --- Paramètres spécifiques AT1 ---
cw = 8.0 / 3.0 
seuil_critique = Gc / (cw * l0)

mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

# =============================================================================
# 3. INITIALISATION
# =============================================================================
mdata = create_titanic_rivet_mesh()
domain = mdata.mesh
facet_tags = mdata.facet_tags

V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
D = fem.functionspace(domain, ("Lagrange", 1))

u = fem.Function(V, name="Deplacement")
d = fem.Function(D, name="Endommagement")
d_old = fem.Function(D)

ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
dx = ufl.Measure("dx", domain=domain)

# =============================================================================
# 4. CONDITIONS LIMITES (BCs)
# =============================================================================
fdim = domain.topology.dim - 1
left_facets = facet_tags.find(2)
dofs_left = fem.locate_dofs_topological(V, fdim, left_facets)
bcs_u = [fem.dirichletbc(np.array([0,0,0], dtype=default_scalar_type), dofs_left, V)]

# =============================================================================
# 5. FORMULATION VARIATIONNELLE (MANUELLE POUR AT1)
# =============================================================================

# --- A. Fonctions ---
k_ell = 1e-6
def g(d):
    return (1 - d)**2 + k_ell

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma_undamaged(u):
    return 2.0 * mu * epsilon(u) + lmbda * ufl.tr(epsilon(u)) * ufl.Identity(len(u))

psi_elastic = 0.5 * ufl.inner(sigma_undamaged(u), epsilon(u))

# --- B. Formes pour le Déplacement ---
u_trial = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
T_mag = fem.Constant(domain, default_scalar_type(0.0))

a_u = g(d) * ufl.inner(sigma_undamaged(u_trial), epsilon(v)) * dx
L_u = ufl.dot(T_mag * ufl.as_vector([1.0, 0.0, 0.0]), v) * ds(3)

form_a_u = fem.form(a_u)
form_L_u = fem.form(L_u)

# --- C. Formes pour le Phase Field (AT1 STABILISÉ) ---
d_trial = ufl.TrialFunction(D)
w = ufl.TestFunction(D)

# On garde un petit terme stabilisateur (epsilon) pour que la matrice 
# ne soit jamais singulière quand l'énergie élastique est nulle.
eps_stab = 1e-8 * (Gc / l0) 

# L'énergie élastique "effective" pour l'endommagement.
# On prend le max pour éviter tout bruit numérique négatif sur u.
psi_eff = ufl.max_value(psi_elastic, 0.0)

# Forme Bilinéaire (A_d)
# On ajoute eps_stab pour que le terme (d_trial * w) ne soit jamais zéro.
a_d = (2.0 * psi_eff + eps_stab) * ufl.inner(d_trial, w) * dx + \
      (Gc * l0 / cw) * ufl.inner(ufl.grad(d_trial), ufl.grad(w)) * dx

# Forme Linéaire (L_d)
# On s'assure que le terme moteur ne s'active qu'au-dessus du seuil.
terme_moteur_at1 = ufl.max_value(2.0 * psi_eff - seuil_critique, 0.0)
L_d = terme_moteur_at1 * w * dx

form_a_d = fem.form(a_d)
form_L_d = fem.form(L_d)

# --- D. Création des solveurs PETSc ---
def create_solver():
    solver = PETSc.KSP().create(domain.comm)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    return solver

solver_u = create_solver()
solver_d = create_solver()

# =============================================================================
# 6. BOUCLE TEMPORELLE
# =============================================================================
folder_name = "resultats_titanic_AT1"

# Récupère le chemin absolu du dossier où se trouve ce script Python
script_dir = os.path.dirname(os.path.abspath(__file__))

# Crée le chemin du dossier de résultats à l'intérieur de ce dossier
output_dir = os.path.join(script_dir, folder_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

export_path = os.path.join(output_dir, "titanic_simulation.bp")
vtx = dolfinx.io.VTXWriter(domain.comm, export_path, [u, d], engine="BP4")

print(f"Démarrage de la simulation AT1. Les fichiers seront dans : {export_path}")
vtx.write(0.0)

steps = 100
max_traction = 300e6
tractions = np.linspace(0, max_traction, steps)

# Tolérance pour la convergence
tol = 1e-4

for step, t_val in enumerate(tractions):
    T_mag.value = t_val
    
    max_iter = 10
    iter_count = 0
    
    for i in range(max_iter):
        iter_count += 1
        d_prev = d.x.array.copy()
        
        # --- 1. Résolution de u (Assemblage manuel) ---
        A_u = fem.petsc.assemble_matrix(form_a_u, bcs=bcs_u)
        A_u.assemble()
        b_u = fem.petsc.assemble_vector(form_L_u)
        fem.petsc.apply_lifting(b_u, [form_a_u], [bcs_u])
        b_u.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_u, bcs_u)
        
        solver_u.setOperators(A_u)
        solver_u.solve(b_u, u.x.petsc_vec)
        u.x.scatter_forward()
        
        # --- 2. Résolution de d (Assemblage manuel) ---
        A_d = fem.petsc.assemble_matrix(form_a_d)
        A_d.assemble()
        b_d = fem.petsc.assemble_vector(form_L_d)
        b_d.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        
        solver_d.setOperators(A_d)
        solver_d.solve(b_d, d.x.petsc_vec)
        d.x.scatter_forward()
        
        # --- 3. Irréversibilité et bornes ---
        d.x.array[:] = np.maximum(d.x.array, d_old.x.array)
        d.x.array[:] = np.minimum(d.x.array, 1.0)
        d.x.array[:] = np.maximum(d.x.array, 0.0)
        
        # --- 4. Vérification de la convergence ---
        error = np.linalg.norm(d.x.array - d_prev)
        if error < tol:
            break

    # Mise à jour de l'historique
    d_old.x.array[:] = d.x.array
    
    vtx.write(step+1)
    
    max_d = np.max(d.x.array)
    max_u = np.max(np.abs(u.x.array))
    
    print(f"Step {step}: Force={t_val/1e6:.1f} MPa | U={max_u:.2e} m | Damage={max_d:.4f} | (Itér: {iter_count})")
    
    if max_d > 0.99:
        print("Fissure complète détectée !")
        break

vtx.close()
print("Terminé.")