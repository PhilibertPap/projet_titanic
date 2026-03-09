from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import dolfinx.io.gmsh
import gmsh
import numpy as np
import petsc4py.PETSc as PETSc
import ufl
from dolfinx import default_scalar_type, fem
import dolfinx.fem.petsc
from mpi4py import MPI


def config_par_defaut() -> dict:
    return {
        # Geometrie / maillage local rivet
        "L": 0.5,
        "W": 0.2,
        "H": 0.01,
        "r": 0.0165,
        "lc_fine": 0.003,
        "lc_coarse": 0.01,
        # Materiau / phase-field AT1 (inchange)
        "E": 190e9,
        "nu": 0.3,
        "Gc": 2400,
        "l0": 0.005,
        "k_ell": 1e-6,
        "eps_stab_factor": 1e-8,
        # Chargement / temps
        "steps": 100,
        "max_traction": 100e6,
        "max_iter": 10,
        "tol": 1e-4,
        # Sorties
        "resultats_dossier": "resultats_titanic_AT1",
        "export_filename": "titanic_simulation.bp",
    }


def creer_config(**updates):
    data = config_par_defaut()
    data.update(updates)
    return SimpleNamespace(**data)


def config_vers_dict(cfg) -> dict:
    return dict(vars(cfg))


DEFAULT_CONFIG = creer_config()


def create_titanic_rivet_mesh(
    L=0.5,
    W=0.2,
    H=0.01,
    r=0.02,
    lc_fine=0.003,
    lc_coarse=0.01,
):
    gmsh.initialize()
    model = gmsh.model()
    model.add("Plaque_Rivet")

    factory = model.occ
    box = factory.addBox(-L / 2, -W / 2, -H / 2, L, W, H)
    cylinder = factory.addCylinder(0, 0, -H, 0, 0, 2 * H, r)

    factory.cut([(3, box)], [(3, cylinder)])
    factory.synchronize()

    curves = model.getEntities(1)
    hole_curves = []
    for c in curves:
        com = model.occ.getCenterOfMass(1, c[1])
        dist_from_center = np.sqrt(com[0] ** 2 + com[1] ** 2)
        if dist_from_center < r * 1.5:
            hole_curves.append(c[1])

    model.mesh.field.add("Distance", 1)
    model.mesh.field.setNumbers(1, "CurvesList", hole_curves)
    model.mesh.field.setNumber(1, "Sampling", 100)

    model.mesh.field.add("Threshold", 2)
    model.mesh.field.setNumber(2, "InField", 1)
    model.mesh.field.setNumber(2, "SizeMin", lc_fine)
    model.mesh.field.setNumber(2, "SizeMax", lc_coarse)
    model.mesh.field.setNumber(2, "DistMin", r)
    model.mesh.field.setNumber(2, "DistMax", r + 0.05)
    model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    eps = 1e-6
    surfaces = model.getEntities(2)
    left_tags = []
    right_tags = []
    for s in surfaces:
        com = model.occ.getCenterOfMass(2, s[1])
        if np.abs(com[0] + L / 2) < eps:
            left_tags.append(s[1])
        elif np.abs(com[0] - L / 2) < eps:
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


def _create_solver(comm):
    solver = PETSc.KSP().create(comm)
    solver.setType("preonly")
    solver.getPC().setType("lu")
    return solver


def _build_output_path(cfg) -> Path:
    script_dir = Path(__file__).resolve().parent
    output_dir = script_dir / cfg.resultats_dossier
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / cfg.export_filename


def creer_preset_bandes_grande_echelle(
    path: str | Path | None = None,
    *,
    x_start_m: float = 177.0,
    x_end_m: float = 268.0,
    n_bandes_x: int = 8,
    largeur_x_m: float = 0.30,
    z_min_m: float = -10.2,
    z_max_m: float = 0.2,
    facteur_E: float = 0.98,
    facteur_epaisseur: float = 1.00,
    facteur_Gc: float = 0.88,
):
    """Ecrit un preset JSON simple pour grande_echelle (homogeneisation en bandes)."""
    x_margin = 0.5 * largeur_x_m
    x_centres = np.linspace(x_start_m + x_margin, x_end_m - x_margin, n_bandes_x)
    bandes_rivets = []
    for i_bande, x_centre in enumerate(x_centres, start=1):
        bandes_rivets.append(
            {
                "nom": f"bande_x_{i_bande:02d}",
                "x_centre_m": float(x_centre),
                "largeur_x_m": float(largeur_x_m),
                "z_min_m": float(z_min_m),
                "z_max_m": float(z_max_m),
                "facteur_E": float(facteur_E),
                "facteur_epaisseur": float(facteur_epaisseur),
                "facteur_Gc": float(facteur_Gc),
            }
        )

    if path is None:
        path = Path(__file__).resolve().parent / "bandes_rivets_grande_echelle.json"
    else:
        path = Path(path)

    payload = {
        "format": "titanic-rivet-bands/v1",
        "origine": "template_simple_depuis_rivet_local",
        "bandes_rivets_z": bandes_rivets,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Preset grande_echelle ecrit : {path}")
    return path


def lancer_calcul(cfg=None):
    cfg = DEFAULT_CONFIG if cfg is None else cfg

    # 1) Maillage
    mdata = create_titanic_rivet_mesh(
        L=cfg.L,
        W=cfg.W,
        H=cfg.H,
        r=cfg.r,
        lc_fine=cfg.lc_fine,
        lc_coarse=cfg.lc_coarse,
    )
    domain = mdata.mesh
    facet_tags = mdata.facet_tags

    # 2) Parametres physiques (inchanges)
    E = cfg.E
    nu = cfg.nu
    Gc = cfg.Gc
    l0 = cfg.l0
    k_ell = cfg.k_ell

    cw = 8.0 / 3.0
    seuil_critique = Gc / (cw * l0)
    mu = E / (2.0 * (1.0 + nu))
    lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    # 3) Espaces / champs
    V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    D = fem.functionspace(domain, ("Lagrange", 1))

    u = fem.Function(V, name="Deplacement")
    d = fem.Function(D, name="Endommagement")
    d_old = fem.Function(D)

    ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
    dx = ufl.Measure("dx", domain=domain)

    # 4) Conditions aux limites
    fdim = domain.topology.dim - 1
    left_facets = facet_tags.find(2)
    dofs_left = fem.locate_dofs_topological(V, fdim, left_facets)
    bcs_u = [fem.dirichletbc(np.array([0, 0, 0], dtype=default_scalar_type), dofs_left, V)]

    # 5) Formulation variationnelle AT1 (inchangée)
    def g(damage):
        return (1 - damage) ** 2 + k_ell

    def epsilon(disp):
        return ufl.sym(ufl.grad(disp))

    def sigma_undamaged(disp):
        return 2.0 * mu * epsilon(disp) + lmbda * ufl.tr(epsilon(disp)) * ufl.Identity(len(disp))

    psi_elastic = 0.5 * ufl.inner(sigma_undamaged(u), epsilon(u))

    u_trial = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    T_mag = fem.Constant(domain, default_scalar_type(0.0))
    a_u = g(d) * ufl.inner(sigma_undamaged(u_trial), epsilon(v)) * dx
    L_u = ufl.dot(T_mag * ufl.as_vector([1.0, 0.0, 0.0]), v) * ds(3)
    form_a_u = fem.form(a_u)
    form_L_u = fem.form(L_u)

    d_trial = ufl.TrialFunction(D)
    w = ufl.TestFunction(D)
    eps_stab = cfg.eps_stab_factor * (Gc / l0)
    psi_eff = ufl.max_value(psi_elastic, 0.0)
    a_d = (
        (2.0 * psi_eff + eps_stab) * ufl.inner(d_trial, w) * dx
        + (Gc * l0 / cw) * ufl.inner(ufl.grad(d_trial), ufl.grad(w)) * dx
    )
    terme_moteur_at1 = ufl.max_value(2.0 * psi_eff - seuil_critique, 0.0)
    L_d = terme_moteur_at1 * w * dx
    form_a_d = fem.form(a_d)
    form_L_d = fem.form(L_d)

    solver_u = _create_solver(domain.comm)
    solver_d = _create_solver(domain.comm)

    # 6) Sorties
    export_path = _build_output_path(cfg)
    vtx = dolfinx.io.VTXWriter(domain.comm, str(export_path), [u, d], engine="BP4")
    print(f"Demarrage de la simulation AT1. Les fichiers seront dans : {export_path}")
    vtx.write(0.0)

    # 7) Boucle de chargement (inchangée)
    tractions = np.linspace(0.0, cfg.max_traction, cfg.steps)
    max_damage_final = 0.0
    max_u_final = 0.0
    last_step = -1
    last_traction = 0.0
    rupture_detected = False
    rupture_traction = None

    for step, t_val in enumerate(tractions):
        T_mag.value = t_val
        iter_count = 0

        for _ in range(cfg.max_iter):
            iter_count += 1
            d_prev = d.x.array.copy()

            A_u = dolfinx.fem.petsc.assemble_matrix(form_a_u, bcs=bcs_u)
            A_u.assemble()
            b_u = dolfinx.fem.petsc.assemble_vector(form_L_u)
            dolfinx.fem.petsc.apply_lifting(b_u, [form_a_u], [bcs_u])
            b_u.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            dolfinx.fem.petsc.set_bc(b_u, bcs_u)
            solver_u.setOperators(A_u)
            solver_u.solve(b_u, u.x.petsc_vec)
            u.x.scatter_forward()

            A_d = dolfinx.fem.petsc.assemble_matrix(form_a_d)
            A_d.assemble()
            b_d = dolfinx.fem.petsc.assemble_vector(form_L_d)
            b_d.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            solver_d.setOperators(A_d)
            solver_d.solve(b_d, d.x.petsc_vec)
            d.x.scatter_forward()

            d.x.array[:] = np.maximum(d.x.array, d_old.x.array)
            d.x.array[:] = np.minimum(d.x.array, 1.0)
            d.x.array[:] = np.maximum(d.x.array, 0.0)

            error = np.linalg.norm(d.x.array - d_prev)
            if error < cfg.tol:
                break

        d_old.x.array[:] = d.x.array
        vtx.write(step + 1)

        max_d = float(np.max(d.x.array))
        max_u = float(np.max(np.abs(u.x.array)))
        max_damage_final = max_d
        max_u_final = max_u
        last_step = step
        last_traction = float(t_val)

        print(
            f"Step {step}: Force={t_val/1e6:.1f} MPa | U={max_u:.2e} m | "
            f"Damage={max_d:.4f} | (Iter: {iter_count})"
        )

        if max_d > 0.99:
            print("Fissure complete detectee !")
            rupture_detected = True
            rupture_traction = float(t_val)
            break

    vtx.close()
    summary_path = export_path.parent / "run_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": config_vers_dict(cfg),
                "export_path": str(export_path),
                "last_step": last_step,
                "last_traction_pa": last_traction,
                "max_u_final_m": max_u_final,
                "max_damage_final": max_damage_final,
                "rupture_detected": rupture_detected,
                "rupture_traction_pa": rupture_traction,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Resume ecrit : {summary_path}")
    print("Termine.")
    return {
        "export_path": str(export_path),
        "output_dir": str(export_path.parent),
        "last_step": last_step,
        "last_traction_pa": last_traction,
        "max_u_final_m": max_u_final,
        "max_damage_final": max_damage_final,
        "rupture_detected": rupture_detected,
        "rupture_traction_pa": rupture_traction,
        "summary_path": str(summary_path),
    }


run = lancer_calcul


if __name__ == "__main__":
    lancer_calcul()
