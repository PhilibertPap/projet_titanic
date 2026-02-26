from dataclasses import dataclass

import basix
import numpy as np
import ufl
from dolfinx import fem


def _normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))


def _local_frame(domain):
    t = ufl.Jacobian(domain)
    if domain.geometry.dim == 2:
        t1 = ufl.as_vector([t[0, 0], t[1, 0], 0])
        t2 = ufl.as_vector([t[0, 1], t[1, 1], 0])
    else:
        t1 = ufl.as_vector([t[0, 0], t[1, 0], t[2, 0]])
        t2 = ufl.as_vector([t[0, 1], t[1, 1], t[2, 1]])

    e3 = _normalize(ufl.cross(t1, t2))
    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1 = ufl.cross(ey, e3)
    norm_e1 = ufl.sqrt(ufl.dot(e1, e1))
    e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, _normalize(e1))
    e2 = _normalize(ufl.cross(e3, e1))
    return e1, e2, e3


def _make_dg0_scalar_function(domain, name: str, value: float):
    V0 = fem.functionspace(domain, ("DG", 0))
    field = fem.Function(V0, name=name)
    field.x.array[:] = value
    return field


def _bandes_rivets_rectangles(bandes_cfg):
    """
    Convertit la config des bandes rivets en rectangles (x,z) sur la coque.

    Formats supportes:
    - legacy: z_centre_m + largeur_m (+ x_debut_m/x_fin_m optionnels)
    - nouveau: x_centre_m + largeur_x_m (+ z_min_m/z_max_m)
    """
    rectangles = []
    for bande in bandes_cfg:
        # x-interval
        if "x_centre_m" in bande:
            xc = float(bande["x_centre_m"])
            largeur_x = float(bande.get("largeur_x_m", bande.get("largeur_m", 0.30)))
            xmin = xc - 0.5 * largeur_x
            xmax = xc + 0.5 * largeur_x
        elif "x_debut_m" in bande and "x_fin_m" in bande:
            xmin = min(float(bande["x_debut_m"]), float(bande["x_fin_m"]))
            xmax = max(float(bande["x_debut_m"]), float(bande["x_fin_m"]))
        else:
            xmin = -np.inf
            xmax = np.inf

        # z-interval
        if "z_centre_m" in bande:
            zc = float(bande["z_centre_m"])
            largeur_z = float(bande.get("largeur_m", bande.get("largeur_z_m", 0.30)))
            zmin = zc - 0.5 * largeur_z
            zmax = zc + 0.5 * largeur_z
        else:
            zmin = float(bande.get("z_min_m", -np.inf))
            zmax = float(bande.get("z_max_m", np.inf))
            if zmax < zmin:
                zmin, zmax = zmax, zmin

        rectangles.append(
            {
                "xmin": xmin,
                "xmax": xmax,
                "zmin": zmin,
                "zmax": zmax,
            }
        )
    return rectangles


def _champ_facteur_bandes_rivets(domain, bandes_cfg, nom_facteur: str, default: float = 1.0):
    V0 = fem.functionspace(domain, ("DG", 0))
    facteur = fem.Function(V0, name=f"{nom_facteur}_BandesRivets")
    facteur.x.array[:] = default
    if not bandes_cfg:
        return facteur

    bandes = _bandes_rivets_rectangles(bandes_cfg)
    for bande, rect in zip(bandes_cfg, bandes):
        rect["valeur"] = float(bande.get(nom_facteur, default))

    def valeur_par_bande(x):
        xcoord = x[0]
        z = x[2]
        out = np.full_like(z, float(default), dtype=float)
        for bande in bandes:
            masque = (
                (z >= bande["zmin"])
                & (z <= bande["zmax"])
                & (xcoord >= bande["xmin"])
                & (xcoord <= bande["xmax"])
            )
            out[masque] = bande["valeur"]
        return out

    facteur.interpolate(valeur_par_bande)
    return facteur


def _champ_masque_bandes_rivets(domain, bandes_cfg):
    """Masque DG0: 1 dans les bandes rivets homogenisees, 0 ailleurs."""
    V0 = fem.functionspace(domain, ("DG", 0))
    masque = fem.Function(V0, name="RivetBandsMask")
    masque.x.array[:] = 0.0
    if not bandes_cfg:
        return masque

    bands = _bandes_rivets_rectangles(bandes_cfg)

    def valeur_masque(x):
        xcoord = x[0]
        z = x[2]
        out = np.zeros_like(z, dtype=float)
        for bande in bands:
            masque = (
                (z >= bande["zmin"])
                & (z <= bande["zmax"])
                & (xcoord >= bande["xmin"])
                & (xcoord <= bande["xmax"])
            )
            out[masque] = 1.0
        return out

    masque.interpolate(valeur_masque)
    return masque


def _champ_masque_bandes_rivets_viz(domain, bandes_cfg):
    """Masque de visualisation (CG1) pour ParaView, plus lisible que DG0."""
    Vviz = fem.functionspace(domain, ("CG", 1))
    mask_viz = fem.Function(Vviz, name="RivetBandsMaskViz")
    mask_viz.x.array[:] = 0.0
    if not bandes_cfg:
        return mask_viz

    bands = _bandes_rivets_rectangles(bandes_cfg)

    def valeur_masque_viz(x):
        xcoord = x[0]
        z = x[2]
        out = np.zeros_like(z, dtype=float)
        for bande in bands:
            masque = (
                (z >= bande["zmin"])
                & (z <= bande["zmax"])
                & (xcoord >= bande["xmin"])
                & (xcoord <= bande["xmax"])
            )
            out[masque] = 1.0
        return out

    mask_viz.interpolate(valeur_masque_viz)
    return mask_viz


def _build_material_fields(domain, cell_tags, cfg):
    """
    Champs matériaux par cellules (coque + bande rivets via tags).
    Gardé ici pour rendre le modèle coque plus autoportant.
    """
    E = _make_dg0_scalar_function(domain, "YoungModulus", cfg.shell_young_modulus)
    nu = _make_dg0_scalar_function(domain, "PoissonRatio", cfg.shell_poisson_ratio)
    thick = _make_dg0_scalar_function(domain, "Thickness", cfg.shell_thickness)

    if cell_tags is not None:
        rivet_cells = cell_tags.find(cfg.rivet_cell_tag)
        if rivet_cells is not None and len(rivet_cells) > 0:
            E.x.array[rivet_cells] = cfg.rivet_young_modulus
            nu.x.array[rivet_cells] = cfg.rivet_poisson_ratio
            thick.x.array[rivet_cells] = cfg.rivet_thickness

    if getattr(cfg, "utiliser_bandes_rivets_z", False):
        bandes = list(getattr(cfg, "bandes_rivets_z", []))
        facteur_E = _champ_facteur_bandes_rivets(domain, bandes, "facteur_E", 1.0)
        facteur_t = _champ_facteur_bandes_rivets(domain, bandes, "facteur_epaisseur", 1.0)
        E.x.array[:] *= facteur_E.x.array
        thick.x.array[:] *= facteur_t.x.array

    return E, nu, thick


def _build_local_basis_fields(domain, gdim):
    """Base locale (e1, e2, e3) interpolee sur un espace DG0 vectoriel."""
    VT = fem.functionspace(domain, ("DG", 0, (gdim,)))
    V0, _ = VT.sub(0).collapse()
    frame_expr = _local_frame(domain)
    basis_vectors = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(gdim)]
    for i in range(gdim):
        e_exp = fem.Expression(frame_expr[i], V0.element.interpolation_points)
        basis_vectors[i].interpolate(e_exp)
    return basis_vectors


def _build_shell_spaces(domain, gdim):
    """Espaces EF du modele de coque (deplacement + rotation + dommage)."""
    Ue = basix.ufl.element("P", domain.basix_cell(), 2, shape=(gdim,))
    Te = basix.ufl.element("CR", domain.basix_cell(), 1, shape=(gdim,))
    V = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))
    Vu, _ = V.sub(0).collapse()
    Vtheta, _ = V.sub(1).collapse()
    Vd = fem.functionspace(domain, ("CG", 1))
    return V, Vu, Vtheta, Vd


def _build_boundary_conditions(V, Vu, Vtheta, facets, cfg):
    """Conditions aux limites encastrees sur les bords tagges."""
    edge_tags = [cfg.left_facet_tag, cfg.right_facet_tag]
    if cfg.clamp_all_edges:
        edge_tags.extend([cfg.bottom_facet_tag, cfg.top_facet_tag])
    edge_tags = list(dict.fromkeys(edge_tags))

    uD = fem.Function(Vu)
    thetaD = fem.Function(Vtheta)
    bcs = []
    for facet_tag in edge_tags:
        tagged_facets = facets.find(facet_tag)
        disp_dofs = fem.locate_dofs_topological((V.sub(0), Vu), 1, tagged_facets)
        bcs.append(fem.dirichletbc(uD, disp_dofs, V.sub(0)))
        if cfg.clamp_rotations:
            rot_dofs = fem.locate_dofs_topological((V.sub(1), Vtheta), 1, tagged_facets)
            bcs.append(fem.dirichletbc(thetaD, rot_dofs, V.sub(1)))
    return bcs


@dataclass
class ShellModel:
    domain: any
    facets: any
    gdim: int
    tdim: int
    V: any
    Vu: any
    Vtheta: any
    Vd: any
    v: any
    u_test: any
    a: any
    bcs: list
    e1: any
    e2: any
    e3: any
    E_field: any
    nu_field: any
    thick_field: any
    gc_factor_field: any
    rivet_bands_mask_field: any
    rivet_bands_mask_viz_field: any
    damage_state: any


def build_shell_model(domain, cell_tags, facets, cfg) -> ShellModel:
    # ------------------------------------------------------------
    # 1) Geometrie / champs materiaux
    # ------------------------------------------------------------
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    if facets is None:
        raise ValueError(
            "Facet tags are required to build shell boundary conditions, but facet_tags is None. "
            "Regenerate the mesh with physical groups on boundary facets/edges."
        )

    E, nu, thick = _build_material_fields(domain, cell_tags, cfg)
    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)

    # Base locale + espaces EF (organisation proche d'un TD)
    e1, e2, e3 = _build_local_basis_fields(domain, gdim)
    V, Vu, Vtheta, Vd = _build_shell_spaces(domain, gdim)

    v = fem.Function(V)
    u, theta = ufl.split(v)
    v_test = ufl.TestFunction(V)
    u_test, theta_test = ufl.split(v_test)
    dv = ufl.TrialFunction(V)

    # Matrice 3x2 de projection sur le plan tangent local [e1 e2]
    P_plane = ufl.as_matrix(
        [
            [e1[0], e2[0]],
            [e1[1], e2[1]],
            [e1[2], e2[2]],
        ]
    )

    def t_grad(field):
        grad_field = ufl.grad(field)
        return ufl.dot(grad_field, P_plane)

    # ------------------------------------------------------------
    # 2) Cinematique coque (membrane / flexion / cisaillement)
    # ------------------------------------------------------------
    t_gu = ufl.dot(P_plane.T, t_grad(u))
    eps = ufl.sym(t_gu)
    beta = ufl.cross(e3, theta)
    kappa = ufl.sym(ufl.dot(P_plane.T, t_grad(beta)))
    gamma = t_grad(ufl.dot(u, e3)) - ufl.dot(P_plane.T, beta)

    eps_test = ufl.derivative(eps, v, v_test)
    kappa_test = ufl.derivative(kappa, v, v_test)
    gamma_test = ufl.derivative(gamma, v, v_test)

    def plane_stress_elasticity(strain):
        return lmbda_ps * ufl.tr(strain) * ufl.Identity(tdim) + 2 * mu * strain

    # Efforts generalises coque
    N = thick * plane_stress_elasticity(eps)
    M = thick**3 / 12 * plane_stress_elasticity(kappa)
    Q = mu * thick * gamma

    # Stabilisation drilling rotation (forme simple)
    drilling_strain = (t_gu[0, 1] - t_gu[1, 0]) / 2 + ufl.dot(theta, e3)
    drilling_strain_test = ufl.replace(drilling_strain, {v: v_test})
    h_mesh = ufl.CellDiameter(domain)
    drilling_stiffness = E * thick**3 / h_mesh**2
    drilling_stress = drilling_stiffness * drilling_strain

    damage_state = fem.Function(Vd, name="DamageState")
    damage_state.x.array[:] = 0.0
    if getattr(cfg, "utiliser_bandes_rivets_z", False):
        bandes_gc = list(getattr(cfg, "bandes_rivets_z", []))
    else:
        bandes_gc = []
    gc_factor_field = _champ_facteur_bandes_rivets(domain, bandes_gc, "facteur_Gc", 1.0)
    rivet_bands_mask_field = _champ_masque_bandes_rivets(domain, bandes_gc)
    rivet_bands_mask_viz_field = _champ_masque_bandes_rivets_viz(domain, bandes_gc)
    k_res_mech = fem.Constant(
        domain,
        cfg.phase_field_residual_stiffness if cfg.enable_global_phase_field else 0.0,
    )
    degradation = (1.0 - damage_state) ** 2 + k_res_mech

    # ------------------------------------------------------------
    # 3) Conditions aux limites
    # ------------------------------------------------------------
    bcs = _build_boundary_conditions(V, Vu, Vtheta, facets, cfg)

    # Couplage staggered avec le phase-field global :
    # la rigidite mecanique est degradee par le dommage courant.
    Wdef = (
        degradation
        * (
            ufl.inner(N, eps_test)
            + ufl.inner(M, kappa_test)
            + ufl.dot(Q, gamma_test)
            + drilling_stress * drilling_strain_test
        )
    ) * ufl.dx
    a = ufl.derivative(Wdef, v, dv)

    return ShellModel(
        domain=domain,
        facets=facets,
        gdim=gdim,
        tdim=tdim,
        V=V,
        Vu=Vu,
        Vtheta=Vtheta,
        Vd=Vd,
        v=v,
        u_test=u_test,
        a=a,
        bcs=bcs,
        e1=e1,
        e2=e2,
        e3=e3,
        E_field=E,
        nu_field=nu,
        thick_field=thick,
        gc_factor_field=gc_factor_field,
        rivet_bands_mask_field=rivet_bands_mask_field,
        rivet_bands_mask_viz_field=rivet_bands_mask_viz_field,
        damage_state=damage_state,
    )
