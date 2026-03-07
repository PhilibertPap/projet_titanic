from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import basix
import numpy as np
import ufl
from dolfinx import fem


# ============================================================================
# OUTILS GEOMETRIE / CHAMPS
# ============================================================================

def _normalize(v):
    return v / ufl.sqrt(ufl.dot(v, v))


def _local_frame(domain):
    jac = ufl.Jacobian(domain)
    if domain.geometry.dim == 2:
        t1 = ufl.as_vector([jac[0, 0], jac[1, 0], 0])
        t2 = ufl.as_vector([jac[0, 1], jac[1, 1], 0])
    else:
        t1 = ufl.as_vector([jac[0, 0], jac[1, 0], jac[2, 0]])
        t2 = ufl.as_vector([jac[0, 1], jac[1, 1], jac[2, 1]])

    e3 = _normalize(ufl.cross(t1, t2))
    ey = ufl.as_vector([0, 1, 0])
    ez = ufl.as_vector([0, 0, 1])
    e1 = ufl.cross(ey, e3)
    norm_e1 = ufl.sqrt(ufl.dot(e1, e1))
    e1 = ufl.conditional(ufl.lt(norm_e1, 0.5), ez, _normalize(e1))
    e2 = _normalize(ufl.cross(e3, e1))
    return e1, e2, e3


def _creer_champ_scalaire_dg0(domain, name: str, value: float):
    V0 = fem.functionspace(domain, ("DG", 0))
    field = fem.Function(V0, name=name)
    field.x.array[:] = value
    return field


# ============================================================================
# BANDES RIVETS
# ============================================================================

def _rectangles_bandes_rivets(bandes_cfg):
    rectangles = []
    for bande in bandes_cfg:
        if "x_centre_m" not in bande:
            raise ValueError("Each rivet band must define 'x_centre_m'")

        xc = float(bande["x_centre_m"])
        largeur_x = float(bande.get("largeur_x_m", 0.30))
        xmin = xc - 0.5 * largeur_x
        xmax = xc + 0.5 * largeur_x

        zmin = float(bande.get("z_min_m", -np.inf))
        zmax = float(bande.get("z_max_m", np.inf))
        if zmax < zmin:
            zmin, zmax = zmax, zmin

        rectangles.append({"xmin": xmin, "xmax": xmax, "zmin": zmin, "zmax": zmax})
    return rectangles


def _interpoler_bandes_rectangles(domain, bandes_cfg, space, name: str, default: float, value_from_band):
    field = fem.Function(space, name=name)
    field.x.array[:] = default
    if not bandes_cfg:
        return field

    bandes = _rectangles_bandes_rivets(bandes_cfg)
    for bande, rect in zip(bandes_cfg, bandes):
        rect["valeur"] = float(value_from_band(bande))

    def values(x):
        xcoord = x[0]
        z = x[2]
        out = np.full_like(z, float(default), dtype=float)
        for bande in bandes:
            mask = (
                (z >= bande["zmin"])
                & (z <= bande["zmax"])
                & (xcoord >= bande["xmin"])
                & (xcoord <= bande["xmax"])
            )
            out[mask] = bande["valeur"]
        return out

    field.interpolate(values)
    return field


def _champ_facteur_bandes_rivets(domain, bandes_cfg, nom_facteur: str, default: float = 1.0):
    V0 = fem.functionspace(domain, ("DG", 0))
    return _interpoler_bandes_rectangles(
        domain,
        bandes_cfg,
        V0,
        name=f"{nom_facteur}_BandesRivets",
        default=default,
        value_from_band=lambda bande: bande.get(nom_facteur, default),
    )


def _champ_masque_bandes_rivets(domain, bandes_cfg):
    V0 = fem.functionspace(domain, ("DG", 0))
    return _interpoler_bandes_rectangles(
        domain,
        bandes_cfg,
        V0,
        name="RivetBandsMask",
        default=0.0,
        value_from_band=lambda _: 1.0,
    )


def _champ_masque_bandes_rivets_viz(domain, bandes_cfg):
    Vviz = fem.functionspace(domain, ("CG", 1))
    return _interpoler_bandes_rectangles(
        domain,
        bandes_cfg,
        Vviz,
        name="RivetBandsMaskViz",
        default=0.0,
        value_from_band=lambda _: 1.0,
    )


# ============================================================================
# BUILDERS MODELE COQUE
# ============================================================================

def _construire_champs_materiaux(domain, cell_tags, cfg):
    E = _creer_champ_scalaire_dg0(domain, "YoungModulus", cfg.young_coque)
    nu = _creer_champ_scalaire_dg0(domain, "PoissonRatio", cfg.poisson_coque)
    thick = _creer_champ_scalaire_dg0(domain, "Thickness", cfg.epaisseur_coque)

    if cell_tags is not None:
        rivet_cells = cell_tags.find(cfg.tag_cellule_rivet)
        if rivet_cells is not None and len(rivet_cells) > 0:
            E.x.array[rivet_cells] = cfg.young_rivet
            nu.x.array[rivet_cells] = cfg.poisson_rivet
            thick.x.array[rivet_cells] = cfg.epaisseur_rivet

    if cfg.utiliser_bandes_rivets_z:
        bandes = list(cfg.bandes_rivets_z)
        facteur_E = _champ_facteur_bandes_rivets(domain, bandes, "facteur_E", 1.0)
        facteur_t = _champ_facteur_bandes_rivets(domain, bandes, "facteur_epaisseur", 1.0)
        E.x.array[:] *= facteur_E.x.array
        thick.x.array[:] *= facteur_t.x.array

    return E, nu, thick


def _construire_base_locale(domain, gdim):
    VT = fem.functionspace(domain, ("DG", 0, (gdim,)))
    V0, _ = VT.sub(0).collapse()
    frame_expr = _local_frame(domain)

    basis_vectors = [fem.Function(VT, name=f"Basis_vector_e{i+1}") for i in range(gdim)]
    for i in range(gdim):
        e_expr = fem.Expression(frame_expr[i], V0.element.interpolation_points)
        basis_vectors[i].interpolate(e_expr)
    return basis_vectors


def _construire_espaces_coque(domain, gdim):
    Ue = basix.ufl.element("P", domain.basix_cell(), 2, shape=(gdim,))
    Te = basix.ufl.element("CR", domain.basix_cell(), 1, shape=(gdim,))
    V = fem.functionspace(domain, basix.ufl.mixed_element([Ue, Te]))
    Vu, _ = V.sub(0).collapse()
    Vtheta, _ = V.sub(1).collapse()
    Vd = fem.functionspace(domain, ("CG", 1))
    return V, Vu, Vtheta, Vd


def _construire_conditions_limites(V, Vu, Vtheta, facets, cfg):
    edge_tags = [cfg.tag_facet_gauche, cfg.tag_facet_droite]
    if cfg.encastrer_tous_bords:
        edge_tags.extend([cfg.tag_facet_bas, cfg.tag_facet_haut])
    edge_tags = list(dict.fromkeys(edge_tags))

    uD = fem.Function(Vu)
    thetaD = fem.Function(Vtheta)
    bcs = []

    for facet_tag in edge_tags:
        tagged_facets = facets.find(facet_tag)
        disp_dofs = fem.locate_dofs_topological((V.sub(0), Vu), 1, tagged_facets)
        bcs.append(fem.dirichletbc(uD, disp_dofs, V.sub(0)))

        if cfg.encastrer_rotations:
            rot_dofs = fem.locate_dofs_topological((V.sub(1), Vtheta), 1, tagged_facets)
            bcs.append(fem.dirichletbc(thetaD, rot_dofs, V.sub(1)))

    return bcs


@dataclass
class ModeleCoque:
    domain: Any
    facets: Any
    gdim: int
    tdim: int
    V: Any
    Vu: Any
    Vtheta: Any
    Vd: Any
    v: Any
    u_test: Any
    a: Any
    bcs: list
    e1: Any
    e2: Any
    e3: Any
    E_field: Any
    nu_field: Any
    thick_field: Any
    gc_factor_field: Any
    rivet_bands_mask_field: Any
    rivet_bands_mask_viz_field: Any
    damage_state: Any


# ============================================================================
# MODELE COQUE (pipeline lisible type TD)
# ============================================================================

def construire_modele_coque(domain, cell_tags, facets, cfg) -> ModeleCoque:
    if facets is None:
        raise ValueError(
            "Facet tags are required to build shell boundary conditions, but facet_tags is None. "
            "Regenerate the mesh with physical groups on boundary facets/edges."
        )

    # 1) Champs materiaux
    gdim = domain.geometry.dim
    tdim = domain.topology.dim
    E, nu, thick = _construire_champs_materiaux(domain, cell_tags, cfg)

    lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
    mu = E / 2 / (1 + nu)
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)

    # 2) Base locale + espaces + inconnues
    e1, e2, e3 = _construire_base_locale(domain, gdim)
    V, Vu, Vtheta, Vd = _construire_espaces_coque(domain, gdim)

    v = fem.Function(V)
    u, theta = ufl.split(v)
    v_test = ufl.TestFunction(V)
    u_test, _ = ufl.split(v_test)
    dv = ufl.TrialFunction(V)

    P_plane = ufl.as_matrix(
        [
            [e1[0], e2[0]],
            [e1[1], e2[1]],
            [e1[2], e2[2]],
        ]
    )

    def t_grad(field):
        return ufl.dot(ufl.grad(field), P_plane)

    # 3) Cinematique coque
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

    N = thick * plane_stress_elasticity(eps)
    M = thick**3 / 12 * plane_stress_elasticity(kappa)
    Q = mu * thick * gamma

    drilling_strain = (t_gu[0, 1] - t_gu[1, 0]) / 2 + ufl.dot(theta, e3)
    drilling_strain_test = ufl.replace(drilling_strain, {v: v_test})
    h_mesh = ufl.CellDiameter(domain)
    drilling_stiffness = E * thick**3 / h_mesh**2
    drilling_stress = drilling_stiffness * drilling_strain

    # 4) Couplage dommage global
    damage_state = fem.Function(Vd, name="DamageState")
    damage_state.x.array[:] = 0.0

    bandes_gc = list(cfg.bandes_rivets_z) if cfg.utiliser_bandes_rivets_z else []
    gc_factor_field = _champ_facteur_bandes_rivets(domain, bandes_gc, "facteur_Gc", 1.0)
    rivet_bands_mask_field = _champ_masque_bandes_rivets(domain, bandes_gc)
    rivet_bands_mask_viz_field = _champ_masque_bandes_rivets_viz(domain, bandes_gc)

    k_res_mech = fem.Constant(
        domain,
        cfg.phase_field_raideur_residuelle if cfg.activer_phase_field_global else 0.0,
    )
    degradation = (1.0 - damage_state) ** 2 + k_res_mech

    # 5) CL + forme variationnelle
    bcs = _construire_conditions_limites(V, Vu, Vtheta, facets, cfg)

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

    return ModeleCoque(
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


# Alias de compatibilite
ShellModel = ModeleCoque
build_shell_model = construire_modele_coque
