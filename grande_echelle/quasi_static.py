from __future__ import annotations

from contextlib import nullcontext
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc


# ============================================================================
# OUTILS CSV / METRIQUES
# ============================================================================

def _ecrire_csv_suivi(path, rows):
    lines = [
        "step,time,max_u_inf,max_damage,mean_damage,frac_damage_ge_095,"
        "temps_pas_s,temps_meca_s,temps_phase_field_s"
    ]
    for row in rows:
        step, time_value, max_u, max_d, mean_d, frac_d95, step_wall, mech_wall, damage_wall = row
        lines.append(
            f"{step},{time_value:.12g},{max_u:.12e},{max_d:.12e},{mean_d:.12e},{frac_d95:.12e},"
            f"{step_wall:.12e},{mech_wall:.12e},{damage_wall:.12e}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _metriques_globales_endommagement(comm, damage_array, u_array):
    local_u_inf = float(np.linalg.norm(u_array, ord=np.inf)) if u_array.size else 0.0
    local_max_d = float(np.max(damage_array)) if damage_array.size else 0.0
    local_sum_d = float(np.sum(damage_array))
    local_count = int(damage_array.size)
    local_count_d95 = int(np.count_nonzero(damage_array >= 0.95))

    u_max = float(comm.allreduce(local_u_inf, op=MPI.MAX))
    max_d = float(comm.allreduce(local_max_d, op=MPI.MAX))
    sum_d = float(comm.allreduce(local_sum_d, op=MPI.SUM))
    count = int(comm.allreduce(local_count, op=MPI.SUM))
    count_d95 = int(comm.allreduce(local_count_d95, op=MPI.SUM))

    mean_d = (sum_d / count) if count > 0 else 0.0
    frac_d95 = (count_d95 / count) if count > 0 else 0.0
    return u_max, max_d, mean_d, frac_d95


def _afficher_etape(n, n_last, tn, u_max, max_d, mean_d, frac_d95, step_wall_s, mech_wall_s, damage_wall_s):
    print(
        f"Step {n}/{n_last}, t={tn:.3e}, "
        f"max|u|={u_max:.3e}, max(d)={max_d:.3e}, "
        f"mean(d)={mean_d:.3e}, frac(d>=0.95)={frac_d95:.3e}, "
        f"temps_pas={step_wall_s:.2f}s (meca={mech_wall_s:.2f}s, "
        f"phase_field={damage_wall_s:.2f}s)"
    )


# ============================================================================
# OUTILS PHASE-FIELD
# ============================================================================

def _construire_solveur_vi_endommagement(Vd, damage, F_d, J_d, cfg):
    if not cfg.phase_field_utiliser_snes_vi:
        return None

    try:
        snes_probe = PETSc.SNES().create(MPI.COMM_SELF)
        snes_probe.setType("vinewtonrsls")
        if not hasattr(snes_probe, "setVariableBounds"):
            snes_probe.destroy()
            return None
        snes_probe.destroy()
    except Exception:
        return None

    try:
        problem_vi = dolfinx.fem.petsc.NonlinearProblem(F_d, damage, bcs=[], J=J_d)
        solver_vi = problem_vi.solver

        solver_vi.setType("vinewtonrsls")
        solver_vi.setTolerances(
            rtol=float(cfg.phase_field_snes_rtol),
            atol=float(cfg.phase_field_snes_atol),
            max_it=int(cfg.phase_field_snes_max_it),
        )

        ksp = solver_vi.getKSP()
        damage_opts = cfg.options_petsc_endommagement or cfg.options_petsc or {}
        if "ksp_type" in damage_opts:
            ksp.setType(damage_opts["ksp_type"])
        if "ksp_rtol" in damage_opts:
            ksp.setTolerances(rtol=float(damage_opts["ksp_rtol"]))
        if "pc_type" in damage_opts:
            ksp.getPC().setType(damage_opts["pc_type"])

        if hasattr(solver_vi, "setFromOptions"):
            solver_vi.setFromOptions()
        if hasattr(solver_vi, "setErrorIfNotConverged"):
            solver_vi.setErrorIfNotConverged(False)

        d_lb = fem.Function(Vd, name="DamageLowerBound")
        d_ub = fem.Function(Vd, name="DamageUpperBound")
        d_lb.x.array[:] = 0.0
        d_ub.x.array[:] = 1.0

        if not (hasattr(solver_vi, "setVariableBounds") and hasattr(d_lb.x, "petsc_vec")):
            return None

        solver_vi.setVariableBounds(d_lb.x.petsc_vec, d_ub.x.petsc_vec)
        return {"solver": solver_vi, "lb": d_lb, "ub": d_ub}
    except Exception:
        return None


def _update_history(psi_drive, psi_eval, history, pf_seuil):
    psi_drive.interpolate(psi_eval)
    psi_effective = np.maximum(psi_drive.x.array - pf_seuil, 0.0)
    history.x.array[:] = np.maximum(history.x.array, psi_effective)


# ============================================================================
# CINEMATIQUE CONTACT / TEMPS
# ============================================================================

def _calculer_cinematique_contact(domain, cfg):
    comm = domain.comm
    xmin = comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN)
    xmax = comm.allreduce(domain.geometry.x[:, 0].max(), op=MPI.MAX)
    ymin = comm.allreduce(domain.geometry.x[:, 1].min(), op=MPI.MIN)
    ymax = comm.allreduce(domain.geometry.x[:, 1].max(), op=MPI.MAX)
    zmin = comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN)
    zmax = comm.allreduce(domain.geometry.x[:, 2].max(), op=MPI.MAX)

    y_mid = float(np.clip(cfg.iceberg_centre_y, ymin, ymax))
    if cfg.iceberg_hauteur_au_dessus_fond_m is not None:
        z_target = zmin + float(cfg.iceberg_hauteur_au_dessus_fond_m)
    else:
        z_target = cfg.flottaison_z - cfg.iceberg_profondeur_sous_flottaison_m
    z_mid = float(np.clip(z_target, zmin, zmax))

    x_zone_debut = float(np.clip(cfg.iceberg_zone_x_debut_m, xmin, xmax))
    x_zone_fin = float(np.clip(cfg.iceberg_zone_x_fin_m, xmin, xmax))
    x_start, x_end = sorted((x_zone_debut, x_zone_fin))
    if cfg.iceberg_de_xmax_vers_xmin:
        x0, x1 = x_end, x_start
    else:
        x0, x1 = x_start, x_end

    t_contact_start = float(np.clip(cfg.iceberg_contact_t_debut, 0.0, cfg.temps_final))
    t_contact_end = float(np.clip(cfg.iceberg_contact_t_fin, t_contact_start, cfg.temps_final))
    t_contact_duration = max(t_contact_end - t_contact_start, 1e-12)

    return {
        "x0": x0,
        "x1": x1,
        "y_mid": y_mid,
        "z_mid": z_mid,
        "t_contact_start": t_contact_start,
        "t_contact_end": t_contact_end,
        "t_contact_duration": t_contact_duration,
    }


def _progression_contact(tn: float, kin) -> float:
    if tn <= kin["t_contact_start"]:
        return 0.0
    if tn >= kin["t_contact_end"]:
        return 1.0
    return (tn - kin["t_contact_start"]) / kin["t_contact_duration"]


def _rampe_contact(tn: float, kin, ramp_amplitude: bool) -> float:
    if not (kin["t_contact_start"] <= tn <= kin["t_contact_end"]):
        return 0.0
    if not ramp_amplitude:
        return 1.0
    return _progression_contact(tn, kin)


def _construire_pas_temps(cfg, x0: float, x1: float, t_contact_duration: float):
    if cfg.iceberg_dx_max_par_pas_m is not None and float(cfg.iceberg_dx_max_par_pas_m) > 0.0:
        longueur_parcours = abs(x1 - x0)
        contact_fraction = max(min(t_contact_duration / max(float(cfg.temps_final), 1e-12), 1.0), 1e-12)
        n_intervalles_min = (
            int(np.ceil(longueur_parcours / (float(cfg.iceberg_dx_max_par_pas_m) * contact_fraction)))
            if longueur_parcours > 0
            else 1
        )
        n_intervalles = max(int(cfg.nombre_pas), n_intervalles_min)
        return np.linspace(0.0, cfg.temps_final, n_intervalles + 1)

    if cfg.temps_relatifs:
        return cfg.temps_final * np.array(cfg.temps_relatifs, dtype=float)

    return np.linspace(0.0, cfg.temps_final, cfg.nombre_pas + 1)


# ============================================================================
# PHASE-FIELD BUILD
# ============================================================================

def _construire_systeme_phase_field(model, cfg, phase_field_preset):
    domain = model.domain
    Vd = model.Vd

    damage = model.damage_state
    damage.name = "Damage"
    damage_prev = fem.Function(Vd, name="DamagePrevStep")
    damage_iter_prev = fem.Function(Vd, name="DamagePrevIter")
    history = fem.Function(Vd, name="HistoryField")

    damage.x.array[:] = 0.0
    damage_prev.x.array[:] = 0.0
    damage_iter_prev.x.array[:] = 0.0
    history.x.array[:] = 0.0

    if not cfg.activer_phase_field_global:
        return {
            "enabled": False,
            "damage": damage,
            "damage_prev": damage_prev,
            "damage_iter_prev": damage_iter_prev,
            "history": history,
            "psi_drive": None,
            "psi_eval": None,
            "problem_d": None,
            "damage_vi": None,
        }

    gc_value = cfg.phase_field_gc_j_m2
    l0_value = cfg.phase_field_l0_m
    if cfg.phase_field_utiliser_preset_selectionne and phase_field_preset:
        gc_value = float(phase_field_preset.get("Gc_J_m2", gc_value))
        l0_value = float(phase_field_preset.get("l0_m", l0_value))

    gc = fem.Function(model.gc_factor_field.function_space, name="GcField")
    gc.x.array[:] = gc_value * model.gc_factor_field.x.array
    l0 = fem.Constant(domain, l0_value)
    residual_stiffness = fem.Constant(domain, cfg.phase_field_raideur_residuelle)

    u_sol, _ = ufl.split(model.v)
    P_plane = ufl.as_matrix(
        [
            [model.e1[0], model.e2[0]],
            [model.e1[1], model.e2[1]],
            [model.e1[2], model.e2[2]],
        ]
    )
    t_grad_u = ufl.dot(ufl.grad(u_sol), P_plane)
    eps = ufl.sym(ufl.dot(P_plane.T, t_grad_u))

    lmbda = model.E_field * model.nu_field / (1 + model.nu_field) / (1 - 2 * model.nu_field)
    mu = model.E_field / 2 / (1 + model.nu_field)
    lmbda_ps = 2 * lmbda * mu / (lmbda + 2 * mu)

    if cfg.phase_field_scinder_traction_compression:
        tr_eps = ufl.tr(eps)
        dev_eps = eps - (ufl.tr(eps) / 2.0) * ufl.Identity(2)
        tr_eps_pos = 0.5 * (tr_eps + ufl.sqrt(tr_eps * tr_eps))
        psi_drive_expr = 0.5 * lmbda_ps * tr_eps_pos**2 + mu * ufl.inner(dev_eps, dev_eps)
    else:
        psi_drive_expr = 0.5 * lmbda_ps * ufl.tr(eps) ** 2 + mu * ufl.inner(eps, eps)

    psi_drive = fem.Function(Vd, name="PsiDrive")
    psi_eval = fem.Expression(psi_drive_expr, Vd.element.interpolation_points)

    dd = ufl.TrialFunction(Vd)
    eta = ufl.TestFunction(Vd)
    a_d = (
        gc * l0 * ufl.dot(ufl.grad(dd), ufl.grad(eta))
        + (gc / l0 + 2.0 * history + residual_stiffness) * dd * eta
    ) * ufl.dx
    L_d = (2.0 * history) * eta * ufl.dx

    F_d = (
        gc * l0 * ufl.dot(ufl.grad(damage), ufl.grad(eta))
        + (gc / l0 + 2.0 * history + residual_stiffness) * damage * eta
        - (2.0 * history) * eta
    ) * ufl.dx
    J_d = ufl.derivative(F_d, damage, dd)

    problem_d = dolfinx.fem.petsc.LinearProblem(
        a_d,
        L_d,
        u=damage,
        bcs=[],
        petsc_options_prefix="coque_damage",
        petsc_options=cfg.options_petsc_endommagement or cfg.options_petsc,
    )
    damage_vi = _construire_solveur_vi_endommagement(Vd, damage, F_d, J_d, cfg)

    return {
        "enabled": True,
        "damage": damage,
        "damage_prev": damage_prev,
        "damage_iter_prev": damage_iter_prev,
        "history": history,
        "psi_drive": psi_drive,
        "psi_eval": psi_eval,
        "problem_d": problem_d,
        "damage_vi": damage_vi,
    }


# ============================================================================
# BOUCLE QUASI-STATIQUE
# ============================================================================

def executer_quasi_statique(model, cfg, output_layout, phase_field_preset=None):
    domain = model.domain
    comm = domain.comm

    kin = _calculer_cinematique_contact(domain, cfg)
    time_steps = _construire_pas_temps(cfg, kin["x0"], kin["x1"], kin["t_contact_duration"])

    # 1) Chargement iceberg: deplacement impose
    zero_vec = fem.Constant(domain, (0.0, 0.0, 0.0))
    L = ufl.dot(zero_vec, model.u_test) * ufl.dx

    radius_y = cfg.facteur_rayon_patch_iceberg * cfg.sigma
    radius_z = cfg.facteur_rayon_patch_iceberg * cfg.sigma

    def impact_region(x):
        y_scaled = (x[1] - kin["y_mid"]) / max(radius_y, 1e-12)
        z_scaled = (x[2] - kin["z_mid"]) / max(radius_z, 1e-12)
        inside_patch = (y_scaled * y_scaled + z_scaled * z_scaled) <= 1.0
        submerged = x[2] <= cfg.flottaison_z
        return inside_patch & submerged

    ice_dofs = fem.locate_dofs_geometrical((model.V.sub(0), model.Vu), impact_region)
    u_ice = fem.Function(model.Vu, name="IcebergDisplacement")
    extra_bcs = [fem.dirichletbc(u_ice, ice_dofs, model.V.sub(0))]

    sigma = fem.Constant(domain, float(cfg.sigma))
    x_center = fem.Constant(domain, float(kin["x0"]))
    disp_scale = fem.Constant(domain, 0.0)
    x = ufl.SpatialCoordinate(domain)
    r2 = (x[0] - x_center) ** 2 + (x[2] - kin["z_mid"]) ** 2
    disp_amp = disp_scale * ufl.exp(-r2 / (2 * sigma**2))
    u_ice_expr = disp_amp * model.e3
    u_ice_eval = fem.Expression(u_ice_expr, model.Vu.element.interpolation_points)

    # 2) Problemes EF
    problem_u = dolfinx.fem.petsc.LinearProblem(
        model.a,
        L,
        u=model.v,
        bcs=model.bcs + extra_bcs,
        petsc_options=cfg.options_petsc_mecanique or cfg.options_petsc,
        petsc_options_prefix="coque",
    )

    pf = _construire_systeme_phase_field(model, cfg, phase_field_preset)

    if MPI.COMM_WORLD.rank == 0 and pf["enabled"]:
        mode = "SNESVI bounds" if pf["damage_vi"] is not None else "projected bounds fallback"
        print(f"[phase-field] irreversibilite: {mode}")

    # 3) Parametres boucle
    vtk_stride = int(cfg.vtk_tous_les_n_pas)
    print_stride = int(cfg.console_tous_les_n_pas)

    ecrire_rotation_vtk = bool(cfg.ecrire_vtk_rotation)
    ecrire_endommagement_vtk = bool(cfg.ecrire_vtk_endommagement)
    if not pf["enabled"] and not cfg.ecrire_vtk_endommagement_si_desactive:
        ecrire_endommagement_vtk = False

    pf_stride = max(1, int(cfg.phase_field_mise_a_jour_tous_les_n_pas))
    pf_seuil = float(cfg.phase_field_seuil_nucleation_j_m3)
    pf_alt_max_iters = max(1, int(cfg.phase_field_nb_iters_alternance))
    pf_alt_min_iters = max(1, int(cfg.phase_field_nb_iters_min_alternance))
    pf_alt_tol = float(cfg.phase_field_tol_alternance)

    u_out = fem.Function(model.Vu, name="Displacement")
    theta_out = fem.Function(model.Vtheta, name="Rotation")

    monitor_rows = []
    n_last = len(time_steps) - 1

    rot_ctx = io.VTKFile(MPI.COMM_WORLD, output_layout["rotation_file"], "w") if ecrire_rotation_vtk else nullcontext(None)
    dmg_ctx = io.VTKFile(MPI.COMM_WORLD, output_layout["damage_file"], "w") if ecrire_endommagement_vtk else nullcontext(None)

    with io.VTKFile(MPI.COMM_WORLD, output_layout["displacement_file"], "w") as disp_vtk:
        with rot_ctx as rot_vtk:
            with dmg_ctx as damage_vtk:
                for n, tn in enumerate(time_steps):
                    step_t0 = perf_counter()
                    mech_wall_s = 0.0
                    damage_wall_s = 0.0

                    progress = _progression_contact(float(tn), kin)
                    x_center.value = kin["x0"] + (kin["x1"] - kin["x0"]) * progress
                    disp_scale.value = (
                        _rampe_contact(float(tn), kin, cfg.rampe_amplitude_iceberg)
                        * cfg.signe_deplacement_iceberg
                        * cfg.deplacement_pic_iceberg
                    )
                    u_ice.interpolate(u_ice_eval)

                    do_damage_update = pf["enabled"] and ((n % pf_stride == 0) or (n == n_last))

                    if not pf["enabled"]:
                        pf["damage"].x.array[:] = 0.0
                        mech_t0 = perf_counter()
                        problem_u.solve()
                        mech_wall_s += perf_counter() - mech_t0

                    elif not do_damage_update:
                        pf["damage"].x.array[:] = pf["damage_prev"].x.array
                        mech_t0 = perf_counter()
                        problem_u.solve()
                        mech_wall_s += perf_counter() - mech_t0

                        _update_history(pf["psi_drive"], pf["psi_eval"], pf["history"], pf_seuil)
                        pf["damage"].x.array[:] = pf["damage_prev"].x.array

                    else:
                        pf["damage"].x.array[:] = pf["damage_prev"].x.array
                        pf["damage_iter_prev"].x.array[:] = pf["damage_prev"].x.array

                        for k in range(pf_alt_max_iters):
                            mech_t0 = perf_counter()
                            problem_u.solve()
                            mech_wall_s += perf_counter() - mech_t0

                            _update_history(pf["psi_drive"], pf["psi_eval"], pf["history"], pf_seuil)

                            damage_t0 = perf_counter()
                            if pf["damage_vi"] is not None:
                                try:
                                    pf["damage_vi"]["lb"].x.array[:] = pf["damage_prev"].x.array
                                    pf["damage_vi"]["ub"].x.array[:] = 1.0
                                    pf["damage_vi"]["solver"].setVariableBounds(
                                        pf["damage_vi"]["lb"].x.petsc_vec,
                                        pf["damage_vi"]["ub"].x.petsc_vec,
                                    )
                                    pf["damage_vi"]["solver"].solve(None, pf["damage"].x.petsc_vec)
                                except Exception:
                                    if MPI.COMM_WORLD.rank == 0:
                                        print("[phase-field] SNESVI indisponible au solve -> fallback projection")
                                    pf["damage_vi"] = None
                                    pf["problem_d"].solve()
                            else:
                                pf["problem_d"].solve()
                            damage_wall_s += perf_counter() - damage_t0

                            pf["damage"].x.array[:] = np.clip(pf["damage"].x.array, pf["damage_prev"].x.array, 1.0)
                            alt_increment = np.linalg.norm(
                                pf["damage"].x.array - pf["damage_iter_prev"].x.array,
                                ord=np.inf,
                            )
                            pf["damage_iter_prev"].x.array[:] = pf["damage"].x.array
                            if k + 1 >= pf_alt_min_iters and alt_increment < pf_alt_tol:
                                break

                        pf["damage_prev"].x.array[:] = pf["damage"].x.array

                    u_out.interpolate(model.v.sub(0))
                    theta_out.interpolate(model.v.sub(1))

                    if (n % vtk_stride == 0) or (n == n_last):
                        disp_vtk.write_function(u_out, tn)
                        if rot_vtk is not None:
                            rot_vtk.write_function(theta_out, tn)
                        if damage_vtk is not None:
                            damage_vtk.write_function(pf["damage"], tn)

                    u_max, max_d, mean_d, frac_d95 = _metriques_globales_endommagement(
                        comm,
                        pf["damage"].x.array,
                        u_out.x.array,
                    )
                    step_wall_s = perf_counter() - step_t0

                    monitor_rows.append(
                        (
                            n,
                            tn,
                            u_max,
                            max_d,
                            mean_d,
                            frac_d95,
                            step_wall_s,
                            mech_wall_s,
                            damage_wall_s,
                        )
                    )

                    if MPI.COMM_WORLD.rank == 0 and ((n % print_stride == 0) or (n == n_last)):
                        _afficher_etape(
                            n,
                            n_last,
                            tn,
                            u_max,
                            max_d,
                            mean_d,
                            frac_d95,
                            step_wall_s,
                            mech_wall_s,
                            damage_wall_s,
                        )

    if MPI.COMM_WORLD.rank == 0:
        _ecrire_csv_suivi(output_layout["monitor_file"], monitor_rows)


# Alias de compatibilite
run_quasi_static = executer_quasi_statique
