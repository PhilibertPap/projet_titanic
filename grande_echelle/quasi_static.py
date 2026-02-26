from contextlib import nullcontext
from time import perf_counter

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc


def _write_monitor_csv(path, rows):
    lines = [
        "step,time,max_u_inf,max_damage,mean_damage,frac_damage_ge_095,"
        "temps_pas_s,temps_meca_s,temps_phase_field_s"
    ]
    for step, time_value, max_u, max_d, mean_d, frac_d95, step_wall, mech_wall, damage_wall in rows:
        lines.append(
            f"{step},{time_value:.12g},{max_u:.12e},{max_d:.12e},{mean_d:.12e},{frac_d95:.12e},"
            f"{step_wall:.12e},{mech_wall:.12e},{damage_wall:.12e}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_build_damage_vi_solver(Vd, damage, F_d, J_d, cfg):
    if not bool(getattr(cfg, "phase_field_use_snes_vi", True)):
        return None
    # Preflight PETSc SNESVI support before constructing a DOLFINx NonlinearProblem.
    # On some versions, constructing NonlinearProblem may fail partway and emit a
    # noisy __del__ warning if SNES VI is unavailable.
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
        solver_vi = getattr(problem_vi, "solver", None)
        if solver_vi is None:
            return None

        solver_vi.setType("vinewtonrsls")
        solver_vi.setTolerances(
            rtol=float(getattr(cfg, "phase_field_snes_rtol", 1.0e-9)),
            atol=float(getattr(cfg, "phase_field_snes_atol", 1.0e-9)),
            max_it=int(getattr(cfg, "phase_field_snes_max_it", 50)),
        )

        ksp = solver_vi.getKSP()
        damage_opts = cfg.damage_petsc_options or cfg.petsc_options or {}
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


def run_quasi_static(model, cfg, output_layout, phase_field_preset=None):
    domain = model.domain
    comm = domain.comm
    t = fem.Constant(domain, 0.0)

    # ============================================================
    # 1) Geometrie coque et cinematique du contact iceberg
    # ============================================================
    xmin = comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN)
    xmax = comm.allreduce(domain.geometry.x[:, 0].max(), op=MPI.MAX)
    ymin = comm.allreduce(domain.geometry.x[:, 1].min(), op=MPI.MIN)
    ymax = comm.allreduce(domain.geometry.x[:, 1].max(), op=MPI.MAX)
    zmin = comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN)
    zmax = comm.allreduce(domain.geometry.x[:, 2].max(), op=MPI.MAX)

    y_mid = float(np.clip(cfg.iceberg_center_y, ymin, ymax))
    z_target = cfg.waterline_z - cfg.iceberg_depth_below_waterline
    z_mid = float(np.clip(z_target, zmin, zmax))

    x_zone_debut = float(np.clip(getattr(cfg, "iceberg_zone_x_debut_m", xmin), xmin, xmax))
    x_zone_fin = float(np.clip(getattr(cfg, "iceberg_zone_x_fin_m", xmax), xmin, xmax))
    x_start, x_end = sorted((x_zone_debut, x_zone_fin))
    if getattr(cfg, "iceberg_moves_from_xmax_to_xmin", False):
        x0, x1 = x_end, x_start
    else:
        x0, x1 = x_start, x_end

    t_contact_start = float(np.clip(getattr(cfg, "iceberg_contact_t_start", 0.0), 0.0, cfg.t_final))
    t_contact_end = float(np.clip(getattr(cfg, "iceberg_contact_t_end", cfg.t_final), t_contact_start, cfg.t_final))
    t_contact_duration = max(t_contact_end - t_contact_start, 1e-12)
    ramp_amplitude = bool(getattr(cfg, "ramp_amplitude_iceberg", False))

    def contact_progress(tn: float) -> float:
        if tn <= t_contact_start:
            return 0.0
        if tn >= t_contact_end:
            return 1.0
        return (tn - t_contact_start) / t_contact_duration

    def contact_ramp(tn: float) -> float:
        if not (t_contact_start <= tn <= t_contact_end):
            return 0.0
        if not ramp_amplitude:
            return 1.0
        return contact_progress(tn)

    # ============================================================
    # 2) Grille de temps (uniforme, relative, ou limite dx iceberg)
    # ============================================================
    temps_relatifs = getattr(cfg, "temps_relatifs", None)
    dx_max_par_pas = getattr(cfg, "iceberg_max_dx_par_pas_m", None)
    if dx_max_par_pas is not None and float(dx_max_par_pas) > 0.0:
        longueur_parcours = abs(x1 - x0)
        contact_fraction = max(min(t_contact_duration / max(float(cfg.t_final), 1e-12), 1.0), 1e-12)
        # The iceberg moves only during the contact window, so the total number
        # of intervals must be increased accordingly to respect dx_max per step.
        n_intervalles_min = (
            int(np.ceil(longueur_parcours / (float(dx_max_par_pas) * contact_fraction)))
            if longueur_parcours > 0
            else 1
        )
        n_intervalles = max(int(cfg.num_steps), n_intervalles_min)
        time_steps = np.linspace(0.0, cfg.t_final, n_intervalles + 1)
    elif temps_relatifs:
        time_steps = cfg.t_final * np.array(temps_relatifs, dtype=float)
    else:
        time_steps = np.linspace(0.0, cfg.t_final, cfg.num_steps + 1)

    # ============================================================
    # 3) Chargement iceberg Dirichlet (deplacement impose sur e3)
    # ============================================================
    zero_vec = fem.Constant(domain, (0.0, 0.0, 0.0))
    L = ufl.dot(zero_vec, model.u_test) * ufl.dx

    radius_y = cfg.iceberg_patch_radius_factor * cfg.sigma
    radius_z = cfg.iceberg_patch_radius_factor * cfg.sigma
    waterline_z = cfg.waterline_z

    def impact_region(x):
        y_scaled = (x[1] - y_mid) / max(radius_y, 1e-12)
        z_scaled = (x[2] - z_mid) / max(radius_z, 1e-12)
        inside_patch = (y_scaled * y_scaled + z_scaled * z_scaled) <= 1.0
        submerged = x[2] <= waterline_z
        return inside_patch & submerged

    ice_dofs = fem.locate_dofs_geometrical((model.V.sub(0), model.Vu), impact_region)
    u_ice = fem.Function(model.Vu, name="IcebergDisplacement")
    extra_bcs = [fem.dirichletbc(u_ice, ice_dofs, model.V.sub(0))]

    sigma = fem.Constant(domain, float(cfg.sigma))
    x_center = fem.Constant(domain, float(x0))
    disp_scale = fem.Constant(domain, 0.0)
    x = ufl.SpatialCoordinate(domain)
    r2 = (x[0] - x_center) ** 2 + (x[2] - z_mid) ** 2
    disp_amp = disp_scale * ufl.exp(-r2 / (2 * sigma**2))
    u_ice_expr = disp_amp * model.e3
    u_ice_eval = fem.Expression(u_ice_expr, model.Vu.element.interpolation_points)

    # ============================================================
    # 4) Probleme mecanique coque
    # ============================================================
    mechanics_petsc_options = cfg.mechanics_petsc_options or cfg.petsc_options
    problem_u = dolfinx.fem.petsc.LinearProblem(
        model.a,
        L,
        u=model.v,
        bcs=model.bcs + extra_bcs,
        petsc_options=mechanics_petsc_options,
        petsc_options_prefix="coque",
    )

    # ============================================================
    # 5) Sorties et phase-field global (AT1 lineaire + irreversibilite)
    # ============================================================
    u_out = fem.Function(model.Vu, name="Displacement")
    theta_out = fem.Function(model.Vtheta, name="Rotation")

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

    damage_enabled = bool(cfg.enable_global_phase_field)
    psi_drive = None
    psi_eval = None
    problem_d = None
    damage_vi = None

    if damage_enabled:
        gc_value = cfg.phase_field_gc_j_m2
        l0_value = cfg.phase_field_l0_m
        if cfg.phase_field_use_selected_preset and phase_field_preset:
            gc_value = float(phase_field_preset.get("Gc_J_m2", gc_value))
            l0_value = float(phase_field_preset.get("l0_m", l0_value))

        gc = fem.Function(model.gc_factor_field.function_space, name="GcField")
        gc.x.array[:] = gc_value * model.gc_factor_field.x.array
        l0 = fem.Constant(domain, l0_value)
        residual_stiffness = fem.Constant(domain, cfg.phase_field_residual_stiffness)

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

        if getattr(cfg, "phase_field_split_traction_compression", False):
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
            petsc_options=cfg.damage_petsc_options or cfg.petsc_options,
        )
        damage_vi = _try_build_damage_vi_solver(Vd, damage, F_d, J_d, cfg)

    if MPI.COMM_WORLD.rank == 0 and damage_enabled:
        mode = "SNESVI bounds" if damage_vi is not None else "projected bounds fallback"
        print(f"[phase-field] irreversibilite: {mode}")

    # ============================================================
    # 6) Parametres de sorties et de boucle
    # ============================================================
    vtk_stride = int(cfg.ecrire_vtk_tous_les_n_pas)
    print_stride = int(cfg.afficher_console_tous_les_n_pas)
    write_rotation_vtk = bool(getattr(cfg, "write_rotation_vtk", True))
    write_damage_vtk = bool(getattr(cfg, "write_damage_vtk", True))
    if not damage_enabled and not bool(getattr(cfg, "write_damage_vtk_if_disabled", False)):
        write_damage_vtk = False

    pf_stride = max(1, int(getattr(cfg, "phase_field_mise_a_jour_tous_les_n_pas", 1)))
    pf_seuil = float(getattr(cfg, "phase_field_seuil_nucleation_j_m3", 0.0))
    pf_alt_max_iters = max(1, int(getattr(cfg, "phase_field_n_alt_iters", 6)))
    pf_alt_min_iters = max(1, int(getattr(cfg, "phase_field_alt_min_iters", 1)))
    pf_alt_tol = float(getattr(cfg, "phase_field_alt_tol", 1e-4))

    monitor_rows = []
    n_last = len(time_steps) - 1

    rot_ctx = io.VTKFile(MPI.COMM_WORLD, output_layout["rotation_file"], "w") if write_rotation_vtk else nullcontext(None)
    dmg_ctx = io.VTKFile(MPI.COMM_WORLD, output_layout["damage_file"], "w") if write_damage_vtk else nullcontext(None)

    # ============================================================
    # 7) Boucle quasi-statique (style notebook / script)
    # ============================================================
    with io.VTKFile(MPI.COMM_WORLD, output_layout["displacement_file"], "w") as disp_vtk:
        with rot_ctx as rot_vtk:
            with dmg_ctx as damage_vtk:
                for n, tn in enumerate(time_steps):
                    step_t0 = perf_counter()
                    mech_wall_s = 0.0
                    damage_wall_s = 0.0

                    t.value = tn
                    progress = contact_progress(float(tn))
                    x_center.value = x0 + (x1 - x0) * progress
                    disp_scale.value = contact_ramp(float(tn)) * cfg.iceberg_disp_sign * cfg.iceberg_disp_peak
                    u_ice.interpolate(u_ice_eval)

                    do_damage_update = damage_enabled and ((n % pf_stride == 0) or (n == n_last))

                    # --- solveur mecanique seul (si pas de phase-field) ---
                    if not damage_enabled:
                        damage.x.array[:] = 0.0
                        mech_t0 = perf_counter()
                        problem_u.solve()
                        mech_wall_s += perf_counter() - mech_t0

                    # --- solveur meca + history, sans update dommage (stride PF) ---
                    elif not do_damage_update:
                        damage.x.array[:] = damage_prev.x.array
                        mech_t0 = perf_counter()
                        problem_u.solve()
                        mech_wall_s += perf_counter() - mech_t0

                        psi_drive.interpolate(psi_eval)
                        psi_effective = np.maximum(psi_drive.x.array - pf_seuil, 0.0)
                        history.x.array[:] = np.maximum(history.x.array, psi_effective)
                        damage.x.array[:] = damage_prev.x.array

                    # --- alternance meca / phase-field avec irreversibilite ---
                    else:
                        damage.x.array[:] = damage_prev.x.array
                        damage_iter_prev.x.array[:] = damage_prev.x.array

                        for k in range(pf_alt_max_iters):
                            mech_t0 = perf_counter()
                            problem_u.solve()
                            mech_wall_s += perf_counter() - mech_t0

                            psi_drive.interpolate(psi_eval)
                            psi_effective = np.maximum(psi_drive.x.array - pf_seuil, 0.0)
                            history.x.array[:] = np.maximum(history.x.array, psi_effective)

                            damage_t0 = perf_counter()
                            if damage_vi is not None:
                                try:
                                    damage_vi["lb"].x.array[:] = damage_prev.x.array
                                    damage_vi["ub"].x.array[:] = 1.0
                                    damage_vi["solver"].setVariableBounds(
                                        damage_vi["lb"].x.petsc_vec,
                                        damage_vi["ub"].x.petsc_vec,
                                    )
                                    damage_vi["solver"].solve(None, damage.x.petsc_vec)
                                except Exception:
                                    if MPI.COMM_WORLD.rank == 0:
                                        print("[phase-field] SNESVI indisponible au solve -> fallback projection")
                                    damage_vi = None
                                    problem_d.solve()
                            else:
                                problem_d.solve()
                            damage_wall_s += perf_counter() - damage_t0

                            damage.x.array[:] = np.clip(damage.x.array, damage_prev.x.array, 1.0)
                            alt_increment = np.linalg.norm(damage.x.array - damage_iter_prev.x.array, ord=np.inf)
                            damage_iter_prev.x.array[:] = damage.x.array
                            if k + 1 >= pf_alt_min_iters and alt_increment < pf_alt_tol:
                                break

                        damage_prev.x.array[:] = damage.x.array

                    # --- sorties ---
                    u_out.interpolate(model.v.sub(0))
                    theta_out.interpolate(model.v.sub(1))

                    if (n % vtk_stride == 0) or (n == n_last):
                        disp_vtk.write_function(u_out, tn)
                        if rot_vtk is not None:
                            rot_vtk.write_function(theta_out, tn)
                        if damage_vtk is not None:
                            damage_vtk.write_function(damage, tn)

                    u_max = np.linalg.norm(u_out.x.array, ord=np.inf)
                    max_d = float(np.max(damage.x.array))
                    mean_d = float(np.mean(damage.x.array))
                    frac_d95 = float(np.mean(damage.x.array >= 0.95))
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
                        print(
                            f"Step {n}/{n_last}, t={tn:.3e}, "
                            f"max|u|={u_max:.3e}, max(d)={max_d:.3e}, "
                            f"mean(d)={mean_d:.3e}, frac(d>=0.95)={frac_d95:.3e}, "
                            f"temps_pas={step_wall_s:.2f}s (meca={mech_wall_s:.2f}s, "
                            f"phase_field={damage_wall_s:.2f}s)"
                        )

    if MPI.COMM_WORLD.rank == 0:
        _write_monitor_csv(output_layout["monitor_file"], monitor_rows)
