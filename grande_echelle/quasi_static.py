import numpy as np
from time import perf_counter
from contextlib import nullcontext
from mpi4py import MPI
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc


def moving_gaussian_pressure(domain, c0, v_ice, t, sigma, p0):
    x = ufl.SpatialCoordinate(domain)
    c = c0 + t * v_ice
    r2 = ufl.dot(x - c, x - c)
    return p0 * ufl.exp(-r2 / (2 * sigma**2))


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


def run_quasi_static(model, cfg, output_layout, phase_field_preset=None):
    domain = model.domain
    t = fem.Constant(domain, 0.0)
    zero_vec = fem.Constant(domain, (0.0, 0.0, 0.0))

    # ------------------------------------------------------------
    # 1) Parametres geometriques utiles pour definir la zone d'impact
    # ------------------------------------------------------------

    comm = domain.comm
    xmin = comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN)
    xmax = comm.allreduce(domain.geometry.x[:, 0].max(), op=MPI.MAX)
    ymin = comm.allreduce(domain.geometry.x[:, 1].min(), op=MPI.MIN)
    ymax = comm.allreduce(domain.geometry.x[:, 1].max(), op=MPI.MAX)
    zmin = comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN)
    zmax = comm.allreduce(domain.geometry.x[:, 2].max(), op=MPI.MAX)

    iceberg_center_y = getattr(cfg, "iceberg_center_y", None)
    if iceberg_center_y is not None:
        y_mid = float(np.clip(iceberg_center_y, ymin, ymax))
    else:
        y_mid = cfg.y_mid_factor * ymax
    if hasattr(cfg, "waterline_z") and hasattr(cfg, "iceberg_depth_below_waterline"):
        z_target = cfg.waterline_z - cfg.iceberg_depth_below_waterline
        z_mid = float(np.clip(z_target, zmin, zmax))
    else:
        z_mid = zmin + cfg.z_mid_factor * (zmax - zmin)

    x_zone_debut = float(getattr(cfg, "iceberg_zone_x_debut_m", xmin))
    x_zone_fin = float(getattr(cfg, "iceberg_zone_x_fin_m", xmax))
    x_zone_debut = float(np.clip(x_zone_debut, xmin, xmax))
    x_zone_fin = float(np.clip(x_zone_fin, xmin, xmax))
    if x_zone_fin < x_zone_debut:
        x_zone_debut, x_zone_fin = x_zone_fin, x_zone_debut

    if getattr(cfg, "iceberg_moves_from_xmax_to_xmin", False):
        # Mouvement "vers l'avant" du patch numerique dans la zone definie
        x0 = x_zone_fin
        x1 = x_zone_debut
    else:
        x0 = x_zone_debut
        x1 = x_zone_fin
    vx = (x1 - x0) / cfg.t_final
    contact_t_start = float(getattr(cfg, "iceberg_contact_t_start", 0.0))
    contact_t_end = float(getattr(cfg, "iceberg_contact_t_end", cfg.t_final))
    contact_t_start = float(np.clip(contact_t_start, 0.0, cfg.t_final))
    contact_t_end = float(np.clip(contact_t_end, contact_t_start, cfg.t_final))
    contact_duration = max(contact_t_end - contact_t_start, 1e-12)

    # The iceberg patch moves only during the contact window [t_start, t_end].
    # This avoids loading the shell during non-contact stages.
    c0 = fem.Constant(domain, (x0, y_mid, z_mid))
    v_ice = fem.Constant(domain, ((x1 - x0) / contact_duration, 0.0, 0.0))
    sigma = fem.Constant(domain, cfg.sigma)
    p0 = fem.Constant(domain, 0.0 if getattr(cfg, "ramp_amplitude_iceberg", False) else cfg.pressure_peak)

    # ------------------------------------------------------------
    # 2) Chargement externe (pression ou deplacement impose)
    # ------------------------------------------------------------
    # Par defaut: pas de force volumique/surfacique (cas pilotage Dirichlet).
    L = ufl.dot(zero_vec, model.u_test) * ufl.dx
    extra_bcs = []
    if cfg.iceberg_loading == "neumann_pressure":
        p = moving_gaussian_pressure(domain, c0, v_ice, t, sigma, p0)
        # Positive sign chosen so the pressure pushes the shell inward for the
        # current hull orientation (the previous sign was pulling it outward).
        f_ice = p * model.e3
        L = ufl.dot(f_ice, model.u_test) * ufl.dx
    elif cfg.iceberg_loading == "dirichlet_displacement":
        radius_y = cfg.iceberg_patch_radius_factor * cfg.sigma
        radius_z = cfg.iceberg_patch_radius_factor * cfg.sigma
        waterline_z = getattr(cfg, "waterline_z", np.inf)

        def impact_region(x):
            y_scaled = (x[1] - y_mid) / max(radius_y, 1e-12)
            z_scaled = (x[2] - z_mid) / max(radius_z, 1e-12)
            inside_patch = (y_scaled * y_scaled + z_scaled * z_scaled) <= 1.0
            submerged = x[2] <= waterline_z
            return inside_patch & submerged

        ice_dofs = fem.locate_dofs_geometrical((model.V.sub(0), model.Vu), impact_region)
        u_ice = fem.Function(model.Vu, name="IcebergDisplacement")
        extra_bcs.append(fem.dirichletbc(u_ice, ice_dofs, model.V.sub(0)))
    else:
        raise ValueError(
            f"Unknown cfg.iceberg_loading='{cfg.iceberg_loading}'. "
            "Expected 'neumann_pressure' or 'dirichlet_displacement'."
        )

    def _contact_progress(tn: float) -> float:
        if tn <= contact_t_start:
            return 0.0
        if tn >= contact_t_end:
            return 1.0
        return (tn - contact_t_start) / contact_duration

    def _contact_active(tn: float) -> bool:
        return contact_t_start <= tn <= contact_t_end

    # Probleme lineaire mecanique (deplacement/rotation de coque)
    mechanics_petsc_options = cfg.mechanics_petsc_options or cfg.petsc_options
    problem_u = dolfinx.fem.petsc.LinearProblem(
        model.a,
        L,
        u=model.v,
        bcs=model.bcs + extra_bcs,
        petsc_options=mechanics_petsc_options,
        petsc_options_prefix="coque",
    )

    # ------------------------------------------------------------
    # 3) Variables de sortie / dommage global (phase-field)
    # ------------------------------------------------------------
    u_out = fem.Function(model.Vu, name="Displacement")
    theta_out = fem.Function(model.Vtheta, name="Rotation")
    Vd = model.Vd
    damage = model.damage_state
    damage.name = "Damage"
    damage_old = fem.Function(Vd, name="DamageOld")
    history = fem.Function(Vd, name="HistoryField")
    damage.x.array[:] = 0.0
    damage_old.x.array[:] = 0.0
    history.x.array[:] = 0.0

    damage_enabled = bool(cfg.enable_global_phase_field)
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
            psi_drive_expr = 0.5 * lmbda_ps * tr_eps_pos ** 2 + mu * ufl.inner(dev_eps, dev_eps)
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
        # Probleme lineaire phase-field global (dommage)
        problem_d = dolfinx.fem.petsc.LinearProblem(
            a_d,
            L_d,
            u=damage,
            bcs=[],
            petsc_options_prefix="coque_damage",
            petsc_options=cfg.damage_petsc_options or cfg.petsc_options,
        )
    else:
        psi_drive = None
        psi_eval = None
        problem_d = None

    # ------------------------------------------------------------
    # 4) Discretisation en temps + sorties
    # ------------------------------------------------------------
    temps_relatifs = getattr(cfg, "temps_relatifs", None)
    dx_max_par_pas = getattr(cfg, "iceberg_max_dx_par_pas_m", None)
    if dx_max_par_pas is not None and float(dx_max_par_pas) > 0.0:
        longueur_parcours = abs(x1 - x0)
        n_intervalles_min = int(np.ceil(longueur_parcours / float(dx_max_par_pas))) if longueur_parcours > 0 else 1
        n_intervalles = max(int(cfg.num_steps), n_intervalles_min)
        time_steps = np.linspace(0.0, cfg.t_final, n_intervalles + 1)
    elif temps_relatifs:
        time_steps = cfg.t_final * np.array(temps_relatifs, dtype=float)
    else:
        time_steps = np.linspace(0.0, cfg.t_final, cfg.num_steps + 1)

    ecrire_vtk_tous_les_n_pas = int(
        getattr(
            cfg,
            "ecrire_vtk_tous_les_n_pas",
            getattr(cfg, "vtk_write_stride", 1),
        )
    )
    write_rotation_vtk = bool(getattr(cfg, "write_rotation_vtk", True))
    write_damage_vtk = bool(getattr(cfg, "write_damage_vtk", True))
    if not damage_enabled and not bool(getattr(cfg, "write_damage_vtk_if_disabled", False)):
        write_damage_vtk = False

    ramp_amplitude_iceberg = bool(getattr(cfg, "ramp_amplitude_iceberg", False))
    maj_damage_stride = max(
        1,
        int(getattr(cfg, "phase_field_mise_a_jour_tous_les_n_pas", 1)),
    )
    phase_field_seuil = float(getattr(cfg, "phase_field_seuil_nucleation_j_m3", 0.0))

    afficher_console_tous_les_n_pas = int(
        getattr(
            cfg,
            "afficher_console_tous_les_n_pas",
            getattr(cfg, "monitor_print_stride", 1),
        )
    )
    monitor_rows = []

    rot_ctx = (
        io.VTKFile(MPI.COMM_WORLD, output_layout["rotation_file"], "w")
        if write_rotation_vtk
        else nullcontext(None)
    )
    dmg_ctx = (
        io.VTKFile(MPI.COMM_WORLD, output_layout["damage_file"], "w")
        if write_damage_vtk
        else nullcontext(None)
    )
    with io.VTKFile(MPI.COMM_WORLD, output_layout["displacement_file"], "w") as disp_vtk:
        with rot_ctx as rot_vtk:
            with dmg_ctx as damage_vtk:
                # ------------------------------------------------------------
                # 5) Boucle en temps quasi-statique
                # ------------------------------------------------------------
                for n, tn in enumerate(time_steps):
                    step_t0 = perf_counter()
                    mech_wall_s = 0.0
                    damage_wall_s = 0.0
                    t.value = tn
                    in_contact = _contact_active(float(tn))
                    contact_progress = _contact_progress(float(tn))
                    facteur_rampe = 1.0 if in_contact else 0.0
                    if ramp_amplitude_iceberg:
                        facteur_rampe = contact_progress if in_contact else 0.0
                    if cfg.iceberg_loading == "neumann_pressure":
                        p0.value = cfg.pressure_peak * facteur_rampe
                    if cfg.iceberg_loading == "dirichlet_displacement":
                        x_center = x0 + (x1 - x0) * contact_progress

                        def prescribed_displacement(x):
                            r2 = (x[0] - x_center) ** 2 + (x[2] - z_mid) ** 2
                            amplitude = (
                                facteur_rampe
                                * cfg.iceberg_disp_sign
                                * cfg.iceberg_disp_peak
                                * np.exp(
                                -r2 / (2 * cfg.sigma**2)
                                )
                            )
                            return np.vstack(
                                (np.zeros_like(amplitude), amplitude, np.zeros_like(amplitude))
                            )

                        u_ice.interpolate(prescribed_displacement)
                    # Mechanical solve uses damage from the previous converged load step
                    # through `model.damage_state` embedded in the shell bilinear form.
                    mech_t0 = perf_counter()
                    problem_u.solve()
                    mech_wall_s = perf_counter() - mech_t0
                    u_out.interpolate(model.v.sub(0))
                    theta_out.interpolate(model.v.sub(1))

                    if damage_enabled:
                        damage_t0 = perf_counter()
                        psi_drive.interpolate(psi_eval)
                        psi_effective = np.maximum(psi_drive.x.array - phase_field_seuil, 0.0)
                        history.x.array[:] = np.maximum(history.x.array, psi_effective)

                        if (n % maj_damage_stride == 0) or (n == len(time_steps) - 1):
                            problem_d.solve()
                            d_clipped = np.clip(damage.x.array, 0.0, 1.0)
                            d_irrev = np.maximum(damage_old.x.array, d_clipped)
                            damage.x.array[:] = d_irrev
                            damage_old.x.array[:] = d_irrev
                        else:
                            damage.x.array[:] = damage_old.x.array
                        damage_wall_s = perf_counter() - damage_t0
                    else:
                        damage.x.array[:] = 0.0

                    if (n % ecrire_vtk_tous_les_n_pas == 0) or (n == len(time_steps) - 1):
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
                    if MPI.COMM_WORLD.rank == 0 and (
                        n % afficher_console_tous_les_n_pas == 0
                        or n == len(time_steps) - 1
                    ):
                        print(
                            f"Step {n}/{len(time_steps)-1}, t={tn:.3e}, "
                            f"max|u|={u_max:.3e}, max(d)={max_d:.3e}, "
                            f"mean(d)={mean_d:.3e}, frac(d>=0.95)={frac_d95:.3e}, "
                            f"temps_pas={step_wall_s:.2f}s (meca={mech_wall_s:.2f}s, "
                            f"phase_field={damage_wall_s:.2f}s)"
                        )

    if MPI.COMM_WORLD.rank == 0:
        _write_monitor_csv(output_layout["monitor_file"], monitor_rows)
