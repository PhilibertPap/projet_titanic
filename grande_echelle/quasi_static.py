from dataclasses import dataclass
from contextlib import nullcontext
from time import perf_counter

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc

try:
    from .phase_field_solver import (
        PhaseFieldContext,
        StepStats,
        build_phase_field_context,
        phase_field_irreversibility_mode,
        advance_quasi_static_step,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from phase_field_solver import (
        PhaseFieldContext,
        StepStats,
        build_phase_field_context,
        phase_field_irreversibility_mode,
        advance_quasi_static_step,
    )


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


@dataclass
class ContactKinematics:
    x0: float
    x1: float
    y_mid: float
    z_mid: float
    t_start: float
    t_end: float
    duration: float
    ramp_amplitude: bool

    def progress(self, tn: float) -> float:
        if tn <= self.t_start:
            return 0.0
        if tn >= self.t_end:
            return 1.0
        return (tn - self.t_start) / self.duration

    def active(self, tn: float) -> bool:
        return self.t_start <= tn <= self.t_end

    def ramp(self, tn: float) -> float:
        active = self.active(tn)
        if not active:
            return 0.0
        return self.progress(tn) if self.ramp_amplitude else 1.0


@dataclass
class LoadController:
    L: any
    extra_bcs: list
    update: any


@dataclass
class OutputControls:
    vtk_stride: int
    write_rotation: bool
    write_damage: bool
    print_stride: int


def _domain_bounds(domain):
    comm = domain.comm
    return {
        "xmin": comm.allreduce(domain.geometry.x[:, 0].min(), op=MPI.MIN),
        "xmax": comm.allreduce(domain.geometry.x[:, 0].max(), op=MPI.MAX),
        "ymin": comm.allreduce(domain.geometry.x[:, 1].min(), op=MPI.MIN),
        "ymax": comm.allreduce(domain.geometry.x[:, 1].max(), op=MPI.MAX),
        "zmin": comm.allreduce(domain.geometry.x[:, 2].min(), op=MPI.MIN),
        "zmax": comm.allreduce(domain.geometry.x[:, 2].max(), op=MPI.MAX),
    }


def _build_contact_kinematics(domain, cfg) -> ContactKinematics:
    bounds = _domain_bounds(domain)
    xmin = bounds["xmin"]
    xmax = bounds["xmax"]
    ymin = bounds["ymin"]
    ymax = bounds["ymax"]
    zmin = bounds["zmin"]
    zmax = bounds["zmax"]

    iceberg_center_y = getattr(cfg, "iceberg_center_y", None)
    y_mid = float(np.clip(iceberg_center_y, ymin, ymax)) if iceberg_center_y is not None else cfg.y_mid_factor * ymax

    if hasattr(cfg, "waterline_z") and hasattr(cfg, "iceberg_depth_below_waterline"):
        z_target = cfg.waterline_z - cfg.iceberg_depth_below_waterline
        z_mid = float(np.clip(z_target, zmin, zmax))
    else:
        z_mid = zmin + cfg.z_mid_factor * (zmax - zmin)

    x_zone_debut = float(np.clip(getattr(cfg, "iceberg_zone_x_debut_m", xmin), xmin, xmax))
    x_zone_fin = float(np.clip(getattr(cfg, "iceberg_zone_x_fin_m", xmax), xmin, xmax))
    x_start, x_end = sorted((x_zone_debut, x_zone_fin))
    x0, x1 = (x_end, x_start) if getattr(cfg, "iceberg_moves_from_xmax_to_xmin", False) else (x_start, x_end)

    t_start = float(np.clip(getattr(cfg, "iceberg_contact_t_start", 0.0), 0.0, cfg.t_final))
    t_end = float(np.clip(getattr(cfg, "iceberg_contact_t_end", cfg.t_final), t_start, cfg.t_final))
    duration = max(t_end - t_start, 1e-12)

    return ContactKinematics(
        x0=x0,
        x1=x1,
        y_mid=y_mid,
        z_mid=z_mid,
        t_start=t_start,
        t_end=t_end,
        duration=duration,
        ramp_amplitude=bool(getattr(cfg, "ramp_amplitude_iceberg", False)),
    )


def _build_time_steps(cfg, contact: ContactKinematics):
    temps_relatifs = getattr(cfg, "temps_relatifs", None)
    dx_max_par_pas = getattr(cfg, "iceberg_max_dx_par_pas_m", None)
    if dx_max_par_pas is not None and float(dx_max_par_pas) > 0.0:
        longueur_parcours = abs(contact.x1 - contact.x0)
        n_intervalles_min = int(np.ceil(longueur_parcours / float(dx_max_par_pas))) if longueur_parcours > 0 else 1
        n_intervalles = max(int(cfg.num_steps), n_intervalles_min)
        return np.linspace(0.0, cfg.t_final, n_intervalles + 1)
    if temps_relatifs:
        return cfg.t_final * np.array(temps_relatifs, dtype=float)
    return np.linspace(0.0, cfg.t_final, cfg.num_steps + 1)


def _make_neumann_pressure_loading(model, cfg, t, contact: ContactKinematics) -> LoadController:
    domain = model.domain
    zero_vec = fem.Constant(domain, (0.0, 0.0, 0.0))
    sigma = fem.Constant(domain, float(cfg.sigma))
    p0 = fem.Constant(domain, 0.0 if contact.ramp_amplitude else cfg.pressure_peak)
    c0 = fem.Constant(domain, (contact.x0, contact.y_mid, contact.z_mid))
    v_ice = fem.Constant(domain, ((contact.x1 - contact.x0) / contact.duration, 0.0, 0.0))
    p = moving_gaussian_pressure(domain, c0, v_ice, t, sigma, p0)
    f_ice = p * model.e3
    L = ufl.dot(f_ice, model.u_test) * ufl.dx

    def update(tn: float):
        p0.value = cfg.pressure_peak * contact.ramp(tn)

    return LoadController(L=L, extra_bcs=[], update=update)


def _make_dirichlet_normal_loading(model, cfg, t, contact: ContactKinematics) -> LoadController:
    domain = model.domain
    zero_vec = fem.Constant(domain, (0.0, 0.0, 0.0))
    L = ufl.dot(zero_vec, model.u_test) * ufl.dx

    radius_y = cfg.iceberg_patch_radius_factor * cfg.sigma
    radius_z = cfg.iceberg_patch_radius_factor * cfg.sigma
    waterline_z = getattr(cfg, "waterline_z", np.inf)

    def impact_region(x):
        y_scaled = (x[1] - contact.y_mid) / max(radius_y, 1e-12)
        z_scaled = (x[2] - contact.z_mid) / max(radius_z, 1e-12)
        inside_patch = (y_scaled * y_scaled + z_scaled * z_scaled) <= 1.0
        submerged = x[2] <= waterline_z
        return inside_patch & submerged

    ice_dofs = fem.locate_dofs_geometrical((model.V.sub(0), model.Vu), impact_region)
    u_ice = fem.Function(model.Vu, name="IcebergDisplacement")
    extra_bcs = [fem.dirichletbc(u_ice, ice_dofs, model.V.sub(0))]

    sigma = fem.Constant(domain, float(cfg.sigma))
    x_center = fem.Constant(domain, float(contact.x0))
    disp_scale = fem.Constant(domain, 0.0)
    x = ufl.SpatialCoordinate(domain)
    r2 = (x[0] - x_center) ** 2 + (x[2] - contact.z_mid) ** 2
    disp_amp = disp_scale * ufl.exp(-r2 / (2 * sigma**2))
    u_ice_expr = disp_amp * model.e3
    u_ice_eval = fem.Expression(u_ice_expr, model.Vu.element.interpolation_points)

    def update(tn: float):
        progress = contact.progress(tn)
        x_center.value = contact.x0 + (contact.x1 - contact.x0) * progress
        disp_scale.value = contact.ramp(tn) * cfg.iceberg_disp_sign * cfg.iceberg_disp_peak
        u_ice.interpolate(u_ice_eval)

    return LoadController(L=L, extra_bcs=extra_bcs, update=update)


def _build_load_controller(model, cfg, t, contact: ContactKinematics) -> LoadController:
    builders = {
        "neumann_pressure": _make_neumann_pressure_loading,
        "dirichlet_displacement": _make_dirichlet_normal_loading,
    }
    try:
        return builders[cfg.iceberg_loading](model, cfg, t, contact)
    except KeyError as exc:
        raise ValueError(
            f"Unknown cfg.iceberg_loading='{cfg.iceberg_loading}'. "
            "Expected 'neumann_pressure' or 'dirichlet_displacement'."
        ) from exc


def _build_output_controls(cfg, pf: PhaseFieldContext) -> OutputControls:
    vtk_stride = int(getattr(cfg, "ecrire_vtk_tous_les_n_pas", getattr(cfg, "vtk_write_stride", 1)))
    write_rotation = bool(getattr(cfg, "write_rotation_vtk", True))
    write_damage = bool(getattr(cfg, "write_damage_vtk", True))
    if not pf.enabled and not bool(getattr(cfg, "write_damage_vtk_if_disabled", False)):
        write_damage = False
    print_stride = int(getattr(cfg, "afficher_console_tous_les_n_pas", getattr(cfg, "monitor_print_stride", 1)))
    return OutputControls(
        vtk_stride=vtk_stride,
        write_rotation=write_rotation,
        write_damage=write_damage,
        print_stride=print_stride,
    )


def _write_step_outputs(disp_vtk, rot_vtk, damage_vtk, controls: OutputControls, u_out, theta_out, damage, n, tn, n_last):
    if not ((n % controls.vtk_stride == 0) or (n == n_last)):
        return
    disp_vtk.write_function(u_out, tn)
    if rot_vtk is not None:
        rot_vtk.write_function(theta_out, tn)
    if damage_vtk is not None:
        damage_vtk.write_function(damage, tn)


def _append_monitor_row(monitor_rows, n, tn, u_out, damage, step_wall_s, stats: StepStats):
    u_max = np.linalg.norm(u_out.x.array, ord=np.inf)
    max_d = float(np.max(damage.x.array))
    mean_d = float(np.mean(damage.x.array))
    frac_d95 = float(np.mean(damage.x.array >= 0.95))
    monitor_rows.append(
        (
            n,
            tn,
            u_max,
            max_d,
            mean_d,
            frac_d95,
            step_wall_s,
            stats.mech_wall_s,
            stats.damage_wall_s,
        )
    )
    return u_max, max_d, mean_d, frac_d95


def _print_monitor_line(cfg, controls: OutputControls, n, n_last, tn, u_max, max_d, mean_d, frac_d95, step_wall_s, stats):
    if MPI.COMM_WORLD.rank != 0:
        return
    if not ((n % controls.print_stride == 0) or (n == n_last)):
        return
    print(
        f"Step {n}/{n_last}, t={tn:.3e}, "
        f"max|u|={u_max:.3e}, max(d)={max_d:.3e}, "
        f"mean(d)={mean_d:.3e}, frac(d>=0.95)={frac_d95:.3e}, "
        f"temps_pas={step_wall_s:.2f}s (meca={stats.mech_wall_s:.2f}s, "
        f"phase_field={stats.damage_wall_s:.2f}s)"
    )


def run_quasi_static(model, cfg, output_layout, phase_field_preset=None):
    domain = model.domain
    t = fem.Constant(domain, 0.0)

    # 1) Contact/chargement iceberg (geometrie + cinematique)
    contact = _build_contact_kinematics(domain, cfg)
    load = _build_load_controller(model, cfg, t, contact)

    # 2) Probleme mecanique (coque degradee par `model.damage_state`)
    mechanics_petsc_options = cfg.mechanics_petsc_options or cfg.petsc_options
    problem_u = dolfinx.fem.petsc.LinearProblem(
        model.a,
        load.L,
        u=model.v,
        bcs=model.bcs + load.extra_bcs,
        petsc_options=mechanics_petsc_options,
        petsc_options_prefix="coque",
    )

    # 3) Variables de sortie + phase-field global
    u_out = fem.Function(model.Vu, name="Displacement")
    theta_out = fem.Function(model.Vtheta, name="Rotation")
    pf = build_phase_field_context(model, cfg, phase_field_preset)
    controls = _build_output_controls(cfg, pf)

    if MPI.COMM_WORLD.rank == 0 and pf.enabled:
        print(f"[phase-field] irreversibilite: {phase_field_irreversibility_mode(pf)}")

    # 4) Temps + sorties
    time_steps = _build_time_steps(cfg, contact)
    monitor_rows = []
    n_last = len(time_steps) - 1

    rot_ctx = io.VTKFile(MPI.COMM_WORLD, output_layout["rotation_file"], "w") if controls.write_rotation else nullcontext(None)
    dmg_ctx = io.VTKFile(MPI.COMM_WORLD, output_layout["damage_file"], "w") if controls.write_damage else nullcontext(None)

    with io.VTKFile(MPI.COMM_WORLD, output_layout["displacement_file"], "w") as disp_vtk:
        with rot_ctx as rot_vtk:
            with dmg_ctx as damage_vtk:
                # 5) Boucle en temps quasi-statique (lecture lineaire, style TD)
                for n, tn in enumerate(time_steps):
                    step_t0 = perf_counter()
                    t.value = tn
                    load.update(float(tn))

                    stats = advance_quasi_static_step(problem_u, pf, n, len(time_steps))

                    u_out.interpolate(model.v.sub(0))
                    theta_out.interpolate(model.v.sub(1))
                    _write_step_outputs(
                        disp_vtk,
                        rot_vtk,
                        damage_vtk,
                        controls,
                        u_out,
                        theta_out,
                        pf.damage,
                        n,
                        tn,
                        n_last,
                    )

                    step_wall_s = perf_counter() - step_t0
                    u_max, max_d, mean_d, frac_d95 = _append_monitor_row(
                        monitor_rows,
                        n,
                        tn,
                        u_out,
                        pf.damage,
                        step_wall_s,
                        stats,
                    )
                    _print_monitor_line(
                        cfg,
                        controls,
                        n,
                        n_last,
                        tn,
                        u_max,
                        max_d,
                        mean_d,
                        frac_d95,
                        step_wall_s,
                        stats,
                    )

    if MPI.COMM_WORLD.rank == 0:
        _write_monitor_csv(output_layout["monitor_file"], monitor_rows)
