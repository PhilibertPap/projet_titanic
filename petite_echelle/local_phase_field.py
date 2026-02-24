from dataclasses import dataclass
import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import mesh, fem, io
import dolfinx.fem.petsc


@dataclass
class LocalPhaseFieldRunConfig:
    case_name: str = "local_phase_field_baseline"
    results_root: str = "petite_echelle/results"
    baseline_file: str = "petite_echelle/results/titanic_local_calibration/baseline.json"
    young_modulus_pa: float = 210e9
    poisson_ratio: float = 0.30
    residual_stiffness: float = 1e-6
    length_m: float = 8.0
    height_m: float = 2.0
    nx: int = 120
    ny: int = 30
    notch_length_m: float = 0.30
    notch_half_thickness_m: float = 0.02
    initial_notch_damage: float = 0.95
    uy_max_m: float = 0.03
    n_steps: int = 20
    n_alt_iters: int = 10
    alt_tol: float = 1e-4
    use_tension_compression_split: bool = True
    ksp_type: str = "preonly"
    pc_type: str = "lu"
    ecrire_vtk_tous_les_n_pas: int = 1
    afficher_console_tous_les_n_pas: int = 1


def _load_baseline(config: LocalPhaseFieldRunConfig) -> tuple[float, float]:
    baseline_path = Path(config.baseline_file)
    if not baseline_path.exists():
        return 7000.0, 1.5
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    return float(baseline["Gc_J_m2"]), float(baseline["l0_m"])


def _build_mesh(config: LocalPhaseFieldRunConfig):
    return mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([config.length_m, config.height_m])],
        [config.nx, config.ny],
        cell_type=mesh.CellType.triangle,
    )


def run_local_phase_field(config: LocalPhaseFieldRunConfig | None = None) -> Path:
    if config is None:
        config = LocalPhaseFieldRunConfig()

    gc_value, l0_value = _load_baseline(config)
    domain = _build_mesh(config)

    Vu = fem.functionspace(domain, ("CG", 1, (2,)))
    Vd = fem.functionspace(domain, ("CG", 1))

    u = fem.Function(Vu, name="Displacement")
    d = fem.Function(Vd, name="Damage")
    d_old = fem.Function(Vd, name="DamageOld")

    def left(x):
        return np.isclose(x[0], 0.0)

    def top(x):
        return np.isclose(x[1], config.height_m)

    def notch(x):
        return (x[0] <= config.notch_length_m) & (
            np.abs(x[1] - 0.5 * config.height_m) <= config.notch_half_thickness_m
        )

    Vux, _ = Vu.sub(0).collapse()
    Vuy, _ = Vu.sub(1).collapse()
    ux0 = fem.Function(Vux)
    uy0 = fem.Function(Vuy)
    uy_top = fem.Function(Vuy)
    uy_top.x.array[:] = 0.0

    left_dofs_x = fem.locate_dofs_geometrical((Vu.sub(0), Vux), left)
    left_dofs_y = fem.locate_dofs_geometrical((Vu.sub(1), Vuy), left)
    top_dofs_y = fem.locate_dofs_geometrical((Vu.sub(1), Vuy), top)
    bcs_u = [
        fem.dirichletbc(ux0, left_dofs_x, Vu.sub(0)),
        fem.dirichletbc(uy0, left_dofs_y, Vu.sub(1)),
        fem.dirichletbc(uy_top, top_dofs_y, Vu.sub(1)),
    ]

    notch_dofs = fem.locate_dofs_geometrical(Vd, notch)
    d.x.array[:] = 0.0
    d.x.array[notch_dofs] = config.initial_notch_damage
    d_old.x.array[:] = d.x.array

    E = fem.Constant(domain, config.young_modulus_pa)
    nu = fem.Constant(domain, config.poisson_ratio)
    gc = fem.Constant(domain, gc_value)
    l0 = fem.Constant(domain, l0_value)
    k_res = fem.Constant(domain, config.residual_stiffness)
    f_zero = fem.Constant(domain, (0.0, 0.0))

    lmbda = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def psi_bulk(w):
        return 0.5 * lmbda * ufl.tr(eps(w)) ** 2 + mu * ufl.inner(eps(w), eps(w))

    k_bulk = lmbda + 2.0 * mu / 3.0

    def positive_part(x):
        return 0.5 * (x + ufl.sqrt(x * x))

    def deviatoric(tensor):
        return tensor - (ufl.tr(tensor) / 2.0) * ufl.Identity(2)

    def psi_tension(w):
        e = eps(w)
        tr_e = ufl.tr(e)
        dev_e = deviatoric(e)
        # Amor split: only tensile volumetric part contributes to damage driving force
        return 0.5 * k_bulk * positive_part(tr_e) ** 2 + mu * ufl.inner(dev_e, dev_e)

    def sigma(w):
        return lmbda * ufl.tr(eps(w)) * ufl.Identity(2) + 2.0 * mu * eps(w)

    du = ufl.TrialFunction(Vu)
    v = ufl.TestFunction(Vu)
    dd = ufl.TrialFunction(Vd)
    eta = ufl.TestFunction(Vd)

    results_dir = Path(config.results_root) / config.case_name
    results_dir.mkdir(parents=True, exist_ok=True)
    if MPI.COMM_WORLD.rank == 0:
        metadata = {
            "Gc_J_m2": gc_value,
            "l0_m": l0_value,
            "young_modulus_pa": config.young_modulus_pa,
            "poisson_ratio": config.poisson_ratio,
            "mesh_nx": config.nx,
            "mesh_ny": config.ny,
            "length_m": config.length_m,
            "height_m": config.height_m,
        }
        (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    time_values = np.linspace(0.0, 1.0, config.n_steps + 1)
    monitor_rows = []
    petsc_options = {"ksp_type": config.ksp_type, "pc_type": config.pc_type}

    # Staggered scheme: reuse assembled problem objects while updating the coefficient fields
    # (`d_old`, `u`) between iterations.
    g = (1.0 - d_old) ** 2 + k_res
    a_u = ufl.inner(g * sigma(du), eps(v)) * ufl.dx
    L_u = ufl.dot(f_zero, v) * ufl.dx
    problem_u = dolfinx.fem.petsc.LinearProblem(
        a_u,
        L_u,
        u=u,
        bcs=bcs_u,
        petsc_options_prefix="pf_u",
        petsc_options=petsc_options,
    )

    psi_u = psi_bulk(u)
    psi_drive_expr = psi_tension(u) if config.use_tension_compression_split else psi_u
    a_d = (
        gc * l0 * ufl.dot(ufl.grad(dd), ufl.grad(eta))
        + (gc / l0 + 2.0 * psi_drive_expr) * dd * eta
    ) * ufl.dx
    L_d = (2.0 * psi_drive_expr) * eta * ufl.dx
    problem_d = dolfinx.fem.petsc.LinearProblem(
        a_d,
        L_d,
        u=d,
        bcs=[],
        petsc_options_prefix="pf_d",
        petsc_options=petsc_options,
    )

    with io.VTKFile(MPI.COMM_WORLD, results_dir / "displacement.pvd", "w") as disp_vtk:
        with io.VTKFile(MPI.COMM_WORLD, results_dir / "damage.pvd", "w") as dam_vtk:
            for step, tau in enumerate(time_values):
                uy_top.x.array[:] = tau * config.uy_max_m

                step_t0 = perf_counter()
                mech_wall_s = 0.0
                damage_wall_s = 0.0
                for _ in range(config.n_alt_iters):
                    # Solve displacement with frozen damage
                    mech_t0 = perf_counter()
                    problem_u.solve()
                    mech_wall_s += perf_counter() - mech_t0

                    damage_t0 = perf_counter()
                    problem_d.solve()
                    damage_wall_s += perf_counter() - damage_t0

                    d_new = np.clip(d.x.array, 0.0, 1.0)
                    d_new = np.maximum(d_old.x.array, d_new)  # irreversibility
                    increment = np.linalg.norm(d_new - d_old.x.array, ord=np.inf)
                    d_old.x.array[:] = d_new
                    d.x.array[:] = d_new
                    if increment < config.alt_tol:
                        break

                if (step % config.ecrire_vtk_tous_les_n_pas == 0) or (step == config.n_steps):
                    disp_vtk.write_function(u, float(step))
                    dam_vtk.write_function(d, float(step))

                max_u = np.linalg.norm(u.x.array, ord=np.inf)
                max_d = float(np.max(d.x.array))
                mean_d = float(np.mean(d.x.array))
                frac_d95 = float(np.mean(d.x.array >= 0.95))
                step_wall_s = perf_counter() - step_t0
                monitor_rows.append(
                    (step, tau, max_u, max_d, mean_d, frac_d95, step_wall_s, mech_wall_s, damage_wall_s)
                )
                if MPI.COMM_WORLD.rank == 0 and (
                    step % config.afficher_console_tous_les_n_pas == 0 or step == config.n_steps
                ):
                    print(
                        f"Step {step}/{config.n_steps}, tau={tau:.3f}, "
                        f"max|u|={max_u:.3e}, max(d)={max_d:.3e}, "
                        f"mean(d)={mean_d:.3e}, frac(d>=0.95)={frac_d95:.3e}, "
                        f"temps_pas={step_wall_s:.2f}s (u={mech_wall_s:.2f}s, phase_field={damage_wall_s:.2f}s)"
                    )

    if MPI.COMM_WORLD.rank == 0:
        with (results_dir / "monitor.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "tau",
                    "max_u_inf",
                    "max_damage",
                    "mean_damage",
                    "frac_damage_ge_095",
                    "temps_pas_s",
                    "temps_u_s",
                    "temps_phase_field_s",
                ]
            )
            writer.writerows(monitor_rows)

    return results_dir
