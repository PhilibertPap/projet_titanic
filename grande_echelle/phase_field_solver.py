from dataclasses import dataclass
from time import perf_counter

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem
import dolfinx.fem.petsc


@dataclass
class PhaseFieldContext:
    enabled: bool
    damage: any
    damage_prev: any
    damage_iter_prev: any
    history: any
    psi_drive: any
    psi_eval: any
    problem_d: any
    damage_vi: any
    maj_stride: int
    seuil: float
    alt_max_iters: int
    alt_min_iters: int
    alt_tol: float


@dataclass
class StepStats:
    mech_wall_s: float = 0.0
    damage_wall_s: float = 0.0


def _try_build_damage_vi_solver(Vd, damage, F_d, J_d, cfg):
    if not bool(getattr(cfg, "phase_field_use_snes_vi", True)):
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


def build_phase_field_context(model, cfg, phase_field_preset):
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

    enabled = bool(cfg.enable_global_phase_field)
    common = dict(
        enabled=enabled,
        damage=damage,
        damage_prev=damage_prev,
        damage_iter_prev=damage_iter_prev,
        history=history,
        psi_drive=None,
        psi_eval=None,
        problem_d=None,
        damage_vi=None,
        maj_stride=max(1, int(getattr(cfg, "phase_field_mise_a_jour_tous_les_n_pas", 1))),
        seuil=float(getattr(cfg, "phase_field_seuil_nucleation_j_m3", 0.0)),
        alt_max_iters=max(1, int(getattr(cfg, "phase_field_n_alt_iters", 6))),
        alt_min_iters=max(1, int(getattr(cfg, "phase_field_alt_min_iters", 1))),
        alt_tol=float(getattr(cfg, "phase_field_alt_tol", 1e-4)),
    )
    if not enabled:
        return PhaseFieldContext(**common)

    domain = model.domain
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

    common.update(
        psi_drive=psi_drive,
        psi_eval=psi_eval,
        problem_d=problem_d,
        damage_vi=damage_vi,
    )
    return PhaseFieldContext(**common)


def phase_field_irreversibility_mode(pf: PhaseFieldContext) -> str:
    return "SNESVI bounds" if pf.damage_vi is not None else "projected bounds fallback"


def _update_history_field(pf: PhaseFieldContext):
    pf.psi_drive.interpolate(pf.psi_eval)
    psi_effective = np.maximum(pf.psi_drive.x.array - pf.seuil, 0.0)
    pf.history.x.array[:] = np.maximum(pf.history.x.array, psi_effective)


def _solve_damage_bounded(pf: PhaseFieldContext, stats: StepStats):
    damage_t0 = perf_counter()
    if pf.damage_vi is not None:
        try:
            pf.damage_vi["lb"].x.array[:] = pf.damage_prev.x.array
            pf.damage_vi["ub"].x.array[:] = 1.0
            pf.damage_vi["solver"].setVariableBounds(
                pf.damage_vi["lb"].x.petsc_vec,
                pf.damage_vi["ub"].x.petsc_vec,
            )
            pf.damage_vi["solver"].solve(None, pf.damage.x.petsc_vec)
        except Exception:
            if MPI.COMM_WORLD.rank == 0:
                print("[phase-field] SNESVI indisponible au solve -> fallback projection")
            pf.damage_vi = None
            pf.problem_d.solve()
    else:
        pf.problem_d.solve()
    stats.damage_wall_s += perf_counter() - damage_t0
    pf.damage.x.array[:] = np.clip(pf.damage.x.array, pf.damage_prev.x.array, 1.0)


def _solve_mechanics(problem_u, stats: StepStats):
    mech_t0 = perf_counter()
    problem_u.solve()
    stats.mech_wall_s += perf_counter() - mech_t0


def advance_quasi_static_step(problem_u, pf: PhaseFieldContext, step_index: int, n_steps: int) -> StepStats:
    stats = StepStats()
    is_last_step = step_index == n_steps - 1
    do_damage_update = pf.enabled and ((step_index % pf.maj_stride == 0) or is_last_step)

    if not pf.enabled:
        pf.damage.x.array[:] = 0.0
        _solve_mechanics(problem_u, stats)
        return stats

    if not do_damage_update:
        pf.damage.x.array[:] = pf.damage_prev.x.array
        _solve_mechanics(problem_u, stats)
        _update_history_field(pf)
        pf.damage.x.array[:] = pf.damage_prev.x.array
        return stats

    pf.damage.x.array[:] = pf.damage_prev.x.array
    pf.damage_iter_prev.x.array[:] = pf.damage_prev.x.array
    for k in range(pf.alt_max_iters):
        _solve_mechanics(problem_u, stats)
        _update_history_field(pf)
        _solve_damage_bounded(pf, stats)

        alt_increment = np.linalg.norm(pf.damage.x.array - pf.damage_iter_prev.x.array, ord=np.inf)
        pf.damage_iter_prev.x.array[:] = pf.damage.x.array
        if k + 1 >= pf.alt_min_iters and alt_increment < pf.alt_tol:
            break

    pf.damage_prev.x.array[:] = pf.damage.x.array
    return stats
