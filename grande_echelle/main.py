from __future__ import annotations

from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from mpi4py import MPI
from dolfinx import fem, io

try:
    from .shell import build_shell_model
    from .quasi_static import run_quasi_static
except ImportError:  # pragma: no cover - script execution fallback
    from shell import build_shell_model
    from quasi_static import run_quasi_static


def config_par_defaut() -> dict:
    iceberg_zone_x_debut_m = 177.0
    iceberg_zone_x_fin_m = 268.0

    # 8 bandes rivets homogenisees, verticales (dirigees selon z) et
    # reparties regulierement selon x dans la zone de passage de l'iceberg.
    n_bandes_rivets_x = 8
    largeur_bande_x_m = 0.30
    x_margin = 0.5 * largeur_bande_x_m
    x_centres = np.linspace(
        iceberg_zone_x_debut_m + x_margin,
        iceberg_zone_x_fin_m - x_margin,
        n_bandes_rivets_x,
    )
    bandes_rivets = []
    for i_bande, x_centre in enumerate(x_centres, start=1):
        bandes_rivets.append(
            {
                "nom": f"bande_x_{i_bande:02d}",
                "x_centre_m": float(x_centre),
                "largeur_x_m": largeur_bande_x_m,
                "z_min_m": -10.2,
                "z_max_m": 0.2,
                "facteur_E": 0.98,
                "facteur_epaisseur": 1.00,
                "facteur_Gc": 0.88,
            }
        )

    return {
        # Maillage / sorties
        "mesh_stem": "mesh/coque",
        "results_root": "results",
        "case_name": "titanic",
        "shell_cell_tag": 1,
        "rivet_cell_tag": 2,
        # Materiau coque / rivets homogenises
        "shell_thickness": 2.5e-2,
        "shell_young_modulus": 210e9,
        "shell_poisson_ratio": 0.3,
        "rivet_thickness": 2.5e-2,
        "rivet_young_modulus": 190e9,
        "rivet_poisson_ratio": 0.3,
        # Chargement iceberg (Dirichlet normal local)
        "sigma": 3.0,
        "ramp_amplitude_iceberg": True,
        "iceberg_disp_peak": 2.0e-2,
        "iceberg_disp_sign": 1.0,
        "iceberg_patch_radius_factor": 3.0,
        # Temps / sorties
        "t_final": 8.5,
        "num_steps": 240,
        "ecrire_vtk_tous_les_n_pas": 1,
        "afficher_console_tous_les_n_pas": 1,
        "write_local_frame_outputs": True,
        "write_rotation_vtk": True,
        "write_damage_vtk": True,
        "write_damage_vtk_if_disabled": False,
        # CL sur les bords tags Gmsh
        "left_facet_tag": 1,
        "right_facet_tag": 2,
        "bottom_facet_tag": 3,
        "top_facet_tag": 4,
        "clamp_all_edges": True,
        "clamp_rotations": True,
        # Zone d'impact / trajectoire
        "iceberg_center_y": -10.8,
        "iceberg_zone_x_debut_m": iceberg_zone_x_debut_m,
        "iceberg_zone_x_fin_m": iceberg_zone_x_fin_m,
        "waterline_z": 0.0,
        "iceberg_depth_below_waterline": 6.3,
        "iceberg_moves_from_xmax_to_xmin": True,
        "iceberg_max_dx_par_pas_m": None,
        "iceberg_contact_t_start": 0.0,
        "iceberg_contact_t_end": 8.5,
        # Phase-field global
        "phase_field_preset_file": None,
        "enable_global_phase_field": True,
        "phase_field_use_selected_preset": False,
        "phase_field_gc_j_m2": 7000.0,
        "phase_field_l0_m": 0.4,
        "phase_field_residual_stiffness": 1e-6,
        "phase_field_split_traction_compression": True,
        "phase_field_seuil_nucleation_j_m3": 8.0e4,
        "phase_field_mise_a_jour_tous_les_n_pas": 1,
        # Homogeneisation des rivets en bandes selon z
        "utiliser_bandes_rivets_z": True,
        "bandes_rivets_z": bandes_rivets,
        "rivet_bandes_preset_file": None,
        # Solveurs PETSc
        "mechanics_petsc_options": None,
        "damage_petsc_options": None,
        "petsc_options": {"ksp_type": "preonly", "pc_type": "lu"},
    }


def creer_config(**updates):
    data = config_par_defaut()
    data.update(updates)
    return SimpleNamespace(**data)


def config_vers_dict(cfg) -> dict:
    data = dict(vars(cfg))
    if data.get("mechanics_petsc_options") is None:
        data["mechanics_petsc_options"] = dict(data["petsc_options"])
    if data.get("damage_petsc_options") is None:
        data["damage_petsc_options"] = dict(data["petsc_options"])
    return data


def verifier_config(cfg) -> None:
    # Defaults optionnels pour compatibilite pratique des anciens objets config
    if not hasattr(cfg, "utiliser_bandes_rivets_z"):
        cfg.utiliser_bandes_rivets_z = False
    if not hasattr(cfg, "bandes_rivets_z"):
        cfg.bandes_rivets_z = []
    if not hasattr(cfg, "rivet_bandes_preset_file"):
        cfg.rivet_bandes_preset_file = None
    if not hasattr(cfg, "iceberg_max_dx_par_pas_m"):
        cfg.iceberg_max_dx_par_pas_m = None
    if not hasattr(cfg, "iceberg_contact_t_start"):
        cfg.iceberg_contact_t_start = 0.0
    if not hasattr(cfg, "iceberg_contact_t_end"):
        cfg.iceberg_contact_t_end = cfg.t_final
    if not hasattr(cfg, "write_local_frame_outputs"):
        cfg.write_local_frame_outputs = True
    if not hasattr(cfg, "write_rotation_vtk"):
        cfg.write_rotation_vtk = True
    if not hasattr(cfg, "write_damage_vtk"):
        cfg.write_damage_vtk = True
    if not hasattr(cfg, "write_damage_vtk_if_disabled"):
        cfg.write_damage_vtk_if_disabled = False

    if cfg.num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if cfg.t_final <= 0.0:
        raise ValueError("t_final must be > 0")
    if cfg.ecrire_vtk_tous_les_n_pas <= 0:
        raise ValueError("ecrire_vtk_tous_les_n_pas must be >= 1")
    if cfg.afficher_console_tous_les_n_pas <= 0:
        raise ValueError("afficher_console_tous_les_n_pas must be >= 1")
    for name in ("iceberg_center_y", "waterline_z", "iceberg_depth_below_waterline"):
        if not hasattr(cfg, name):
            raise ValueError(f"Missing required config field: {name}")


DEFAULT_CONFIG = creer_config()


def config_apercu_rapide():
    return creer_config(
        case_name="preview_paraview_ultra_fast",
        num_steps=8,
        t_final=0.8,
        ecrire_vtk_tous_les_n_pas=1,
        sigma=2.5,
    )


def config_etude_rivets(with_rivets: bool, base=None):
    base = DEFAULT_CONFIG if base is None else base
    data = config_vers_dict(base)
    data["case_name"] = f"{base.case_name}_{'with' if with_rivets else 'without'}_rivets"
    if not with_rivets:
        data["rivet_young_modulus"] = data["shell_young_modulus"]
        data["rivet_poisson_ratio"] = data["shell_poisson_ratio"]
        data["rivet_thickness"] = data["shell_thickness"]
        data["utiliser_bandes_rivets_z"] = False
    return SimpleNamespace(**data)


def config_etude_rivets_rapide(with_rivets: bool = True):
    base = creer_config(
        case_name="rivets_rapide",
        num_steps=20,
        ecrire_vtk_tous_les_n_pas=1,
        afficher_console_tous_les_n_pas=2,
        iceberg_disp_peak=1.5e-2,
        temps_relatifs=[
            0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.56, 0.62, 0.68, 0.74, 0.80, 0.86, 0.92, 0.96, 1.0,
        ],
        phase_field_mise_a_jour_tous_les_n_pas=1,
        iceberg_max_dx_par_pas_m=1.0,
    )
    return config_etude_rivets(with_rivets=with_rivets, base=base)


def config_etude_rivets_production(with_rivets: bool = True):
    base = creer_config(
        case_name="titanic_rivets_production",
        num_steps=36,
        ecrire_vtk_tous_les_n_pas=1,
        afficher_console_tous_les_n_pas=3,
        iceberg_disp_peak=1.5e-2,
        temps_relatifs=[
            0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86,
            0.90, 0.94, 0.97, 1.0,
        ],
        mechanics_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        damage_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        iceberg_max_dx_par_pas_m=0.5,
    )
    return config_etude_rivets(with_rivets=with_rivets, base=base)


def config_etude_rivets_screening(with_rivets: bool = True):
    base = creer_config(
        case_name="rivets_screening",
        num_steps=16,
        ecrire_vtk_tous_les_n_pas=4,
        afficher_console_tous_les_n_pas=4,
        write_local_frame_outputs=False,
        write_rotation_vtk=False,
        write_damage_vtk=False,
        phase_field_mise_a_jour_tous_les_n_pas=2,
        iceberg_max_dx_par_pas_m=3.0,
        iceberg_disp_peak=1.5e-2,
    )
    return config_etude_rivets(with_rivets=with_rivets, base=base)


def _charger_bandes_rivets_preset_si_disponible(cfg) -> None:
    preset_file = getattr(cfg, "rivet_bandes_preset_file", None)
    if not preset_file:
        auto_rel = Path("rivet/bandes_rivets_grande_echelle_calibre.json")
        auto_candidates = [Path.cwd() / auto_rel, Path(__file__).resolve().parent.parent / auto_rel]
        auto_found = next((p for p in auto_candidates if p.exists()), None)
        if auto_found is None:
            return
        preset_file = str(auto_found)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Auto-using calibrated rivet bands preset: {auto_found}")

    candidat = Path(preset_file)
    if candidat.is_absolute():
        preset_candidates = [candidat]
    else:
        preset_candidates = [Path.cwd() / candidat, Path(__file__).resolve().parent / candidat]
    preset_path = next((p for p in preset_candidates if p.exists()), None)
    if preset_path is None:
        print(f"Rivet preset not found. Tried: {preset_candidates}")
        return

    data = json.loads(preset_path.read_text(encoding="utf-8"))
    bandes = data.get("bandes_rivets_z")
    if not isinstance(bandes, list):
        raise ValueError(f"Invalid rivet preset format in {preset_path}: expected key 'bandes_rivets_z' (list)")

    cfg.bandes_rivets_z = bandes
    cfg.utiliser_bandes_rivets_z = True
    cfg.rivet_bandes_preset_file = str(preset_path)
    print(f"Using rivet bands preset: {preset_path}")


def lancer_comparaison_rivets_rapide():
    print("=== Cas 1 : avec effet des rivets ===")
    lancer_calcul(config_etude_rivets_rapide(with_rivets=True))
    print("=== Cas 2 : sans effet des rivets ===")
    lancer_calcul(config_etude_rivets_rapide(with_rivets=False))
    print("Comparaison terminee.")
    print("Comparer les fichiers monitor.csv et les champs de dommage dans results/.")


def lancer_comparaison_rivets_production():
    print("=== Cas 1 : avec effet des rivets ===")
    lancer_calcul(config_etude_rivets_production(with_rivets=True))
    print("=== Cas 2 : sans effet des rivets ===")
    lancer_calcul(config_etude_rivets_production(with_rivets=False))
    print("Comparaison terminee.")
    print("Comparer les fichiers monitor.csv et les champs de dommage dans results/.")


def lancer_comparaison_rivets_screening():
    print("=== Cas 1 : avec effet des rivets (screening) ===")
    lancer_calcul(config_etude_rivets_screening(with_rivets=True))
    print("=== Cas 2 : sans effet des rivets (screening) ===")
    lancer_calcul(config_etude_rivets_screening(with_rivets=False))
    print("Comparaison screening terminee.")
    print("Regarder d'abord monitor.csv, puis relancer en mode rapide/production si besoin.")


def analyser_monitor_csv(monitor_csv_file) -> dict:
    path = Path(monitor_csv_file)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"monitor_csv_file": str(path), "n_steps": 0}

    total_step_s = sum(float(r.get("temps_pas_s", 0.0)) for r in rows)
    total_mech_s = sum(float(r.get("temps_meca_s", r.get("temps_u_s", 0.0))) for r in rows)
    total_pf_s = sum(float(r.get("temps_phase_field_s", 0.0)) for r in rows)
    return {
        "monitor_csv_file": str(path),
        "n_steps": len(rows),
        "total_step_s": total_step_s,
        "total_mech_s": total_mech_s,
        "total_phase_field_s": total_pf_s,
        "fraction_mech": (total_mech_s / total_step_s) if total_step_s > 0 else None,
        "fraction_phase_field": (total_pf_s / total_step_s) if total_step_s > 0 else None,
        "fraction_other": ((total_step_s - total_mech_s - total_pf_s) / total_step_s) if total_step_s > 0 else None,
        "last_max_damage": float(rows[-1]["max_damage"]),
        "last_mean_damage": float(rows[-1]["mean_damage"]),
    }


def _print_field_min_max(name: str, field) -> None:
    local_min = float(np.min(field.x.array)) if field.x.array.size else np.inf
    local_max = float(np.max(field.x.array)) if field.x.array.size else -np.inf
    gmin = field.function_space.mesh.comm.allreduce(local_min, op=MPI.MIN)
    gmax = field.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)
    if field.function_space.mesh.comm.rank == 0:
        print(f"{name}: min={gmin:.6g}, max={gmax:.6g}")


def lancer_calcul(cfg=None):
    cfg = DEFAULT_CONFIG if cfg is None else cfg
    verifier_config(cfg)
    _charger_bandes_rivets_preset_si_disponible(cfg)

    # ============================================================
    # 1) Lecture du maillage
    # ============================================================
    candidat = Path(cfg.mesh_stem)
    if candidat.is_absolute():
        mesh_candidates = [candidat.with_suffix(".msh")]
    else:
        mesh_candidates = [
            (Path.cwd() / candidat).with_suffix(".msh"),
            (Path(__file__).resolve().parent / candidat).with_suffix(".msh"),
        ]
    mesh_path = next((p for p in mesh_candidates if p.exists()), None)
    if mesh_path is None:
        raise FileNotFoundError(f"Mesh file not found. Tried: {', '.join(str(p) for p in mesh_candidates)}")

    meshdata = io.gmsh.read_from_msh(str(mesh_path), MPI.COMM_WORLD)
    domain = meshdata.mesh
    cell_tags = meshdata.cell_tags
    facets = meshdata.facet_tags

    cell_values = [] if cell_tags is None else [int(v) for v in np.unique(cell_tags.values)]
    facet_values = [] if facets is None else [int(v) for v in np.unique(facets.values)]
    print(f"Mesh diagnostics: gdim={domain.geometry.dim}, tdim={domain.topology.dim}")
    print(f"Cell tags present: {cell_tags is not None}; values={cell_values}")
    print(f"Facet tags present: {facets is not None}; values={facet_values}")

    # ============================================================
    # 2) Dossiers de sortie
    # ============================================================
    base_dir = Path(cfg.results_root) / cfg.case_name
    local_frame_dir = base_dir / "local_frame"
    quasi_static_dir = base_dir / "quasi_static"
    local_frame_dir.mkdir(exist_ok=True, parents=True)
    quasi_static_dir.mkdir(exist_ok=True, parents=True)

    output_layout = {
        "base_dir": base_dir,
        "local_frame_dir": local_frame_dir,
        "quasi_static_dir": quasi_static_dir,
        "local_frame_file": local_frame_dir / "local_basis_vectors.pvd",
        "displacement_file": quasi_static_dir / "displacement.pvd",
        "rotation_file": quasi_static_dir / "rotation.pvd",
        "damage_file": quasi_static_dir / "damage.pvd",
        "monitor_file": quasi_static_dir / "monitor.csv",
        "metadata_file": base_dir / "run_metadata.json",
    }

    # ============================================================
    # 3) Preset phase-field global (optionnel)
    # ============================================================
    if not cfg.phase_field_preset_file:
        phase_field_preset = None
        print("No phase-field preset file configured (using config values for Gc/l0).")
    else:
        preset_cfg_path = Path(cfg.phase_field_preset_file)
        if preset_cfg_path.is_absolute():
            preset_candidates = [preset_cfg_path]
        else:
            preset_candidates = [Path.cwd() / preset_cfg_path, Path(__file__).resolve().parent / preset_cfg_path]
        preset_path = next((p for p in preset_candidates if p.exists()), None)
        if preset_path is None:
            phase_field_preset = None
            print(f"Phase-field preset not found. Tried: {preset_candidates}")
        else:
            phase_field_preset = json.loads(preset_path.read_text(encoding="utf-8"))
            print(f"Using phase-field preset: {preset_path}")

    # ============================================================
    # 4) Modele coque + diagnostics champs
    # ============================================================
    model = build_shell_model(domain, cell_tags, facets, cfg)
    _print_field_min_max("YoungModulus", model.E_field)
    _print_field_min_max("Thickness", model.thick_field)
    _print_field_min_max("GcFactorBandes", model.gc_factor_field)
    _print_field_min_max("RivetBandsMask", model.rivet_bands_mask_field)

    if bool(getattr(cfg, "write_local_frame_outputs", True)):
        with io.VTKFile(MPI.COMM_WORLD, output_layout["local_frame_file"], "w") as vtk:
            vtk.write_function(model.e1, 0.0)
            vtk.write_function(model.e2, 0.0)
            vtk.write_function(model.e3, 0.0)

        V0 = fem.functionspace(domain, ("DG", 0))
        material_regions = fem.Function(V0, name="MaterialRegion")
        material_regions.x.array[:] = float(cfg.shell_cell_tag)
        if cell_tags is not None and len(cell_tags.indices) > 0:
            material_regions.x.array[cell_tags.indices] = cell_tags.values.astype(float)
        with io.VTKFile(MPI.COMM_WORLD, local_frame_dir / "material_regions.pvd", "w") as vtk:
            vtk.write_function(material_regions, 0.0)

        with io.VTKFile(MPI.COMM_WORLD, local_frame_dir / "material_fields.pvd", "w") as vtk:
            vtk.write_function(model.E_field, 0.0)
            vtk.write_function(model.thick_field, 0.0)
            vtk.write_function(model.gc_factor_field, 0.0)
            vtk.write_function(model.rivet_bands_mask_field, 0.0)
            vtk.write_function(model.rivet_bands_mask_viz_field, 0.0)

    # ============================================================
    # 5) Metadonnees + calcul quasi-statique
    # ============================================================
    if MPI.COMM_WORLD.rank == 0:
        metadata = {
            "metadata_schema": "titanic-fem-run/v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": config_vers_dict(cfg),
            "phase_field_preset": phase_field_preset,
        }
        output_layout["metadata_file"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    run_quasi_static(model, cfg, output_layout, phase_field_preset=phase_field_preset)


run = lancer_calcul


if __name__ == "__main__":
    lancer_calcul()
