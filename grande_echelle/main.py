from __future__ import annotations

from datetime import datetime, timezone
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
    # Bandes de rivets "homogeneisees" (approximation grande echelle):
    # on place des bandes horizontales selon z (strakes/seams), autour de la zone immergee.
    # Valeurs par defaut prudentes et modifiables.
    bandes_rivets = []
    for z_centre in (-9.3, -8.1, -6.9, -5.7, -4.5, -3.3, -2.1, -0.9):
        bandes_rivets.append(
            {
                "nom": f"bande_z_{z_centre:+.1f}m".replace(".", "p"),
                "z_centre_m": z_centre,
                "largeur_m": 0.30,  # zone influencee (homogeneisee), pas juste le diametre du rivet
                "facteur_E": 0.98,
                "facteur_epaisseur": 1.00,
                "facteur_Gc": 0.88,
            }
        )

    return {
        "mesh_stem": "mesh/coque",
        "results_root": "results",
        "case_name": "titanic",
        "shell_cell_tag": 1,
        "rivet_cell_tag": 2,
        "shell_thickness": 2.5e-2,
        "shell_young_modulus": 210e9,
        "shell_poisson_ratio": 0.3,
        "rivet_thickness": 2.5e-2,
        "rivet_young_modulus": 190e9,
        "rivet_poisson_ratio": 0.3,
        "shell_yield_strength_pa": 270e6,
        "shell_ultimate_strength_pa": 431e6,
        "steel_sulfur_mass_fraction": 0.00065,
        "sigma": 3.0,
        "pressure_peak": 5e6,
        "iceberg_loading": "dirichlet_displacement",
        "iceberg_disp_peak": 5e-2,
        "iceberg_disp_sign": 1.0,
        "iceberg_patch_radius_factor": 3.0,
        "t_final": 1.0,
        "num_steps": 20,
        "ecrire_vtk_tous_les_n_pas": 1,
        "afficher_console_tous_les_n_pas": 1,
        "left_facet_tag": 1,
        "right_facet_tag": 2,
        "bottom_facet_tag": 3,
        "top_facet_tag": 4,
        "clamp_all_edges": True,
        "clamp_rotations": True,
        "iceberg_center_y": -10.8,
        "waterline_z": 0.0,
        "iceberg_depth_below_waterline": 7.5,
        "iceberg_moves_from_xmax_to_xmin": True,
        "y_mid_factor": 0.9,
        "z_mid_factor": 0.2,
        "phase_field_preset_file": "petite_echelle/results/local_phase_field_sweep/selected_preset.json",
        "enable_global_phase_field": True,
        "phase_field_use_selected_preset": True,
        "phase_field_gc_j_m2": 7000.0,
        "phase_field_l0_m": 0.4,
        "phase_field_residual_stiffness": 1e-6,
        "utiliser_bandes_rivets_z": True,
        "bandes_rivets_z": bandes_rivets,
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
    if "ecrire_vtk_tous_les_n_pas" not in data and "vtk_write_stride" in data:
        data["ecrire_vtk_tous_les_n_pas"] = data["vtk_write_stride"]
    if "afficher_console_tous_les_n_pas" not in data and "monitor_print_stride" in data:
        data["afficher_console_tous_les_n_pas"] = data["monitor_print_stride"]
    if data.get("mechanics_petsc_options") is None:
        data["mechanics_petsc_options"] = dict(data["petsc_options"])
    if data.get("damage_petsc_options") is None:
        data["damage_petsc_options"] = dict(data["petsc_options"])
    return data


def verifier_config(cfg) -> None:
    # Compatibilite si un ancien objet config est passe
    if not hasattr(cfg, "ecrire_vtk_tous_les_n_pas") and hasattr(cfg, "vtk_write_stride"):
        cfg.ecrire_vtk_tous_les_n_pas = cfg.vtk_write_stride
    if not hasattr(cfg, "afficher_console_tous_les_n_pas") and hasattr(cfg, "monitor_print_stride"):
        cfg.afficher_console_tous_les_n_pas = cfg.monitor_print_stride
    if not hasattr(cfg, "utiliser_bandes_rivets_z"):
        cfg.utiliser_bandes_rivets_z = False
    if not hasattr(cfg, "bandes_rivets_z"):
        cfg.bandes_rivets_z = []

    if cfg.num_steps <= 0:
        raise ValueError("num_steps must be > 0")
    if cfg.t_final <= 0.0:
        raise ValueError("t_final must be > 0")
    if cfg.ecrire_vtk_tous_les_n_pas <= 0:
        raise ValueError("ecrire_vtk_tous_les_n_pas must be >= 1")
    if cfg.afficher_console_tous_les_n_pas <= 0:
        raise ValueError("afficher_console_tous_les_n_pas must be >= 1")
    if cfg.iceberg_loading not in {"neumann_pressure", "dirichlet_displacement"}:
        raise ValueError("iceberg_loading must be 'neumann_pressure' or 'dirichlet_displacement'")


DEFAULT_CONFIG = creer_config()


def config_apercu_rapide():
    return creer_config(
        case_name="preview_paraview_ultra_fast",
        num_steps=8,
        t_final=0.8,
        ecrire_vtk_tous_les_n_pas=2,
        iceberg_loading="neumann_pressure",
        pressure_peak=2e5,
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
    """
    Preset de travail rapide:
    - moins d'ecritures VTK
    - moins d'affichage console
    - moins de pas de temps
    """
    base = creer_config(
        case_name="titanic_rivets_rapide",
        num_steps=12,
        ecrire_vtk_tous_les_n_pas=3,
        afficher_console_tous_les_n_pas=2,
        # Exemple de pas adaptes simples (plus denses au milieu de l'impact)
        temps_relatifs=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.58, 0.66, 0.74, 0.82, 0.9, 1.0],
    )
    return config_etude_rivets(with_rivets=with_rivets, base=base)


def config_etude_rivets_production(with_rivets: bool = True):
    """
    Preset plus propre pour comparaison finale:
    - plus de pas
    - ecriture VTK a chaque pas
    - solveurs separes meca / phase-field
    """
    base = creer_config(
        case_name="titanic_rivets_production",
        num_steps=36,
        ecrire_vtk_tous_les_n_pas=1,
        afficher_console_tous_les_n_pas=3,
        # Plus de points autour de la zone centrale de chargement
        temps_relatifs=[
            0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
            0.50, 0.54, 0.58, 0.62, 0.66, 0.70, 0.74, 0.78, 0.82, 0.86,
            0.90, 0.94, 0.97, 1.0
        ],
        # A ajuster selon votre machine / installation PETSc
        mechanics_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        damage_petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
    )
    return config_etude_rivets(with_rivets=with_rivets, base=base)


def lire_maillage(mesh_stem: str, comm=MPI.COMM_WORLD):
    candidate = Path(mesh_stem)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate.with_suffix(".msh"))
    else:
        candidates.append((Path.cwd() / candidate).with_suffix(".msh"))
        candidates.append((Path(__file__).resolve().parent / candidate).with_suffix(".msh"))
    mesh_path = next((p for p in candidates if p.exists()), None)
    if mesh_path is None:
        raise FileNotFoundError(f"Mesh file not found. Tried: {', '.join(str(p) for p in candidates)}")
    return io.gmsh.read_from_msh(str(mesh_path), comm)


def preparer_dossiers_resultats(results_root: str | Path, case_name: str) -> dict[str, Path]:
    base_dir = Path(results_root) / case_name
    local_frame_dir = base_dir / "local_frame"
    quasi_static_dir = base_dir / "quasi_static"
    local_frame_dir.mkdir(exist_ok=True, parents=True)
    quasi_static_dir.mkdir(exist_ok=True, parents=True)
    return {
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


def ecrire_base_locale(local_frame_file: Path, e1, e2, e3) -> None:
    with io.VTKFile(MPI.COMM_WORLD, local_frame_file, "w") as vtk:
        vtk.write_function(e1, 0.0)
        vtk.write_function(e2, 0.0)
        vtk.write_function(e3, 0.0)


def ecrire_zones_materiaux(material_file: Path, domain, cell_tags, default_tag: int) -> None:
    V0 = fem.functionspace(domain, ("DG", 0))
    material_regions = fem.Function(V0, name="MaterialRegion")
    material_regions.x.array[:] = float(default_tag)
    if cell_tags is not None and len(cell_tags.indices) > 0:
        material_regions.x.array[cell_tags.indices] = cell_tags.values.astype(float)
    with io.VTKFile(MPI.COMM_WORLD, material_file, "w") as vtk:
        vtk.write_function(material_regions, 0.0)


def lire_preset_phase_field(cfg) -> tuple[dict | None, Path | None, list[Path]]:
    preset_cfg_path = Path(cfg.phase_field_preset_file)
    if preset_cfg_path.is_absolute():
        candidates = [preset_cfg_path]
    else:
        candidates = [Path.cwd() / preset_cfg_path, Path(__file__).resolve().parent / preset_cfg_path]
    preset_path = next((p for p in candidates if p.exists()), None)
    if preset_path is None:
        return None, None, candidates
    return json.loads(preset_path.read_text(encoding="utf-8")), preset_path, candidates


def afficher_infos_maillage(domain, cell_tags, facets) -> None:
    cell_values = [] if cell_tags is None else [int(v) for v in np.unique(cell_tags.values)]
    facet_values = [] if facets is None else [int(v) for v in np.unique(facets.values)]
    print(f"Mesh diagnostics: gdim={domain.geometry.dim}, tdim={domain.topology.dim}")
    print(f"Cell tags present: {cell_tags is not None}; values={cell_values}")
    print(f"Facet tags present: {facets is not None}; values={facet_values}")


def ecrire_metadonnees_run(metadata_file: Path, cfg, phase_field_preset: dict | None) -> None:
    if MPI.COMM_WORLD.rank != 0:
        return
    metadata = {
        "metadata_schema": "titanic-fem-run/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_vers_dict(cfg),
        "phase_field_preset": phase_field_preset,
    }
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def lancer_calcul(cfg=None):
    cfg = DEFAULT_CONFIG if cfg is None else cfg
    verifier_config(cfg)

    meshdata = lire_maillage(cfg.mesh_stem)
    domain = meshdata.mesh
    cell_tags = meshdata.cell_tags
    facets = meshdata.facet_tags
    afficher_infos_maillage(domain, cell_tags, facets)

    model = build_shell_model(domain, cell_tags, facets, cfg)
    output_layout = preparer_dossiers_resultats(cfg.results_root, cfg.case_name)
    phase_field_preset, preset_path, preset_candidates = lire_preset_phase_field(cfg)
    if preset_path is not None:
        print(f"Using phase-field preset: {preset_path}")
    else:
        print(f"Phase-field preset not found. Tried: {preset_candidates}")

    ecrire_base_locale(output_layout["local_frame_file"], model.e1, model.e2, model.e3)
    ecrire_zones_materiaux(
        output_layout["local_frame_dir"] / "material_regions.pvd",
        domain,
        cell_tags,
        cfg.shell_cell_tag,
    )
    ecrire_metadonnees_run(output_layout["metadata_file"], cfg, phase_field_preset)
    run_quasi_static(model, cfg, output_layout, phase_field_preset=phase_field_preset)


# Alias simple pour compatibilite avec les anciens appels.
run = lancer_calcul


if __name__ == "__main__":
    lancer_calcul()
