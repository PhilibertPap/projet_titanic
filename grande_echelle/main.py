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
    from .shell import construire_modele_coque
    from .quasi_static import executer_quasi_statique
except ImportError:  # pragma: no cover - script execution fallback
    from shell import construire_modele_coque
    from quasi_static import executer_quasi_statique


# ============================================================================
# PARAMETRES PAR DEFAUT
# ============================================================================

def _bandes_rivets_par_defaut(x_debut: float, x_fin: float) -> list[dict]:
    n_bandes = 10
    largeur_x = 0.35
    x_margin = 0.5 * largeur_x
    x_centres = np.linspace(x_debut + x_margin, x_fin - x_margin, n_bandes)

    bandes = []
    for i_bande, x_centre in enumerate(x_centres, start=1):
        bandes.append(
            {
                "nom": f"bande_x_{i_bande:02d}",
                "x_centre_m": float(x_centre),
                "largeur_x_m": largeur_x,
                "z_min_m": -10.2,
                "z_max_m": 0.2,
                "facteur_E": 0.98,
                "facteur_epaisseur": 1.00,
                "facteur_Gc": 0.88,
            }
        )
    return bandes


def config_par_defaut() -> dict:
    iceberg_zone_x_debut_m = 177.0
    iceberg_zone_x_fin_m = 268.0
    rivet_zone_x_debut_m = 177.0
    rivet_zone_x_fin_m = 252.0

    return {
        # Maillage / sortie
        "fichier_maillage": "mesh/coque",
        "dossier_resultats": "results",
        "nom_cas": "titanic",
        "tag_cellule_coque": 1,
        "tag_cellule_rivet": 2,
        # Materiaux
        "epaisseur_coque": 2.5e-2,
        "young_coque": 210e9,
        "poisson_coque": 0.3,
        "epaisseur_rivet": 2.5e-2,
        "young_rivet": 190e9,
        "poisson_rivet": 0.3,
        # Chargement iceberg
        "sigma": 3.0,
        "rampe_amplitude_iceberg": True,
        "deplacement_pic_iceberg": 2.0e-2,
        "signe_deplacement_iceberg": 1.0,
        "facteur_rayon_patch_iceberg": 3.0,
        # Temps / monitoring
        "temps_final": 8.5,
        "nombre_pas": 96,
        "temps_relatifs": None,
        "vtk_tous_les_n_pas": 8,
        "console_tous_les_n_pas": 4,
        "ecrire_sorties_base_locale": False,
        "ecrire_vtk_rotation": False,
        "ecrire_vtk_endommagement": False,
        "ecrire_vtk_endommagement_si_desactive": False,
        # CL tags
        "tag_facet_gauche": 1,
        "tag_facet_droite": 2,
        "tag_facet_bas": 3,
        "tag_facet_haut": 4,
        "encastrer_tous_bords": True,
        "encastrer_rotations": True,
        # Trajectoire iceberg
        "iceberg_centre_y": -10.8,
        "iceberg_zone_x_debut_m": iceberg_zone_x_debut_m,
        "iceberg_zone_x_fin_m": iceberg_zone_x_fin_m,
        "flottaison_z": 0.0,
        "iceberg_hauteur_au_dessus_fond_m": 3.0,
        "iceberg_profondeur_sous_flottaison_m": 7.5,
        "iceberg_de_xmax_vers_xmin": True,
        "iceberg_dx_max_par_pas_m": None,
        "iceberg_contact_t_debut": 0.45,
        "iceberg_contact_t_fin": 8.0,
        # Phase-field
        "fichier_preset_phase_field": None,
        "activer_phase_field_global": True,
        "phase_field_utiliser_preset_selectionne": False,
        "phase_field_gc_j_m2": 7000.0,
        "phase_field_l0_m": 0.20,
        "phase_field_raideur_residuelle": 1e-6,
        "phase_field_scinder_traction_compression": True,
        "phase_field_seuil_nucleation_j_m3": 8.0e4,
        "phase_field_mise_a_jour_tous_les_n_pas": 2,
        "phase_field_utiliser_snes_vi": False,
        "phase_field_nb_iters_alternance": 6,
        "phase_field_nb_iters_min_alternance": 1,
        "phase_field_tol_alternance": 1e-4,
        "phase_field_snes_rtol": 1e-9,
        "phase_field_snes_atol": 1e-9,
        "phase_field_snes_max_it": 50,
        "uniformiser_facteur_gc_bandes": True,
        # Bandes rivets homogenisees
        "utiliser_bandes_rivets_z": True,
        "bandes_rivets_z": _bandes_rivets_par_defaut(rivet_zone_x_debut_m, rivet_zone_x_fin_m),
        "fichier_preset_bandes_rivets": None,
        "bandes_rivets_x_min_m": rivet_zone_x_debut_m,
        "bandes_rivets_x_max_m": rivet_zone_x_fin_m,
        # Solveurs PETSc
        "options_petsc_mecanique": None,
        "options_petsc_endommagement": None,
        "options_petsc": {"ksp_type": "preonly", "pc_type": "lu"},
    }


ALIASES_CONFIG_ANCIEN_VERS_NOUVEAU = {
    "mesh_stem": "fichier_maillage",
    "results_root": "dossier_resultats",
    "case_name": "nom_cas",
    "shell_cell_tag": "tag_cellule_coque",
    "rivet_cell_tag": "tag_cellule_rivet",
    "shell_thickness": "epaisseur_coque",
    "shell_young_modulus": "young_coque",
    "shell_poisson_ratio": "poisson_coque",
    "rivet_thickness": "epaisseur_rivet",
    "rivet_young_modulus": "young_rivet",
    "rivet_poisson_ratio": "poisson_rivet",
    "ramp_amplitude_iceberg": "rampe_amplitude_iceberg",
    "iceberg_disp_peak": "deplacement_pic_iceberg",
    "iceberg_disp_sign": "signe_deplacement_iceberg",
    "iceberg_patch_radius_factor": "facteur_rayon_patch_iceberg",
    "num_steps": "nombre_pas",
    "t_final": "temps_final",
    "ecrire_vtk_tous_les_n_pas": "vtk_tous_les_n_pas",
    "afficher_console_tous_les_n_pas": "console_tous_les_n_pas",
    "write_local_frame_outputs": "ecrire_sorties_base_locale",
    "write_rotation_vtk": "ecrire_vtk_rotation",
    "write_damage_vtk": "ecrire_vtk_endommagement",
    "write_damage_vtk_if_disabled": "ecrire_vtk_endommagement_si_desactive",
    "left_facet_tag": "tag_facet_gauche",
    "right_facet_tag": "tag_facet_droite",
    "bottom_facet_tag": "tag_facet_bas",
    "top_facet_tag": "tag_facet_haut",
    "clamp_all_edges": "encastrer_tous_bords",
    "clamp_rotations": "encastrer_rotations",
    "iceberg_center_y": "iceberg_centre_y",
    "waterline_z": "flottaison_z",
    "iceberg_height_above_bottom_m": "iceberg_hauteur_au_dessus_fond_m",
    "iceberg_depth_below_waterline": "iceberg_profondeur_sous_flottaison_m",
    "iceberg_moves_from_xmax_to_xmin": "iceberg_de_xmax_vers_xmin",
    "iceberg_max_dx_par_pas_m": "iceberg_dx_max_par_pas_m",
    "iceberg_contact_t_start": "iceberg_contact_t_debut",
    "iceberg_contact_t_end": "iceberg_contact_t_fin",
    "phase_field_preset_file": "fichier_preset_phase_field",
    "enable_global_phase_field": "activer_phase_field_global",
    "phase_field_use_selected_preset": "phase_field_utiliser_preset_selectionne",
    "phase_field_residual_stiffness": "phase_field_raideur_residuelle",
    "phase_field_split_traction_compression": "phase_field_scinder_traction_compression",
    "phase_field_use_snes_vi": "phase_field_utiliser_snes_vi",
    "phase_field_n_alt_iters": "phase_field_nb_iters_alternance",
    "phase_field_alt_min_iters": "phase_field_nb_iters_min_alternance",
    "phase_field_alt_tol": "phase_field_tol_alternance",
    "rivet_bandes_preset_file": "fichier_preset_bandes_rivets",
    "rivet_bandes_x_min_m": "bandes_rivets_x_min_m",
    "rivet_bandes_x_max_m": "bandes_rivets_x_max_m",
    "mechanics_petsc_options": "options_petsc_mecanique",
    "damage_petsc_options": "options_petsc_endommagement",
    "petsc_options": "options_petsc",
}


def _appliquer_aliases_config_data(data: dict) -> None:
    for ancien, nouveau in ALIASES_CONFIG_ANCIEN_VERS_NOUVEAU.items():
        if ancien in data and nouveau not in data:
            data[nouveau] = data[ancien]


def _appliquer_aliases_config_objet(cfg) -> None:
    for ancien, nouveau in ALIASES_CONFIG_ANCIEN_VERS_NOUVEAU.items():
        if hasattr(cfg, ancien) and not hasattr(cfg, nouveau):
            setattr(cfg, nouveau, getattr(cfg, ancien))


def creer_config(**updates):
    data = config_par_defaut()
    data.update(updates)
    _appliquer_aliases_config_data(data)
    return SimpleNamespace(**data)


def config_vers_dict(cfg) -> dict:
    data = dict(vars(cfg))
    if data.get("options_petsc_mecanique") is None:
        data["options_petsc_mecanique"] = dict(data["options_petsc"])
    if data.get("options_petsc_endommagement") is None:
        data["options_petsc_endommagement"] = dict(data["options_petsc"])
    return data


OPTIONAL_CONFIG_DEFAULTS = {
    "utiliser_bandes_rivets_z": False,
    "bandes_rivets_z": [],
    "fichier_preset_bandes_rivets": None,
    "bandes_rivets_x_min_m": None,
    "bandes_rivets_x_max_m": None,
    "iceberg_dx_max_par_pas_m": None,
    "iceberg_contact_t_debut": 0.0,
    "iceberg_contact_t_fin": None,
    "iceberg_hauteur_au_dessus_fond_m": None,
    "ecrire_sorties_base_locale": True,
    "ecrire_vtk_rotation": True,
    "ecrire_vtk_endommagement": True,
    "ecrire_vtk_endommagement_si_desactive": False,
    "phase_field_utiliser_snes_vi": False,
    "phase_field_nb_iters_alternance": 6,
    "phase_field_nb_iters_min_alternance": 1,
    "phase_field_tol_alternance": 1e-4,
    "phase_field_snes_rtol": 1e-9,
    "phase_field_snes_atol": 1e-9,
    "phase_field_snes_max_it": 50,
    "uniformiser_facteur_gc_bandes": True,
    "temps_relatifs": None,
}


def _appliquer_defaults_optionnels(cfg) -> None:
    for key, value in OPTIONAL_CONFIG_DEFAULTS.items():
        if hasattr(cfg, key):
            continue
        if key == "iceberg_contact_t_fin" and value is None:
            setattr(cfg, key, cfg.temps_final)
        else:
            setattr(cfg, key, value)


def verifier_config(cfg) -> None:
    _appliquer_aliases_config_objet(cfg)
    _appliquer_defaults_optionnels(cfg)

    if cfg.nombre_pas <= 0:
        raise ValueError("nombre_pas must be > 0")
    if cfg.temps_final <= 0.0:
        raise ValueError("temps_final must be > 0")
    if cfg.vtk_tous_les_n_pas <= 0:
        raise ValueError("vtk_tous_les_n_pas must be >= 1")
    if cfg.console_tous_les_n_pas <= 0:
        raise ValueError("console_tous_les_n_pas must be >= 1")

    for name in ("iceberg_centre_y", "flottaison_z"):
        if not hasattr(cfg, name):
            raise ValueError(f"Missing required config field: {name}")

    has_height = cfg.iceberg_hauteur_au_dessus_fond_m is not None
    has_depth = hasattr(cfg, "iceberg_profondeur_sous_flottaison_m")
    if not (has_height or has_depth):
        raise ValueError(
            "Missing required vertical iceberg position: set "
            "iceberg_hauteur_au_dessus_fond_m or iceberg_profondeur_sous_flottaison_m"
        )


DEFAULT_CONFIG = creer_config()


# ============================================================================
# CONFIGS D'ETUDES
# ============================================================================

def config_apercu_rapide():
    return creer_config(
        nom_cas="preview_paraview_ultra_fast",
        nombre_pas=8,
        temps_final=0.8,
        vtk_tous_les_n_pas=1,
        sigma=2.5,
    )


def config_etude_rivets(with_rivets: bool, base=None):
    base = DEFAULT_CONFIG if base is None else base
    data = config_vers_dict(base)
    data["nom_cas"] = f"{base.nom_cas}_{'with' if with_rivets else 'without'}_rivets"

    if not with_rivets:
        data["young_rivet"] = data["young_coque"]
        data["poisson_rivet"] = data["poisson_coque"]
        data["epaisseur_rivet"] = data["epaisseur_coque"]
        data["utiliser_bandes_rivets_z"] = False

    return SimpleNamespace(**data)


def config_etude_rivets_rapide(with_rivets: bool = True):
    base = creer_config(
        nom_cas="rivets_rapide",
        nombre_pas=14,
        vtk_tous_les_n_pas=6,
        console_tous_les_n_pas=2,
        deplacement_pic_iceberg=2.5e-2,
        temps_relatifs=[
            0.0, 0.08, 0.16, 0.24, 0.32, 0.40, 0.48,
            0.56, 0.64, 0.72, 0.80, 0.88, 0.95, 1.0,
        ],
        ecrire_sorties_base_locale=False,
        ecrire_vtk_rotation=False,
        ecrire_vtk_endommagement=False,
        phase_field_l0_m=0.20,
        phase_field_mise_a_jour_tous_les_n_pas=2,
        iceberg_dx_max_par_pas_m=2.0,
    )
    return config_etude_rivets(with_rivets=with_rivets, base=base)


def config_etude_rivets_production(with_rivets: bool = True):
    base = creer_config(
        nom_cas="titanic_rivets_production",
        nombre_pas=24,
        vtk_tous_les_n_pas=8,
        console_tous_les_n_pas=4,
        deplacement_pic_iceberg=2.5e-2,
        temps_relatifs=[
            0.0, 0.06, 0.12, 0.18, 0.24, 0.30, 0.36, 0.42, 0.48, 0.54,
            0.60, 0.66, 0.72, 0.78, 0.84, 0.89, 0.93, 0.96, 0.98, 1.0,
        ],
        ecrire_sorties_base_locale=False,
        ecrire_vtk_rotation=False,
        ecrire_vtk_endommagement=False,
        phase_field_l0_m=0.18,
        phase_field_mise_a_jour_tous_les_n_pas=2,
        options_petsc_mecanique={"ksp_type": "preonly", "pc_type": "lu"},
        options_petsc_endommagement={"ksp_type": "preonly", "pc_type": "lu"},
        iceberg_dx_max_par_pas_m=1.2,
    )
    return config_etude_rivets(with_rivets=with_rivets, base=base)


def config_etude_rivets_screening(with_rivets: bool = True):
    base = creer_config(
        nom_cas="rivets_screening",
        nombre_pas=12,
        vtk_tous_les_n_pas=8,
        console_tous_les_n_pas=4,
        ecrire_sorties_base_locale=False,
        ecrire_vtk_rotation=False,
        ecrire_vtk_endommagement=False,
        phase_field_l0_m=0.22,
        phase_field_mise_a_jour_tous_les_n_pas=3,
        iceberg_dx_max_par_pas_m=4.0,
        deplacement_pic_iceberg=2.5e-2,
    )
    return config_etude_rivets(with_rivets=with_rivets, base=base)


def _lancer_comparaison_rivets(build_config, suffixe_console: str, message_fin: str | None = None):
    print(f"=== Cas 1 : avec effet des rivets{suffixe_console} ===")
    lancer_calcul(build_config(with_rivets=True))
    print(f"=== Cas 2 : sans effet des rivets{suffixe_console} ===")
    lancer_calcul(build_config(with_rivets=False))
    print("Comparaison terminee.")
    if message_fin is None:
        message_fin = "Comparer les fichiers monitor.csv et les champs de dommage dans results/."
    print(message_fin)


def lancer_comparaison_rivets_rapide():
    _lancer_comparaison_rivets(config_etude_rivets_rapide, "")


def lancer_comparaison_rivets_production():
    _lancer_comparaison_rivets(config_etude_rivets_production, "")


def lancer_comparaison_rivets_screening():
    _lancer_comparaison_rivets(
        config_etude_rivets_screening,
        " (screening)",
        "Regarder d'abord monitor.csv, puis relancer en mode rapide/production si besoin.",
    )


# ============================================================================
# OUTILS I/O
# ============================================================================

def _resoudre_fichier_existant(path_like: str | Path, search_roots: list[Path], suffix: str | None = None) -> Path | None:
    candidate = Path(path_like)
    suffix_path = candidate.with_suffix(suffix) if suffix is not None else candidate
    if candidate.is_absolute():
        return suffix_path if suffix_path.exists() else None

    for root in search_roots:
        p = root / suffix_path
        if p.exists():
            return p
    return None


def _charger_bandes_rivets_preset_si_disponible(cfg) -> None:
    if not bool(cfg.utiliser_bandes_rivets_z):
        cfg.bandes_rivets_z = []
        if MPI.COMM_WORLD.rank == 0:
            print("Rivet bands disabled by config (no rivet preset loaded).")
        return

    preset_file = cfg.fichier_preset_bandes_rivets
    if not preset_file:
        auto_rel = Path("rivet/bandes_rivets_grande_echelle_calibre.json")
        auto_search = [Path.cwd(), Path(__file__).resolve().parent.parent]
        auto_path = _resoudre_fichier_existant(auto_rel, auto_search)
        if auto_path is None:
            return
        preset_file = str(auto_path)
        if MPI.COMM_WORLD.rank == 0:
            print(f"Auto-using calibrated rivet bands preset: {auto_path}")

    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    preset_path = _resoudre_fichier_existant(preset_file, search_roots)
    if preset_path is None:
        print(f"Rivet preset not found: {preset_file}")
        return

    data = json.loads(preset_path.read_text(encoding="utf-8"))
    bandes = data.get("bandes_rivets_z")
    if not isinstance(bandes, list):
        raise ValueError(
            f"Invalid rivet preset format in {preset_path}: expected key 'bandes_rivets_z' (list)"
        )

    if cfg.bandes_rivets_x_min_m is not None or cfg.bandes_rivets_x_max_m is not None:
        bandes_filtrees = []
        for bande in bandes:
            x_c = bande.get("x_centre_m")
            if x_c is None:
                bandes_filtrees.append(bande)
                continue
            x_val = float(x_c)
            if cfg.bandes_rivets_x_min_m is not None and x_val < float(cfg.bandes_rivets_x_min_m):
                continue
            if cfg.bandes_rivets_x_max_m is not None and x_val > float(cfg.bandes_rivets_x_max_m):
                continue
            bandes_filtrees.append(bande)

        if MPI.COMM_WORLD.rank == 0 and len(bandes_filtrees) != len(bandes):
            print(f"Filtered rivet bands by x-range: kept {len(bandes_filtrees)}/{len(bandes)}")
        bandes = bandes_filtrees

    gc_global = float(cfg.phase_field_gc_j_m2)
    for bande in bandes:
        gc_bande = bande.get("Gc_J_m2")
        if isinstance(gc_bande, (int, float)):
            bande["facteur_Gc"] = float(gc_bande) / max(gc_global, 1e-12)

    cfg.bandes_rivets_z = bandes
    cfg.utiliser_bandes_rivets_z = True
    cfg.fichier_preset_bandes_rivets = str(preset_path)
    print(f"Using rivet bands preset: {preset_path}")


def _charger_maillage(cfg):
    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    mesh_path = _resoudre_fichier_existant(cfg.fichier_maillage, search_roots, suffix=".msh")
    if mesh_path is None:
        tried = [str(root / Path(cfg.fichier_maillage).with_suffix('.msh')) for root in search_roots]
        raise FileNotFoundError(f"Mesh file not found. Tried: {', '.join(tried)}")

    meshdata = io.gmsh.read_from_msh(str(mesh_path), MPI.COMM_WORLD)
    return meshdata


def _afficher_diagnostics_maillage(meshdata) -> None:
    domain = meshdata.mesh
    cell_tags = meshdata.cell_tags
    facets = meshdata.facet_tags
    cell_values = [] if cell_tags is None else [int(v) for v in np.unique(cell_tags.values)]
    facet_values = [] if facets is None else [int(v) for v in np.unique(facets.values)]

    print(f"Mesh diagnostics: gdim={domain.geometry.dim}, tdim={domain.topology.dim}")
    print(f"Cell tags present: {cell_tags is not None}; values={cell_values}")
    print(f"Facet tags present: {facets is not None}; values={facet_values}")


def _construire_plan_sorties(cfg) -> dict:
    base_dir = Path(cfg.dossier_resultats) / cfg.nom_cas
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


def _charger_phase_field_preset(cfg):
    if not cfg.fichier_preset_phase_field:
        print("No phase-field preset file configured (using config values for Gc/l0).")
        return None

    search_roots = [Path.cwd(), Path(__file__).resolve().parent]
    preset_path = _resoudre_fichier_existant(cfg.fichier_preset_phase_field, search_roots)
    if preset_path is None:
        print(f"Phase-field preset not found: {cfg.fichier_preset_phase_field}")
        return None

    print(f"Using phase-field preset: {preset_path}")
    return json.loads(preset_path.read_text(encoding="utf-8"))


def _afficher_min_max_champ(name: str, field) -> None:
    local_min = float(np.min(field.x.array)) if field.x.array.size else np.inf
    local_max = float(np.max(field.x.array)) if field.x.array.size else -np.inf
    gmin = field.function_space.mesh.comm.allreduce(local_min, op=MPI.MIN)
    gmax = field.function_space.mesh.comm.allreduce(local_max, op=MPI.MAX)
    if field.function_space.mesh.comm.rank == 0:
        print(f"{name}: min={gmin:.6g}, max={gmax:.6g}")


def _ecrire_sorties_local_frame(domain, cell_tags, model, output_layout, cfg) -> None:
    if not cfg.ecrire_sorties_base_locale:
        return

    local_frame_dir = output_layout["local_frame_dir"]

    with io.VTKFile(MPI.COMM_WORLD, output_layout["local_frame_file"], "w") as vtk:
        vtk.write_function(model.e1, 0.0)
        vtk.write_function(model.e2, 0.0)
        vtk.write_function(model.e3, 0.0)

    V0 = fem.functionspace(domain, ("DG", 0))
    material_regions = fem.Function(V0, name="MaterialRegion")
    material_regions.x.array[:] = float(cfg.tag_cellule_coque)
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


def _ecrire_metadonnees(output_layout, cfg, phase_field_preset) -> None:
    if MPI.COMM_WORLD.rank != 0:
        return

    metadata = {
        "metadata_schema": "titanic-fem-run/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_vers_dict(cfg),
        "phase_field_preset": phase_field_preset,
    }
    output_layout["metadata_file"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")


# ============================================================================
# POST-TRAITEMENT
# ============================================================================

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
        "fraction_other": (
            (total_step_s - total_mech_s - total_pf_s) / total_step_s
            if total_step_s > 0
            else None
        ),
        "last_max_damage": float(rows[-1]["max_damage"]),
        "last_mean_damage": float(rows[-1]["mean_damage"]),
    }


# ============================================================================
# MAIN PIPELINE (style TD)
# ============================================================================

def lancer_calcul(cfg=None):
    cfg = DEFAULT_CONFIG if cfg is None else cfg
    verifier_config(cfg)
    _charger_bandes_rivets_preset_si_disponible(cfg)

    meshdata = _charger_maillage(cfg)
    _afficher_diagnostics_maillage(meshdata)

    domain = meshdata.mesh
    cell_tags = meshdata.cell_tags
    facets = meshdata.facet_tags
    output_layout = _construire_plan_sorties(cfg)
    phase_field_preset = _charger_phase_field_preset(cfg)

    model = construire_modele_coque(domain, cell_tags, facets, cfg)
    _afficher_min_max_champ("YoungModulus", model.E_field)
    _afficher_min_max_champ("Thickness", model.thick_field)
    _afficher_min_max_champ("GcFactorBandes", model.gc_factor_field)
    _afficher_min_max_champ("RivetBandsMask", model.rivet_bands_mask_field)

    _ecrire_sorties_local_frame(domain, cell_tags, model, output_layout, cfg)
    _ecrire_metadonnees(output_layout, cfg, phase_field_preset)

    executer_quasi_statique(model, cfg, output_layout, phase_field_preset=phase_field_preset)


run = lancer_calcul


if __name__ == "__main__":
    lancer_calcul()
