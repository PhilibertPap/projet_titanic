from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rivet.rivet import creer_config as creer_config_rivet
from rivet.rivet import lancer_calcul as lancer_calcul_rivet
from rivet.rivet import creer_preset_bandes_grande_echelle


def _load_summary(path: str | Path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def _calibrate_factors_from_summary(
    summary: dict,
    *,
    reference_rupture_traction_pa: float,
    facteur_gc_min: float,
    facteur_gc_max: float,
    facteur_gc_exponent: float,
    facteur_e: float,
    facteur_epaisseur: float,
) -> dict:
    rupture_detected = bool(summary.get("rupture_detected", False))
    traction = summary.get("rupture_traction_pa")
    if traction is None:
        traction = summary.get("last_traction_pa", 0.0)
    traction = float(traction)

    # Heuristic bridge local -> global: use rupture traction ratio as a proxy for
    # local fracture resistance and map it to an effective Gc factor.
    ratio = traction / max(reference_rupture_traction_pa, 1.0)
    facteur_gc = float(np.clip(ratio ** facteur_gc_exponent, facteur_gc_min, facteur_gc_max))

    return {
        "rupture_detected": rupture_detected,
        "rupture_traction_pa": traction,
        "reference_rupture_traction_pa": float(reference_rupture_traction_pa),
        "ratio_rupture": float(ratio),
        "facteur_E": float(facteur_e),
        "facteur_epaisseur": float(facteur_epaisseur),
        "facteur_Gc": facteur_gc,
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrer un preset de bandes grande_echelle depuis le modele local rivet")
    parser.add_argument("--summary", type=str, default=None, help="Chemin vers run_summary.json (si deja calcule)")
    parser.add_argument("--run-local", action="store_true", help="Lance d'abord le calcul local rivet")
    parser.add_argument("--output", type=str, default="rivet/bandes_rivets_grande_echelle_calibre.json")
    parser.add_argument("--reference-rupture-mpa", type=float, default=250.0)
    parser.add_argument("--gc-min", type=float, default=0.5)
    parser.add_argument("--gc-max", type=float, default=1.2)
    parser.add_argument("--gc-exponent", type=float, default=2.0)
    parser.add_argument("--facteur-e", type=float, default=0.98)
    parser.add_argument("--facteur-epaisseur", type=float, default=1.0)
    parser.add_argument("--x-start", type=float, default=177.0)
    parser.add_argument("--x-end", type=float, default=268.0)
    parser.add_argument("--n-bandes", type=int, default=8)
    parser.add_argument("--largeur-x", type=float, default=0.30)
    parser.add_argument("--z-min", type=float, default=-10.2)
    parser.add_argument("--z-max", type=float, default=0.2)
    parser.add_argument("--local-steps", type=int, default=None)
    parser.add_argument("--local-max-traction-mpa", type=float, default=None)
    args = parser.parse_args()

    if args.run_local:
        local_updates = {}
        if args.local_steps is not None:
            local_updates["steps"] = int(args.local_steps)
        if args.local_max_traction_mpa is not None:
            local_updates["max_traction"] = float(args.local_max_traction_mpa) * 1e6
        result = lancer_calcul_rivet(creer_config_rivet(**local_updates)) if local_updates else lancer_calcul_rivet()
        summary = _load_summary(result["summary_path"])
    elif args.summary:
        summary = _load_summary(args.summary)
    else:
        raise SystemExit("Provide --summary or use --run-local")

    factors = _calibrate_factors_from_summary(
        summary,
        reference_rupture_traction_pa=args.reference_rupture_mpa * 1e6,
        facteur_gc_min=args.gc_min,
        facteur_gc_max=args.gc_max,
        facteur_gc_exponent=args.gc_exponent,
        facteur_e=args.facteur_e,
        facteur_epaisseur=args.facteur_epaisseur,
    )

    preset_path = creer_preset_bandes_grande_echelle(
        path=args.output,
        x_start_m=args.x_start,
        x_end_m=args.x_end,
        n_bandes_x=args.n_bandes,
        largeur_x_m=args.largeur_x,
        z_min_m=args.z_min,
        z_max_m=args.z_max,
        facteur_E=factors["facteur_E"],
        facteur_epaisseur=factors["facteur_epaisseur"],
        facteur_Gc=factors["facteur_Gc"],
    )

    calib_report = {
        "source_summary": summary,
        "calibrated_factors": factors,
        "preset_path": str(preset_path),
    }
    report_path = Path(preset_path).with_suffix(".calibration.json")
    report_path.write_text(json.dumps(calib_report, indent=2), encoding="utf-8")

    print("Calibration terminee")
    print(json.dumps(factors, indent=2))
    print(f"Preset bandes: {preset_path}")
    print(f"Rapport calibration: {report_path}")


if __name__ == "__main__":
    main()
