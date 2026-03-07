from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rivet.rivet import creer_config, lancer_calcul


def parse_args():
    parser = argparse.ArgumentParser(
        description="Version simple pour lancer le modele local rivet et ecrire les sorties dans vis_rivet/."
    )
    parser.add_argument("--steps", type=int, default=120, help="Nombre de pas de chargement.")
    parser.add_argument("--max-traction-mpa", type=float, default=300.0, help="Traction maximale en MPa.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="vis_rivet/rivet_titanic_AT1.bp",
        help="Dossier/fichier de sortie BP relatif au depot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    resultats_dossier = str(output_path.parent)
    export_filename = output_path.name

    cfg = creer_config(
        steps=int(args.steps),
        max_traction=float(args.max_traction_mpa) * 1e6,
        resultats_dossier=resultats_dossier,
        export_filename=export_filename,
    )
    result = lancer_calcul(cfg)
    print(f"Simulation terminee. Resume: {result['summary_path']}")


if __name__ == "__main__":
    main()
