from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LABEL_FRAC_D95 = "part zone tres endommagee (d >= 0.95)"


@dataclass
class Suivi:
    source: Path
    label: str
    time: np.ndarray
    max_u: np.ndarray
    max_damage: np.ndarray
    mean_damage: np.ndarray
    frac_d95: np.ndarray
    temps_pas: np.ndarray
    temps_meca: np.ndarray
    temps_pf: np.ndarray


def _charger_monitor(path: Path, label: str) -> Suivi:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        raise ValueError(f"monitor.csv vide: {path}")

    def col(name: str) -> np.ndarray:
        return np.asarray([float(r.get(name, 0.0) or 0.0) for r in rows], dtype=float)

    return Suivi(
        source=path,
        label=label,
        time=col("time"),
        max_u=col("max_u_inf"),
        max_damage=col("max_damage"),
        mean_damage=col("mean_damage"),
        frac_d95=col("frac_damage_ge_095"),
        temps_pas=col("temps_pas_s"),
        temps_meca=col("temps_meca_s"),
        temps_pf=col("temps_phase_field_s"),
    )


def _find_case_monitor(results_root: Path, pattern: str) -> Path | None:
    candidats = sorted(results_root.glob(pattern))
    if not candidats:
        return None
    return max(candidats, key=lambda p: p.stat().st_mtime)


def _save(fig, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _grille_temps_commune(a: Suivi, b: Suivi, n: int = 600) -> np.ndarray:
    tmax = min(float(a.time.max()), float(b.time.max()))
    return np.linspace(0.0, tmax, n)


def _interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp)


def tracer_dommages(avec: Suivi, sans: Suivi, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    axes[0].plot(avec.time, avec.max_damage, linewidth=2.2, label=avec.label)
    axes[0].plot(sans.time, sans.max_damage, linewidth=2.2, label=sans.label)
    axes[0].set_title("max_damage")
    axes[0].set_xlabel("Temps")
    axes[0].set_ylabel("Dommage")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(avec.time, avec.mean_damage, linewidth=2.2, label=f"{avec.label} dommage moyen")
    axes[1].plot(sans.time, sans.mean_damage, linewidth=2.2, label=f"{sans.label} dommage moyen")
    axes[1].plot(avec.time, avec.frac_d95, "--", linewidth=2.0, label=f"{avec.label} {LABEL_FRAC_D95}")
    axes[1].plot(sans.time, sans.frac_d95, "--", linewidth=2.0, label=f"{sans.label} {LABEL_FRAC_D95}")
    axes[1].set_title("Dommage moyen et zone tres endommagee")
    axes[1].set_xlabel("Temps")
    axes[1].set_ylabel("Niveau dommage")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    _save(fig, outdir / "comparaison_dommages.png")


def tracer_ecarts_dommages(avec: Suivi, sans: Suivi, outdir: Path):
    grille = _grille_temps_commune(avec, sans)
    delta_max = _interp(grille, avec.time, avec.max_damage) - _interp(grille, sans.time, sans.max_damage)
    delta_mean = _interp(grille, avec.time, avec.mean_damage) - _interp(grille, sans.time, sans.mean_damage)
    delta_frac = _interp(grille, avec.time, avec.frac_d95) - _interp(grille, sans.time, sans.frac_d95)

    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.8), sharex=True)
    axes[0].plot(grille, delta_max, linewidth=2.0)
    axes[0].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    axes[0].set_ylabel("Delta max damage")
    axes[0].set_title("Ecarts temporels (avec - sans rivets)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(grille, delta_mean, linewidth=2.0)
    axes[1].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    axes[1].set_ylabel("Delta dommage moyen")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(grille, delta_frac, linewidth=2.0)
    axes[2].axhline(0.0, color="k", linewidth=1.0, alpha=0.5)
    axes[2].set_ylabel(f"Delta {LABEL_FRAC_D95}")
    axes[2].set_xlabel("Temps")
    axes[2].grid(True, alpha=0.3)

    _save(fig, outdir / "ecarts_temporels_dommages.png")


def tracer_deplacement(avec: Suivi, sans: Suivi, outdir: Path):
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(avec.time, avec.max_u, linewidth=2.2, label=avec.label)
    ax.plot(sans.time, sans.max_u, linewidth=2.2, label=sans.label)
    ax.set_title("max_u_inf (deplacement max)")
    ax.set_xlabel("Temps")
    ax.set_ylabel("Deplacement [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir / "comparaison_deplacement.png")


def tracer_trajectoire_u_d(avec: Suivi, sans: Suivi, outdir: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(avec.max_u, avec.max_damage, linewidth=2.2, label=avec.label)
    ax.plot(sans.max_u, sans.max_damage, linewidth=2.2, label=sans.label)
    ax.set_title("Trajectoire max_u vs max_damage")
    ax.set_xlabel("max_u_inf [m]")
    ax.set_ylabel("max_damage")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, outdir / "trajectoire_u_vs_damage.png")


def tracer_couts(avec: Suivi, sans: Suivi, outdir: Path):
    labels = ["temps_pas", "temps_meca", "temps_phase_field"]
    val_avec = [avec.temps_pas.sum(), avec.temps_meca.sum(), avec.temps_pf.sum()]
    val_sans = [sans.temps_pas.sum(), sans.temps_meca.sum(), sans.temps_pf.sum()]

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(x - width / 2, val_avec, width, label=avec.label)
    ax.bar(x + width / 2, val_sans, width, label=sans.label)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Temps cumule [s]")
    ax.set_title("Comparaison des couts de calcul")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    _save(fig, outdir / "comparaison_couts.png")


def ecrire_resume(avec: Suivi, sans: Suivi, outdir: Path):
    grille = _grille_temps_commune(avec, sans)
    dmean_avec = _interp(grille, avec.time, avec.mean_damage)
    dmean_sans = _interp(grille, sans.time, sans.mean_damage)
    auc_avec = float(np.trapezoid(dmean_avec, grille))
    auc_sans = float(np.trapezoid(dmean_sans, grille))

    lignes = []
    lignes.append(f"avec={avec.source}")
    lignes.append(f"sans={sans.source}")
    lignes.append("")
    lignes.append(f"n_steps_avec={len(avec.time)-1}")
    lignes.append(f"n_steps_sans={len(sans.time)-1}")
    lignes.append("")
    lignes.append(f"max_damage_final_avec={avec.max_damage[-1]:.6f}")
    lignes.append(f"max_damage_final_sans={sans.max_damage[-1]:.6f}")
    lignes.append(f"delta_max_damage_final={(avec.max_damage[-1]-sans.max_damage[-1]):+.6f}")
    lignes.append("")
    lignes.append(f"mean_damage_final_avec={avec.mean_damage[-1]:.6f}")
    lignes.append(f"mean_damage_final_sans={sans.mean_damage[-1]:.6f}")
    lignes.append(f"delta_mean_damage_final={(avec.mean_damage[-1]-sans.mean_damage[-1]):+.6f}")
    lignes.append("")
    lignes.append(f"frac_d95_final_avec={avec.frac_d95[-1]:.6f}")
    lignes.append(f"frac_d95_final_sans={sans.frac_d95[-1]:.6f}")
    lignes.append(f"delta_frac_d95_final={(avec.frac_d95[-1]-sans.frac_d95[-1]):+.6f}")
    lignes.append("")
    lignes.append(f"max_u_final_avec={avec.max_u[-1]:.6e}")
    lignes.append(f"max_u_final_sans={sans.max_u[-1]:.6e}")
    lignes.append(f"delta_max_u_final={(avec.max_u[-1]-sans.max_u[-1]):+.6e}")
    lignes.append("")
    lignes.append(f"temps_total_pas_avec={avec.temps_pas.sum():.3f}")
    lignes.append(f"temps_total_pas_sans={sans.temps_pas.sum():.3f}")
    lignes.append(f"delta_temps_total_pas={(avec.temps_pas.sum()-sans.temps_pas.sum()):+.3f}")
    lignes.append("")
    lignes.append(f"auc_mean_damage_avec={auc_avec:.6f}")
    lignes.append(f"auc_mean_damage_sans={auc_sans:.6f}")
    lignes.append(f"delta_auc_mean_damage={(auc_avec-auc_sans):+.6f}")

    (outdir / "resume_comparaison_rivets.txt").write_text("\n".join(lignes) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Graphes de comparaison avec vs sans rivets")
    parser.add_argument("--monitor-avec", type=str, default=None, help="monitor.csv du cas avec rivets")
    parser.add_argument("--monitor-sans", type=str, default=None, help="monitor.csv du cas sans rivets")
    parser.add_argument("--results-root", type=str, default="results", help="dossier racine des resultats")
    parser.add_argument("--outdir", type=str, default=None, help="dossier de sortie des graphes")
    return parser.parse_args()


def main():
    args = parse_args()
    results_root = Path(args.results_root)

    if args.monitor_avec:
        p_avec = Path(args.monitor_avec)
    else:
        p_avec = _find_case_monitor(results_root, "*with_rivets/quasi_static/monitor.csv")
    if args.monitor_sans:
        p_sans = Path(args.monitor_sans)
    else:
        p_sans = _find_case_monitor(results_root, "*without_rivets/quasi_static/monitor.csv")

    if p_avec is None or not p_avec.exists():
        raise FileNotFoundError("Impossible de trouver le monitor du cas AVEC rivets")
    if p_sans is None or not p_sans.exists():
        raise FileNotFoundError("Impossible de trouver le monitor du cas SANS rivets")

    avec = _charger_monitor(p_avec, "avec rivets")
    sans = _charger_monitor(p_sans, "sans rivets")

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = p_avec.parent.parent / "analyse_comparaison_rivets"
    outdir.mkdir(parents=True, exist_ok=True)

    tracer_dommages(avec, sans, outdir)
    tracer_ecarts_dommages(avec, sans, outdir)
    tracer_deplacement(avec, sans, outdir)
    tracer_trajectoire_u_d(avec, sans, outdir)
    tracer_couts(avec, sans, outdir)
    ecrire_resume(avec, sans, outdir)

    print(f"Graphes de comparaison ecrits dans: {outdir}")


if __name__ == "__main__":
    main()
