from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grande_echelle.config import SimulationConfig
from grande_echelle.main import run


def main():
    cfg = SimulationConfig(
        case_name="preview_paraview_ultra_fast",
        num_steps=1,
        t_final=0.1,
        iceberg_loading="neumann_pressure",
        pressure_peak=2e5,
        sigma=2.5,
        enable_global_phase_field=False,
        phase_field_use_selected_preset=False,
        phase_field_gc_j_m2=7000.0,
        phase_field_l0_m=0.4,
    )
    print("Running ultra-fast ParaView preview (Neumann load, no global phase-field, 1 step)...")
    run(cfg)


if __name__ == "__main__":
    main()
