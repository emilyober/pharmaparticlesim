from pathlib import Path
import torch

from frame_data import FrameDataset
from memory_processor_model import MemoryProcessorModel
from rollout_eval import evaluate_rollouts

root = Path(__file__).resolve().parent.parent
runs = root / "experiments/ml_runs"

checkpoint = torch.load(
    runs / "processor_memory_data42_v1_latest.pt",
    map_location="cpu",
    weights_only=True,
)

control = torch.load(
    runs / "processor_data_42_best.pt",
    map_location="cpu",
    weights_only=True,
)

model = MemoryProcessorModel(
    masses=control["masses"],
    radii=control["radii"],
    **checkpoint["config"],
)
model.load_state_dict(checkpoint["model_state"])
model.integration_reference_dt = checkpoint["integration_reference_dt"]
model.eval()
model.early_gate = False

data = FrameDataset(
    root / "experiments/results/local_timestepping/"
    "pilot_seed43_10rpm_0p1s_dense.csv",
    horizon=control["horizon"],
    rollout_steps=2,
)

print("Checkpoint epoch:", checkpoint["epoch"])
print("Dataset: seed 43")
print("Both modes use the same weights and initial states.")

for reset in (False, True):
    label = "RESET every prediction" if reset else "RETAIN memory"
    print(f"\nMode: {label}")

    result = evaluate_rollouts(
        model,
        data,
        checkpoint["loss_scale"],
        starts=[2000, 6000, 10000, 14000, 18000],
        steps=50,
        use_memory=True,
        reset_memory_each_step=reset,
    )

    for report in result["windows"]:
        print(
            f"start={report['start_frame']}, "
            f"completed={report['completed']}/50, "
            f"failure={report['failure']}, "
            f"score={report['score']}, "
            f"peak overlaps µm="
            f"{[x * 1e6 for x in report['peak_overlap_m']]}"
        )

    print("Eligible:", result["eligible"])
    print("Overall score:", result["score"])