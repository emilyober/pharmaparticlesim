from pathlib import Path
import torch

from frame_data import FrameDataset
from processor_model import ProcessorModel
from rollout_eval import maximum_overlaps, reference_overlap_peak

root = Path(__file__).resolve().parent.parent
checkpoint = torch.load(
    root / "experiments/ml_runs/processor_surface_slip_gated_best.pt",
    map_location="cpu",
    weights_only=True,
)

model = ProcessorModel(
    masses=checkpoint["masses"],
    radii=checkpoint["radii"],
    **checkpoint["config"],
)
model.load_state_dict(checkpoint["model_state"])
model.eval()
data = FrameDataset(
    root / "experiments/results/local_timestepping/"
    "pilot_seed44_10rpm_0p1s_dense.csv",
    horizon=checkpoint["horizon"],
    rollout_steps=2,
)

horizon = data.horizon
dt = checkpoint["config"]["step_dt"]
gravity = torch.tensor([0.0, 0.0, -9.81])
starts = checkpoint["rollout_starts"]

print("Checkpoint epoch:", checkpoint["epoch"])
print("Prediction interval:", dt)
print("RMSE order: position [m], velocity [m/s], spin [rad/s]")


def rmse(state, truth):
    return (
        (state.double() - truth.double())
        .square()
        .reshape(-1, 3, 3)
        .mean(dim=(0, 2))
        .sqrt()
        .tolist()
    )


with torch.no_grad():
    for start in starts:
        if start + 200 * horizon >= data.num_frames:
            raise ValueError(f"Not enough reference data at {start}")
        reference_peak = reference_overlap_peak(data, model.radii, start, steps=200)
        limits = (2.0 * reference_peak).clamp_min(1e-6)

        print("Protocol: full_window_reference_v2")
        print("Full DEM peak overlaps [µm]:",(reference_peak * 1e6).tolist())

        previous = data.states[start - horizon].clone()
        state = data.states[start].clone()
        constant = state.clone()
        ballistic = state.clone()
        peak = maximum_overlaps(state, model.radii)
        failure = None

        print(f"\nStart frame {start}")
        print("Fixed overlap limits [µm]:", (limits * 1e6).tolist())

        for step in range(1, 201):
            prediction = model(state, previous)

            if not torch.isfinite(prediction).all():
                print(f"Nonfinite prediction at step {step}")
                failure = "nonfinite state"
                break

            previous, state = state, prediction

            constant[:, :3] += constant[:, 3:6] * dt

            ballistic[:, :3] += (
                ballistic[:, 3:6] * dt
                + 0.5 * gravity * dt * (dt + 5e-6)
            )
            ballistic[:, 3:6] += gravity * dt

            truth = data.states[start + step * horizon]
            peak = torch.maximum(
                peak, maximum_overlaps(state, model.radii)
            )
            failed = (peak > limits).any().item()

            if step == 1 or step % 25 == 0 or failed:
                print(f"Prediction {step}:")
                for label, candidate in [
                    ("Model", state),
                    ("Constant velocity", constant),
                    ("Gravity only", ballistic),
                ]:
                    print(f"  {label} RMSE:", rmse(candidate, truth))
                print("  Model peak overlap [µm]:",
                      (peak * 1e6).tolist())

            if failed:
                failure = "overlap screening limit exceeded"
                break

        print("Result:", failure or "passed 200 predictions")