from pathlib import Path
import torch

from frame_data import FrameDataset
from processor_model import ProcessorModel
from rollout_eval import maximum_overlaps
from structured_model import aggregate_pair_impulses
from graph import build_wall_ghosts

root = Path(__file__).resolve().parent.parent
checkpoint = torch.load(
    root / "experiments/ml_runs/processor_conditioned_lr1e-4_best.pt",
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
    "pilot_seed43_10rpm_0p1s_dense.csv",
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


assert model.rounds == 1, "This diagnostic assumes one processing round."

start = 2000
state = data.states[start].clone()
previous = data.states[start - horizon].clone()
truth = data.states[start + horizon]
num_particles = len(state)
contributions = {}


def capture(label):
    def hook(module, inputs, outputs):
        _, _, positions, pairs = inputs
        linear, angular, points = outputs

        net_linear, net_spin = aggregate_pair_impulses(
            positions, pairs, linear, angular, points
        )

        contributions[label] = (
            net_linear[:num_particles].clone(),
            net_spin[:num_particles].clone(),
        )

        i, j = pairs
        distance = (positions[j] - positions[i]).norm(dim=1)

        if label == "Particle":
            gaps = distance - model.radii[i] - model.radii[j]
        else:
            gaps = build_wall_ghosts(
                state, previous, model.radii
            )["gaps"]

        print(f"{label} candidate edges: {pairs.shape[1]}")
        print(f"{label} touching/overlapping edges:",
              (gaps <= 0).sum().item())

        if gaps.numel():
            print(f"{label} gap range [µm]:",
                  gaps.min().item() * 1e6,
                  gaps.max().item() * 1e6)

    return hook


handles = [
    model.decoder.register_forward_hook(capture("Particle")),
    model.wall_decoder.register_forward_hook(capture("Wall")),
]

try:
    with torch.no_grad():
        prediction = model(state, previous)
finally:
    for handle in handles:
        handle.remove()


def report(label, values):
    print(
        f"{label}: component RMS="
        f"{values.square().mean().sqrt().item():.6g}, "
        f"max particle magnitude="
        f"{values.norm(dim=1).max().item():.6g}"
    )


total_dw = torch.zeros_like(state[:, 6:9])

for label, (linear, spin) in contributions.items():
    dv = linear / model.masses[:, None]
    dw = spin / model.inertias[:, None]
    total_dw += dw
    report(f"{label} interaction dv [m/s]", dv)
    report(f"{label} spin change [rad/s]", dw)

report("DEM spin change [rad/s]", truth[:, 6:9] - state[:, 6:9])
report("Model spin change [rad/s]", prediction[:, 6:9] - state[:, 6:9])

print(
    "Spin decomposition error:",
    (
        total_dw - (prediction[:, 6:9] - state[:, 6:9])
    ).abs().max().item(),
)

reference_peak = torch.stack([
    maximum_overlaps(frame, model.radii)
    for frame in data.states[start:start + horizon + 1]
]).amax(dim=0)

print("DEM interval peak overlaps [particle, wall; µm]:",
      (reference_peak * 1e6).tolist())