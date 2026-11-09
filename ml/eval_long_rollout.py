from pathlib import Path
import torch
import numpy as np
from frame_data import FrameDataset
from reference_data import ReferenceTrajectory
from processor_model import ProcessorModel
from rollout_eval import maximum_overlaps
import argparse
import json
from graph import build_particle_edges, build_wall_ghosts, particle_edge_geometry

root = Path(__file__).resolve().parent.parent
checkpoint_path = (
    root / "experiments/ml_runs/"
    "processor_data42_finetune_rollout4_swap_aware_matched997_v1_best.pt"
)
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
print(f"Checkpoint: {checkpoint_path.name}")

model = ProcessorModel(
    masses=checkpoint["masses"],
    radii=checkpoint["radii"],
    **checkpoint["config"],
)
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.early_gate = True
model.integration_reference_dt = 5e-6
model.contact_position_fraction = checkpoint["contact_position_fraction"]
print("Integration: gravity correction, reference_dt=5e-6")
assert model.rounds == 1
reference_dir = root / "experiments/results/local_timestepping"
filename = (
    "pilot_seed44_10rpm_6s_sparse.csv"
)
data = ReferenceTrajectory(reference_dir / filename)
dense = ReferenceTrajectory(
    reference_dir / "pilot_seed44_10rpm_0p1s_dense.csv"
)
groups = data.groups
horizon = checkpoint["horizon"]
dt = checkpoint["config"]["step_dt"]
gravity = torch.tensor([0.0, 0.0, -9.81])
starts = [6000]
steps = 1400

print("Rollout duration [s]:", steps * dt)

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

def report_spin_by_slip(current, previous, prediction, truth, radii):
    edges = build_particle_edges(current[:, :3], model.cutoff)
    geometry = particle_edge_geometry(current, radii, edges)
    touching = geometry["overlap"] > 0
    senders, receivers = edges[:, touching]
    slip = geometry["tangential_velocity"][touching].norm(dim=1)
    degree = torch.bincount(senders, minlength=len(current))
    walls = build_wall_ghosts(current, previous, radii)
    at_wall = torch.zeros(len(current), dtype=torch.bool, device=current.device)
    at_wall[walls["owners"][walls["gaps"] <= 0]] = True
    isolated = ((degree[senders] == 1) & (degree[receivers] == 1) & ~at_wall[senders] & ~at_wall[receivers])
    for label, mask in (
        ("near-zero slip", slip <= 1e-4),
        ("high slip", slip >= 1e-2),
    ):
        ids = senders[isolated & mask]
        if ids.numel() == 0:
            print(f"  {label}: 0 particles")
            continue
        error = prediction[ids, 6:9] - truth[ids, 6:9]
        print(
            f"  {label}: n={ids.numel()}, "
            f"spin RMSE={error.square().mean().sqrt().item():.6g} rad/s"
        )

def kinetic_energy(state):
    velocity = state[:, 3:6].double()
    spin = state[:, 6:9].double()

    translational = (0.5 * model.masses.double() * velocity.square().sum(dim=1)).sum()
    rotational = (0.5 * model.inertias.double() * spin.square().sum(dim=1)).sum()

    return torch.stack([translational, rotational])


def bulk_errors(prediction, truth, groups, bin_width):
    n = groups.numel()

    if n == 0 or prediction.shape != truth.shape:
        raise ValueError("Expected matching, nonempty states")
    if prediction.shape[0] != n:
        raise ValueError("Particle labels do not match states")
    if not (0.0 < bin_width < float("inf")):
        raise ValueError("Invalid bin width")
    if not (
        torch.isfinite(prediction).all()
        and torch.isfinite(truth).all()
    ):
        raise ValueError("Nonfinite states")

    groups = groups.to(device=prediction.device)

    positions = torch.cat([
        prediction[:, :3],
        truth[:, :3],
    ]).double()
    cells = torch.floor(positions / bin_width).long()

    unique_cells, inverse = torch.unique(
        cells, dim=0, return_inverse=True
    )

    histograms = []
    for indices in (inverse[:n], inverse[n:]):
        keys = 2 * indices + groups
        counts = torch.bincount(
            keys,
            minlength=2 * unique_cells.shape[0],
        ).reshape(-1, 2).double()
        histograms.append(counts)

    predicted, reference = histograms
    difference = (predicted - reference).abs()

    result = {
        "group_density_tv": (
            difference.sum() / (2 * n)
        ).item(),
        "density_tv": (
            (predicted.sum(dim=1) - reference.sum(dim=1))
            .abs().sum() / (2 * n)
        ).item(),
    }

    for group in (0, 1):
        count = (groups == group).sum().item()
        result[f"group{group}_tv"] = (
            (difference[:, group].sum() / (2 * count)).item()
            if count else float("nan")
        )

    return result

def mixing_score(state,groups):
    cells = torch.floor(state[:, :3].double() / 0.004).long()
    x, inverse = torch.unique(cells, dim=0, return_inverse=True)
    counts = torch.bincount(2 * inverse + groups).reshape(-1, 2).double()
    total = counts.sum(dim=1)
    eligible = total >= 5
    included = total[eligible].sum()
    coverage = (included / groups.numel()).item()
    if included.item() == 0:
        return float("nan"), coverage
    blue = counts[eligible,0]
    fraction = blue.sum() / included
    if fraction.item() in (0.0, 1.0):
        return float("nan"),coverage
    local_fraction = blue / total[eligible]
    variance = (total[eligible] * (local_fraction - fraction).square()).sum() / included
    score = 1.0 - variance / (fraction * (1.0 - fraction))
    return score.item(), coverage

with torch.no_grad():
    for start in starts:
        previous = data.state_at_step(start - horizon).clone()
        state = data.state_at_step(start).clone()

        reference_peak = torch.tensor(
            [52.413556333497134e-6, 28.410831086324308e-6],
            dtype=torch.float64,
        )

        limits = (2.0 * reference_peak).clamp_min(1e-6)
        print("Protocol: bulk_distribution_v1")
        print("DEM guard reference peaks, steps 6000–16000:",(reference_peak * 1e6).tolist())

        constant = state.clone()
        ballistic = state.clone()
        peak = maximum_overlaps(state, model.radii)
        failure = None

        print(f"\nStart frame {start}")
        print("Fixed overlap limits [µm]:", (limits * 1e6).tolist())

        for step in range(1, steps + 1):
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

            target_step = start + step * horizon
            peak = torch.maximum(
                peak, maximum_overlaps(state, model.radii)
            )
            failed = (peak > limits).any().item()
            report = step == 1 or step % 20 == 0 or failed
            if report and target_step not in data.steps:
                print(f"DEM step {target_step}: no recorded reference")
            elif report:
                truth = data.state_at_step(target_step)
                history_step = target_step - 2 * horizon
                current_step = target_step - horizon

                if target_step <= dense.steps[-1]:
                    one_step = model(
                        dense.state_at_step(current_step),
                        dense.state_at_step(history_step),
                    )
                    report_spin_by_slip(
                        dense.state_at_step(current_step),
                        dense.state_at_step(history_step),
                        one_step,
                        truth,
                        model.radii,
                    )
                    print(f"  DEM step {target_step}, one-step RMSE:", rmse(one_step, truth))
                    print(
                        "  One-step / autonomous / DEM rotational energy [J]:",
                        kinetic_energy(one_step)[1].item(),
                        kinetic_energy(state)[1].item(),
                        kinetic_energy(truth)[1].item(),
                    )
                print(f"  Elapsed [s]: {step * dt:.6f}")
                print("  Mixing [score, coverage], ML:", mixing_score(state, groups))
                print("  Mixing [score, coverage], DEM:", mixing_score(truth, groups))

                ml_rot = kinetic_energy(state)[1].item()
                dem_rot = kinetic_energy(truth)[1].item()
                print("  Rotational energy absolute error [J]:", abs(ml_rot - dem_rot))
                print(f"Prediction {step}, DEM step {target_step}:")
                for label, candidate in [
                    ("Model", state),
                    ("Constant velocity", constant),
                    ("Gravity only", ballistic),
                ]:
                    print(f"  {label} RMSE:", rmse(candidate, truth))
                print("  Model peak overlap [µm]:",
                      (peak * 1e6).tolist())
                print(
                    "  Model kinetic energy [translation, rotation; J]:",
                    kinetic_energy(state).tolist(),
                )
                print(
                    "  DEM kinetic energy [translation, rotation; J]:",
                    kinetic_energy(truth).tolist(),
                )

                for width in (0.004, 0.005):
                    print(f"  Bin width {width * 1000:.0f} mm:")

                    for label, candidate in (
                        ("Model", state),
                        ("Constant velocity", constant),
                        ("Gravity only", ballistic),
                    ):
                        metrics = bulk_errors(
                            candidate, truth, groups, width
                        )

                        print(f"    {label}: {metrics}")

            if failed:
                failure = "overlap screening limit exceeded"
                print(f"Overlap stop at DEM step {target_step}, elapsed {step * dt:.6f} s",f"particle/wall peaks [µm]: {(peak * 1e6).tolist()}",)
                break

        print("Result:",failure or f"passed {steps} predictions")