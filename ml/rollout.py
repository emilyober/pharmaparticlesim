from pathlib import Path
import numpy as np
import torch
from model import NeighborhoodMLP
from scipy.spatial import KDTree
from torch.nn.utils.rnn import pad_sequence
from physics import gravity_changes

root = Path(__file__).resolve().parent.parent

def build_inputs(state):
    positions = state[:, :3]
    velocities = state[:, 3:6]
    x, y, z = positions.unbind(dim=1)
    radial = torch.sqrt(x*x + z*z).clamp_min(1e-8)
    zero = torch.zeros_like(x)
    omega = 10.0 * 2.0 * torch.pi / 60.0
    gaps = torch.stack([0.02 - radial - 0.001,y + 0.01 - 0.001,0.01 - y - 0.001], dim=1)
    inward_normal = torch.stack([-x / radial, zero, -z / radial], dim=1)
    cylinder_velocity = torch.stack([omega * 0.02 * z / radial,zero,-omega * 0.02 * x / radial], dim=1)
    cap_velocity = torch.stack([omega * z, zero, -omega * x], dim=1)
    inputs = torch.cat([state, gaps, inward_normal, cylinder_velocity, cap_velocity], dim=1)
    tree = KDTree(positions.numpy())
    neighbor_lists = tree.query_ball_point(positions.numpy(), r=0.0025)
    features = []
    for particle, ids in enumerate(neighbor_lists):
        ids = [j for j in ids if j != particle]
        features.append(torch.cat([positions[ids] - positions[particle],velocities[ids] - velocities[particle],], dim=1))

    lengths = torch.tensor([len(f) for f in features])
    #if every particle has zero neighbors.
    neighbors = pad_sequence(features, batch_first=True)
    if neighbors.shape[1] == 0:
        neighbors = state.new_zeros((len(state), 1, 6))

    mask = (torch.arange(neighbors.shape[1])[None, :] < lengths[:, None])

    return inputs, neighbors, mask


def penetration_metrics(state):
    positions = state[:, :3]
    radius = 0.001

    #particle-particle overlap; exclude self-distances.
    distances = torch.cdist(positions, positions)
    distances.fill_diagonal_(float("inf"))

    particle_overlap = (2 * radius - distances).clamp_min(0).max()

    #particle penetration into cylinder or end caps.
    radial = torch.sqrt(positions[:, 0].square() + positions[:, 2].square())

    cylinder = (radial + radius - 0.02).clamp_min(0)
    lower_cap = (-0.01 + radius - positions[:, 1]).clamp_min(0)
    upper_cap = (positions[:, 1] + radius - 0.01).clamp_min(0)

    wall_overlap = torch.stack([cylinder, lower_cap, upper_cap]).max()

    return particle_overlap.item(), wall_overlap.item()

checkpoint = torch.load(root/"experiments/ml_runs/seed42_to43_gravity_modelseed123_best.pt",map_location="cpu",weights_only=True)
model = NeighborhoodMLP()
model.load_state_dict(checkpoint["model_state"])
model.eval()
horizon = checkpoint["horizon"]
prediction_dt = horizon*5e-6

start_frame = 6000
rollout_steps = 100
num_particles = 322
num_frames = rollout_steps*horizon+1

csv_path = (root / "experiments/results/local_timestepping" / "pilot_seed43_10rpm_0p1s_dense.csv")
rows = np.loadtxt(csv_path,delimiter=",",skiprows=1 + start_frame * num_particles,max_rows=num_frames * num_particles)
frames = rows.reshape(num_frames, num_particles, 14)

assert np.all(frames[:, :, 0]== np.arange(start_frame, start_frame + num_frames)[:, None])
assert np.all(frames[:, :, 2] == frames[0, :, 2])
assert len(np.unique(frames[0, :, 2])) == num_particles

reference = torch.tensor(frames[:, :, 4:13], dtype=torch.float32)
state = reference[0].clone()
baseline_state = reference[0].clone()
gravity_state = reference[0].clone()
gravity = torch.tensor([0.0, 0.0, -9.81])

with torch.no_grad():
    for step in range(1,rollout_steps+1):
        inputs, neighbors, mask = build_inputs(state)
        inputs = (inputs - checkpoint["input_mean"])/checkpoint["input_std"]
        neighbors = (neighbors - checkpoint["neighbor_mean"])/checkpoint["neighbor_std"]
        corrections = model(inputs, neighbors, mask)
        corrections = corrections * checkpoint["target_std"]
        changes = gravity_changes(state,horizon,dem_dt=checkpoint["dem_dt"])
        state = state + changes + corrections
        baseline_state[:, :3] += (baseline_state[:, 3:6] * prediction_dt)
        #match repeated semi-implicit Euler DEM steps under gravity alone.
        dem_dt = 5e-6
        gravity_state[:, :3] += (gravity_state[:, 3:6] * prediction_dt + 0.5 * gravity * prediction_dt * (prediction_dt + dem_dt))
        gravity_state[:, 3:6] += gravity * prediction_dt
        truth = reference[step * horizon]
        def component_rmse(prediction):
            return ((prediction - truth).square().reshape(num_particles, 3, 3).mean(dim=(0, 2)).sqrt())
        if not torch.isfinite(state).all():
            print(f"Stopped at prediction {step}: nonfinite model state")
            break

        if step == 1 or step % 10 == 0:
            print(f"Prediction {step}:")
            print("  Model RMSE:   ", component_rmse(state))
            print("  Baseline RMSE:", component_rmse(baseline_state))
            print("  Gravity RMSE: ", component_rmse(gravity_state))

            for label, prediction in [
                ("Model", state),
                ("DEM", truth),
                ("Baseline", baseline_state),
                ("Gravity", gravity_state),
            ]:
                particle, wall = penetration_metrics(prediction)

                print(
                    f"  {label} max overlap: "
                    f"particle={particle * 1e6:.2f} µm, "
                    f"wall={wall * 1e6:.2f} µm"
                )
