from pathlib import Path
import numpy as np
import torch
from structured_model import ParticleInteractionModel

root = Path(__file__).resolve().parent.parent

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

checkpoint = torch.load(root / "experiments/ml_runs/structured_M1_best.pt",map_location="cpu",weights_only=True)
model = ParticleInteractionModel(masses=checkpoint["masses"],radii=checkpoint["radii"],**checkpoint["config"])
model.load_state_dict(checkpoint["model_state"])
model.eval()
horizon = checkpoint["horizon"]
prediction_dt = checkpoint["config"]["step_dt"]
print("Checkpoint epoch:", checkpoint["epoch"])

start_frame = 6000
rollout_steps = 100
num_particles = len(checkpoint["radii"])
load_start = start_frame - horizon
num_frames = (rollout_steps + 1) * horizon + 1
csv_path = (root / "experiments/results/local_timestepping" / "pilot_seed43_10rpm_0p1s_dense.csv")
rows = np.loadtxt(csv_path,delimiter=",",skiprows=1 + load_start * num_particles,max_rows=num_frames * num_particles)
frames = rows.reshape(num_frames, num_particles, 14)

assert np.all(frames[:, :, 0] == np.arange(load_start, load_start + num_frames)[:, None])
assert np.all(frames[:, :, 2] == frames[0, :, 2])
assert len(np.unique(frames[0, :, 2])) == num_particles
assert np.allclose(frames[0, :, 13], checkpoint["radii"].numpy())

all_states = torch.tensor(frames[:, :, 4:13], dtype=torch.float32)

previous_state = all_states[0].clone()
reference = all_states[horizon:]

state = reference[0].clone()
baseline_state = state.clone()
gravity_state = state.clone()
gravity = torch.tensor([0.0, 0.0, -9.81])

with torch.no_grad():
    for step in range(1,rollout_steps+1):
        next_state = model(state, previous_state)
        previous_state = state
        state = next_state
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
