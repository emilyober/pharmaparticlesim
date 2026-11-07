from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from graph import build_particle_edges, particle_edge_geometry

class FrameDataset(Dataset):
    def __init__(self, csv_path, horizon=10, rollout_steps=1):
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        self.rollout_steps = rollout_steps
        rows = np.loadtxt(csv_path,delimiter=",",skiprows=1,ndmin=2)
        steps, counts = np.unique(rows[:,0],return_counts = True)
        self.num_frames = len(steps)
        self.num_particles = int(counts[0])
        if not np.all(counts == self.num_particles):
            raise ValueError("Particle count changes between frames")
        frames = rows.reshape(self.num_frames, self.num_particles, 14)
        if self.num_frames <= (rollout_steps + 1) * horizon:
            raise ValueError("Not enough frames for history and targets")
        self.states = torch.tensor(frames[:, :, 4:13], dtype=torch.float32)
        self.radii = torch.tensor(frames[0, :, 13], dtype=torch.float32)

    def __len__(self):
        return self.num_frames - (self.rollout_steps+1)*self.horizon

    def __getitem__(self, index):
        frame = index + self.horizon
        targets = torch.stack([self.states[frame + step * self.horizon] for step in range(1, self.rollout_steps + 1)])
        return {
            "previous": self.states[frame-self.horizon],
            "current": self.states[frame],
            "target": targets[0],
            "targets": targets,
        }

if __name__ == "__main__":

    root = Path(__file__).resolve().parent.parent
    dataset = FrameDataset(
        root / (
            "experiments/results/local_timestepping/"
            "pilot_seed42_10rpm_0p1s_dense.csv"
        ),
        horizon=10,
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    print("Examples:", len(dataset))

    for name, values in batch.items():
        print(name, values.shape)

    positions = batch["current"][0, :, :3]
    edge_index = build_particle_edges(positions)

    print("Edge index:", edge_index.shape)
    print("First edges:", edge_index[:, :5])

    geometry = particle_edge_geometry(batch["current"][0], dataset.radii, edge_index)

    for name, values in geometry.items():
        print(name, values.shape)

    assert not (edge_index[0] == edge_index[1]).any()
    tangent_dot_normal = (geometry["tangential_velocity"] * geometry["normal"]).sum(dim=1)

    print("Tangential projection error:",tangent_dot_normal.abs().max().item())

    state = dataset[6000]["current"]

    edge_index = build_particle_edges(state[:, :3])
    geometry = particle_edge_geometry(state, dataset.radii, edge_index)

    projection_error = (geometry["tangential_velocity"] * geometry["normal"]).sum(dim=1).abs().max()

    print("Moving-frame projection error:", projection_error.item())
    print("Maximum tangential speed:",geometry["tangential_velocity"].norm(dim=1).max().item())
    half = edge_index.shape[1] // 2

    print("Normal reversal error:",(geometry["normal"][:half]+ geometry["normal"][half:]).abs().max().item())
    print("Surface-velocity reversal error:",(geometry["relative_surface_velocity"][:half]+ geometry["relative_surface_velocity"][half:]).abs().max().item())