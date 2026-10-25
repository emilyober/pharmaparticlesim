from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import KDTree

def collate_particles(batch):
    inputs, neighbors, targets, contacts = zip(*batch)
    lengths = torch.tensor([len(n) for n in neighbors])
    padded = torch.nn.utils.rnn.pad_sequence(neighbors,batch_first=True)
    mask = (torch.arange(padded.shape[1]))[None, :] < lengths[:, None]
    return (torch.stack(inputs), padded, mask, torch.stack(targets), torch.stack(contacts))



class ParticleDataset(Dataset):
    def __init__(self, csv_path, horizon=10):
        if not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        self.horizon = horizon
        data = np.loadtxt(csv_path,delimiter=",",skiprows=1,dtype=np.float64,ndmin=2)
        self.num_frames = len(np.unique(data[:, 0]))
        self.num_particle = len(np.unique(data[:, 2]))
        frames = data.reshape(self.num_frames, self.num_particle, 14)
        self.positions = torch.tensor(frames[:, :, 4:7], dtype=torch.float32)
        self.velocities = torch.tensor(frames[:, :, 7:10], dtype=torch.float32)
        self.spins = torch.tensor(frames[:, :, 10:13], dtype=torch.float32)
        self.neighbors = []
        for frame in range(self.num_frames):
            positions = self.positions[frame].numpy()
            tree = KDTree(positions)
            neighbor_lists = tree.query_ball_point(positions, r=0.0025)
            neighbor_lists = [[j for j in neighbors if j != particle] for particle,neighbors in enumerate(neighbor_lists)]
            self.neighbors.append(neighbor_lists)
    def __len__(self):
        return (self.num_frames-self.horizon)*self.num_particle
    def __getitem__(self, index):
        frame = index // self.num_particle
        particle = index % self.num_particle
        future = frame + self.horizon
        current_state = torch.cat([self.positions[frame,particle],self.velocities[frame,particle], self.spins[frame,particle]])
        next_state = torch.cat([self.positions[future,particle],self.velocities[future,particle],self.spins[future,particle]])
        x, y, z = self.positions[frame,particle]
        radial_distance = torch.sqrt(x*x + z*z).clamp_min(1e-8)
        gaps = torch.stack([0.02-radial_distance-0.001, y+0.01-0.001, 0.01-y-0.001])
        inward_normal = torch.stack([-x/radial_distance, torch.zeros_like(x), -z/radial_distance])
        omega = 10.0*2.0*torch.pi/60.0
        cylinder_velocity = torch.stack([omega*0.02*z/radial_distance,torch.zeros_like(x),-omega*0.02*x/radial_distance])
        cap_velocity = torch.stack([omega*z,torch.zeros_like(x),-omega*x])
        wall_features = torch.cat([gaps, inward_normal, cylinder_velocity, cap_velocity])
        current = torch.cat([current_state, wall_features])
        neighbor_ids = self.neighbors[frame][particle]
        relative_positions = (self.positions[frame, neighbor_ids]- self.positions[frame, particle])
        in_contact = (relative_positions.square().sum(dim=1) < 0.002**2).any()
        relative_velocities = (self.velocities[frame, neighbor_ids] - self.velocities[frame, particle])
        neighbor_features = torch.cat([relative_positions, relative_velocities], dim=1)
        return current, neighbor_features, next_state-current_state,in_contact

if __name__ == "__main__":
    csv_path = (Path(__file__).resolve().parent.parent / "experiments/results/local_timestepping/pilot_10rpm.csv")
    dataset = ParticleDataset(csv_path, horizon=10)
    loader = DataLoader(dataset,batch_size=256,shuffle=False,num_workers=0,collate_fn=collate_particles)
    inputs, neighbors, mask, targets, contacts = next(iter(loader))
    print("Inputs:", inputs.shape)
    print("Neighbors:", neighbors.shape)
    print("Mask:", mask.shape)
    print("Targets:", targets.shape)
    print("Contact flags:", contacts.shape, contacts.dtype)
    from model import NeighborhoodMLP

    model = NeighborhoodMLP()
    predictions = model(inputs, neighbors, mask)
    print("Predictions:", predictions.shape)