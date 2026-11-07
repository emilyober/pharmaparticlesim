import numpy as np
import torch

class ReferenceTrajectory:
    def __init__(self, csv_path):
        rows = np.loadtxt(csv_path,delimiter=",",skiprows=1,ndmin=2)
        steps, counts = np.unique(rows[:, 0], return_counts=True)
        self.num_frames = len(steps)
        self.num_particles = int(counts[0])
        if not np.all(counts == self.num_particles):
            raise ValueError("particle count changes between frames")
        frames = rows.reshape(self.num_frames, self.num_particles, 14)
        if not np.all(frames[:, :, 0] == steps[:, None]):
            raise ValueError("rows must be grouped in increasing step order")
        ids = frames[0, :, 2]
        if len(np.unique(ids)) != self.num_particles:
            raise ValueError("duplicate particle IDs")
        if not np.all(frames[:, :, 2] == ids[None, :]):
            raise ValueError("particle IDs or order change between frames")
        self.steps = steps.astype(np.int64)
        self.times = frames[:, 0, 1].copy()
        self.ids = ids.astype(np.int64)
        self.groups = torch.tensor(frames[0, :, 3], dtype=torch.long)
        self.radii = torch.tensor(frames[0, :, 13], dtype=torch.float32)
        self.states = torch.tensor(frames[:, :, 4:13], dtype=torch.float32)
        self._index_by_step = {
            int(step): index for index, step in enumerate(self.steps)
        }
    def has_step(self, step):
        return step in self._index_by_step

    
    def state_at_step(self, step):
        index = self._index_by_step[step]
        return self.states[index]