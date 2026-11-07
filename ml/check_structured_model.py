from pathlib import Path

import torch

from frame_data import FrameDataset
from structured_model import ParticleInteractionModel

root = Path(__file__).resolve().parent.parent

dataset = FrameDataset(root / (
        "experiments/results/local_timestepping/"
        "pilot_seed42_10rpm_0p1s_dense.csv"
    ),
    horizon=10,
)


sample = dataset.states[::200]

velocity_scale = (sample[:, :, 3:6].norm(dim=-1).max().clamp_min(1e-6).item())
spin_scale = (sample[:, :, 6:9].norm(dim=-1).max().clamp_min(1e-6).item())
density = 1500.0
radii = dataset.radii
masses = density * (4.0 / 3.0) * torch.pi * radii.pow(3)

torch.manual_seed(123)

model = ParticleInteractionModel(masses=masses,radii=radii,velocity_scale=velocity_scale,spin_scale=spin_scale,step_dt=dataset.horizon * 5e-6)
example = dataset[6000]
prediction = model(example["current"],example["previous"])
scales = prediction.new_tensor([0.002, 0.002, 0.002,velocity_scale, velocity_scale, velocity_scale,spin_scale, spin_scale, spin_scale])
loss = ((prediction - example["target"]) / scales).square().mean()
loss.backward()
gradients = [
    parameter.grad
    for parameter in model.parameters()
    if parameter.grad is not None
]

print("Prediction shape:", prediction.shape)
print("Finite predictions:", torch.isfinite(prediction).all().item())
print("Loss:", loss.item())
print("Finite gradients:",all(torch.isfinite(gradient).all().item() for gradient in gradients))
print("Any nonzero gradient:",any(gradient.abs().max().item() > 0 for gradient in gradients))