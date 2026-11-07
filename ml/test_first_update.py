from pathlib import Path
import torch
from frame_data import FrameDataset
from physics import gravity_changes
from processor_model import ProcessorModel
from rollout_eval import evaluate_rollouts
import csv
from datetime import datetime

root = Path(__file__).resolve().parent.parent
results = root / "experiments/results/local_timestepping"
train_data = FrameDataset(results / "pilot_seed42_10rpm_0p1s_dense.csv",horizon=10,rollout_steps=2)
train_indices = torch.randperm(len(train_data),generator=torch.Generator().manual_seed(42))[:1000]
frames = train_indices + train_data.horizon
current = train_data.states[frames]
previous = train_data.states[frames - train_data.horizon]
future = train_data.states[frames + train_data.horizon]
velocity_scale = torch.stack([current[:, :, 3:6].norm(dim=-1).max(),previous[:, :, 3:6].norm(dim=-1).max()]).max().clamp_min(1e-6).item()
spin_scale = torch.stack([current[:, :, 6:9].norm(dim=-1).max(),previous[:, :, 6:9].norm(dim=-1).max()]).max().clamp_min(1e-6).item()
changes = future - current
group_scales = (changes.reshape(-1,3,3).square().mean(dim=(0,2)).sqrt().clamp_min(1e-8))
loss_scale = group_scales.repeat_interleave(3)
radii = train_data.radii
masses = 1500.0 * (4.0 / 3.0) * torch.pi * radii.pow(3)
config = {
    "velocity_scale": velocity_scale,
    "spin_scale": spin_scale,
    "step_dt": train_data.horizon * 5e-6,
    "cutoff": 0.0025,
    "length_scale": 0.002,
    "hidden_size": 64,
    "rounds": 3
}
torch.manual_seed(123)
model = ProcessorModel(masses=masses,radii=radii,**config,)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
initial_state = {
    name: value.detach().clone()
    for name, value in model.state_dict().items()
}
order = train_indices[
    torch.randperm(
        len(train_indices),
        generator=torch.Generator().manual_seed(987),
    )[:2]
].tolist()

examples = [train_data[index] for index in order]

print("Frames:", [
    index + train_data.horizon for index in order
])


def example_loss(model, example):
    prediction = model(
        example["current"],
        example["previous"],
    )

    return (
        (prediction - example["target"]) / loss_scale
    ).square().mean()


case_config = dict(config)
case_config["rounds"] = 1

for learning_rate in [1e-5, 1e-6, 1e-7]:
    model = ProcessorModel(
        masses=masses,
        radii=radii,
        **case_config,
    )
    model.load_state_dict(initial_state)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    model.eval()
    with torch.no_grad():
        before = [
            example_loss(model, example).item()
            for example in examples
        ]

    model.train()
    optimizer.zero_grad()
    loss = example_loss(model, examples[0])
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]

    if not all(torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Nonfinite gradient before first update")

    gradient_norm = torch.stack([
        gradient.double().square().sum()
        for gradient in gradients
    ]).sum().sqrt().item()

    decoder_before = model.decoder.output.weight.detach().clone()

    optimizer.step()

    decoder_update = (
        model.decoder.output.weight.detach() - decoder_before
    ).abs().max().item()

    model.eval()
    with torch.no_grad():
        after = [
            example_loss(model, example).item()
            for example in examples
        ]

    print(f"\nLearning rate: {learning_rate}")
    print("Loss before, examples 1 / 2:", before)
    print("Loss after,  examples 1 / 2:", after)
    print("Gradient norm before update:", gradient_norm)
    print("Maximum decoder weight change:", decoder_update)