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
val_data = FrameDataset(results / "pilot_seed43_10rpm_0p1s_dense.csv",horizon=10,rollout_steps=2)
train_indices = torch.randperm(len(train_data),generator=torch.Generator().manual_seed(42))[:1000]
val_indices = torch.randperm(len(val_data),generator=torch.Generator().manual_seed(43))[:200]
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
initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
order = train_indices[
    torch.randperm(
        len(train_indices),
        generator=torch.Generator().manual_seed(987),
    )[:100]
].tolist()
output_dir = root / "experiments/summaries/diagnostics"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / (
    f"processor_{datetime.now():%Y%m%d_%H%M%S}.csv"
)
fields = [
    "rounds", "training_steps", "update", "frame",
    "loss_step1", "loss_step2", "loss",
    "gradient_norm", "largest_gradient",
    "largest_gradient_parameter", "status",
]

@torch.no_grad()
def fixed_set_loss(model, indices, training_steps):
    model.eval()
    losses = []

    for index in indices:
        example = train_data[index]
        previous = example["previous"]
        state = example["current"]
        step_losses = []

        for target in example["targets"][:training_steps]:
            prediction = model(state, previous)
            step_losses.append(
                ((prediction - target) / loss_scale).square().mean()
            )
            previous, state = state, prediction

        losses.append(torch.stack(step_losses).mean())

    model.train()
    return torch.stack(losses).mean().item()

with output_path.open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()

    for rounds, training_steps in [(1, 1), (3, 1), (1, 2), (3, 2)]:
        case_config = dict(config)
        case_config["rounds"] = rounds

        model = ProcessorModel(
            masses=masses,
            radii=radii,
            **case_config,
        )
        model.load_state_dict(initial_state)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=1e-6
        )

        print(f"\nCASE: rounds={rounds}, training_steps={training_steps}")
        before = fixed_set_loss(model, order, training_steps)
        print(f"Fixed-set loss before: {before:.6f}")
        for update, index in enumerate(order, start=1):
            example = train_data[index]
            previous = example["previous"]
            state = example["current"]

            optimizer.zero_grad()
            losses = []

            row = {
                "rounds": rounds,
                "training_steps": training_steps,
                "update": update,
                "frame": index + train_data.horizon,
                "status": "ok",
            }

            try:
                for target in example["targets"][:training_steps]:
                    prediction = model(state, previous)

                    step_loss = (
                        (prediction - target) / loss_scale
                    ).square().mean()

                    losses.append(step_loss)
                    previous, state = state, prediction

                loss = torch.stack(losses).mean()

                row["loss_step1"] = losses[0].item()
                row["loss_step2"] = (
                    losses[1].item() if len(losses) == 2 else ""
                )
                row["loss"] = loss.item()

                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite forward loss")

                loss.backward()

                gradients = [
                    (name, parameter.grad)
                    for name, parameter in model.named_parameters()
                    if parameter.grad is not None
                ]

                if not gradients or any(
                    not torch.isfinite(gradient).all()
                    for _, gradient in gradients
                ):
                    raise RuntimeError("Missing or nonfinite gradients")
                gradient_norm = torch.stack([
                    gradient.double().square().sum()
                    for _, gradient in gradients
                ]).sum().sqrt().item()

                largest_name, largest_gradient = max(
                    (
                        (name, gradient.abs().max().item())
                        for name, gradient in gradients
                    ),
                    key=lambda item: item[1],
                )

                row["gradient_norm"] = gradient_norm
                row["largest_gradient"] = largest_gradient
                row["largest_gradient_parameter"] = largest_name
                optimizer.step()

                if any(
                    not torch.isfinite(parameter).all()
                    for parameter in model.parameters()
                ):
                    raise RuntimeError("Nonfinite parameters after update")

            except RuntimeError as error:
                row["status"] = str(error)

            writer.writerow(row)
            file.flush()

            if update == 1 or update % 10 == 0 or row["status"] != "ok":
                print(
                    f"update={update}, frame={row['frame']}, "
                    f"loss1={row.get('loss_step1')}, "
                    f"loss2={row.get('loss_step2')}, "
                    f"grad={row.get('gradient_norm')}, "
                    f"largest={row.get('largest_gradient_parameter')}, "
                    f"status={row['status']}"
                )

            if row["status"] != "ok":
                break
        after = fixed_set_loss(model, order, training_steps)
        print(f"Fixed-set loss after:  {after:.6f}")
        print(f"Loss reduction: {100 * (before - after) / before:.2f}%")

print("\nDiagnostic saved to:", output_path)
