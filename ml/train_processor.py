from pathlib import Path
import torch
from frame_data import FrameDataset
from physics import gravity_changes
from processor_model import ProcessorModel
from rollout_eval import evaluate_rollouts

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
    "rounds": 1,
    "interaction_gate": True,
}
torch.manual_seed(123)
model = ProcessorModel(masses=masses,radii=radii,**config,)
with torch.no_grad():
    for decoder in (model.decoder, model.wall_decoder):
        decoder.linear_scale.copy_(masses.mean() * loss_scale[3])
        decoder.angular_scale.copy_(
            model.inertias.mean() * loss_scale[6]
        )
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def evaluate():
    model.eval()
    totals = torch.zeros(3, 3)

    with torch.no_grad():
        for index in val_indices.tolist():
            example = val_data[index]
            current = example["current"]
            target = example["target"]

            predicted = model(current, example["previous"])
            constant = current.clone()
            constant[:, :3] += (current[:, 3:6] * config["step_dt"])
            gravity = current + gravity_changes(current, train_data.horizon)

            for row, prediction in enumerate([predicted, constant, gravity]):
                error = ((prediction - target) / loss_scale).square()
                totals[row] += (error.reshape(-1, 3, 3).mean(dim=(0, 2)))

    return totals / len(val_indices)

checkpoint_dir = root / "experiments/ml_runs"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_rollout = float("inf")
rollout_starts = [2000, 6000, 10000, 14000, 18000]
rollout_steps = 50

latest_path = checkpoint_dir / "processor_surface_slip_gated_latest.pt"
start_epoch = 0

if latest_path.exists():
    checkpoint = torch.load(latest_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_epoch = checkpoint["epoch"]
    best_rollout = checkpoint["best_rollout_score"]
    torch.set_rng_state(checkpoint["torch_rng_state"])
    print(f"Resuming after epoch {start_epoch}")
for epoch in range(start_epoch,20):
    model.train()
    total_loss = 0.0

    order = train_indices[torch.randperm(len(train_indices))]
    for index in order.tolist():
        example = train_data[index]
        previous = example["previous"]
        state = example["current"]
        optimizer.zero_grad()
        step_losses = []
        for target in example["targets"]:
            prediction = model(state,previous)
            step_loss = ((prediction - target) / loss_scale).square().mean()
            step_losses.append(step_loss)
            previous,state = state, prediction
        loss = torch.stack(step_losses).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss")
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    metrics = evaluate()
    val_loss = metrics[0].mean().item()

    print(
        f"Epoch {epoch + 1}: "
        f"train={total_loss / len(train_indices):.6f}, "
        f"val={val_loss:.6f}"
    )

    for label, components in zip(["Model", "Constant velocity", "Gravity only"],metrics):
        print(
            f"{label}: loss={components.mean().item():.6f}, "
            f"position/velocity/spin={components.tolist()}"
        )

    if (epoch + 1) % 5 == 0:
        rollout = evaluate_rollouts(
            model,
            val_data,
            loss_scale,
            starts=rollout_starts,
            steps=rollout_steps,
            overlap_factor=2.0,
        )

        for report in rollout["windows"]:
            print(
                f"Rollout start={report['start_frame']}: "
                f"train_rollout2={total_loss / len(train_indices):.6f}, "
                f"val_one_step={val_loss:.6f}"
                f"completed={report['completed']}/{rollout_steps}, "
                f"failure={report['failure']}, "
                f"peak particle/wall µm="
                f"{[v * 1e6 for v in report['peak_overlap_m']]}"
            )

        print(
            f"Rollout eligible={rollout['eligible']}, "
            f"score={rollout['score']}"
        )

        if rollout["eligible"] and rollout["score"] < best_rollout:
            best_rollout = rollout["score"]

            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "config": config,
                "masses": masses,
                "radii": radii,
                "loss_scale": loss_scale,
                "horizon": train_data.horizon,
                "train_indices": train_indices,
                "val_indices": val_indices,
                "best_rollout_score": best_rollout,
                "one_step_val_loss": val_loss,
                "rollout": rollout,
                "rollout_starts": rollout_starts,
                "rollout_steps": rollout_steps,
                "overlap_factor": 2.0,
                "training_rollout_steps": train_data.rollout_steps,
                "overlap_floor_m": 1e-6,
                "torch_rng_state": torch.get_rng_state(),
                "boundary_version": "outward_ghost_v2",
                "architecture": "processor_M1_surface_slip_v1",
            }, checkpoint_dir / "processor_surface_slip_gated_best.pt")

            print("Saved best screened rollout checkpoint")
    torch.save({
        "epoch": epoch + 1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_rollout_score": best_rollout,
        "config": config,
        "loss_scale": loss_scale,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "torch_rng_state": torch.get_rng_state(),
    }, latest_path)