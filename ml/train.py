import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from data import ParticleDataset, collate_particles
from pathlib import Path
from model import NeighborhoodMLP
from physics import gravity_changes

results_dir = (Path(__file__).resolve().parent.parent/ "experiments/results/local_timestepping")
dataset = ParticleDataset(results_dir / "pilot_seed42_10rpm_0p1s_dense.csv",horizon=10)
#fixed 100,000-example val subset.
validation_dataset = ParticleDataset(results_dir / "pilot_seed43_10rpm_0p1s_dense.csv",horizon=10,)
val_generator = torch.Generator().manual_seed(43)
val_indices = torch.randperm(len(validation_dataset), generator=val_generator)[:100_000].tolist()
validation_dataset = Subset(validation_dataset, val_indices)
#fixed 100,000-example training subset.
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(dataset), generator=generator)[:100_000].tolist()
train_dataset = Subset(dataset, indices)
train_loader = DataLoader(train_dataset,batch_size=256,shuffle=True,num_workers=0,collate_fn=collate_particles,)
validation_loader = DataLoader(validation_dataset,batch_size=256,shuffle=False,num_workers=0,collate_fn=collate_particles,)

#normalize
neighbor_sum = torch.zeros(6, dtype=torch.float64)
neighbor_sq_sum = torch.zeros(6, dtype=torch.float64)
neighbor_count = 0
input_sum = torch.zeros(21, dtype=torch.float64)
input_sq_sum = torch.zeros(21, dtype=torch.float64)
target_sum = torch.zeros(9, dtype=torch.float64)
target_sq_sum = torch.zeros(9, dtype=torch.float64)
count = 0

for inputs, neighbors, mask, targets, contacts in train_loader:
    real_neighbors = neighbors[mask].double()

    neighbor_sum += real_neighbors.sum(dim=0)
    neighbor_sq_sum += real_neighbors.square().sum(dim=0)
    neighbor_count += real_neighbors.shape[0]

    inputs = inputs.double()
    targets = targets.double()

    input_sum += inputs.sum(dim=0)
    input_sq_sum += inputs.square().sum(dim=0)

    target_sum += targets.sum(dim=0)
    target_sq_sum += targets.square().sum(dim=0)

    count += inputs.shape[0]
neighbor_mean = neighbor_sum / neighbor_count
neighbor_std = (neighbor_sq_sum / neighbor_count - neighbor_mean.square()).clamp_min(0).sqrt().clamp_min(1e-8)
neighbor_mean = neighbor_mean.float()
neighbor_std = neighbor_std.float()
input_mean = input_sum / count
target_mean = target_sum / count
input_std = (input_sq_sum / count - input_mean.square()).clamp_min(0).sqrt().clamp_min(1e-8)
target_std = (target_sq_sum / count - target_mean.square()).clamp_min(0).sqrt().clamp_min(1e-8)
input_mean, input_std = input_mean.float(), input_std.float()
target_mean, target_std = target_mean.float(), target_std.float()

#training + eval loop
loss_fn = nn.MSELoss()
torch.manual_seed(123)
model = NeighborhoodMLP()
nn.init.zeros_(model.predictor[-1].weight)
nn.init.zeros_(model.predictor[-1].bias)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
checkpoint_dir = Path(__file__).resolve().parent.parent / "experiments/ml_runs"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
best_val_loss = float("inf")
# checkpoint = torch.load(checkpoint_dir / "seed42_to43_best.pt",map_location="cpu",weights_only=True,)
# model.load_state_dict(checkpoint["model_state"])
# optimizer.load_state_dict(checkpoint["optimizer_state"])
# input_mean = checkpoint["input_mean"]
# input_std = checkpoint["input_std"]
# neighbor_mean = checkpoint["neicheckpoint = ghbor_mean"]
# neighbor_std = checkpoint["neighbor_std"]
# target_mean = checkpoint["target_mean"]
# target_std = checkpoint["target_std"]

# best_val_loss = checkpoint["best_val_loss"]
# start_epoch = checkpoint["epoch"]

# torch.set_rng_state(checkpoint["torch_rng_state"])
start_epoch = 0
for epoch in range(start_epoch,start_epoch+20):
    model.train()
    train_loss = 0.0
    train_count = 0
    for inputs,neighbors,mask,targets,contacts in train_loader:
        changes = gravity_changes(inputs, dataset.horizon)
        baseline_predictions = (changes - target_mean) / target_std
    
        inputs = (inputs - input_mean) / input_std
        neighbors = (neighbors - neighbor_mean) / neighbor_std
        targets = (targets - target_mean) / target_std

        optimizer.zero_grad()

        corrections = model(inputs, neighbors, mask)
        predictions = baseline_predictions + corrections

        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.shape[0]
        train_count += inputs.shape[0]
    
    model.eval()
    val_loss = 0.0
    val_count = 0
    component_error = torch.zeros(3)
    group_sums = torch.zeros(2, 2)
    group_counts = torch.zeros(2)
    with torch.no_grad():
        for inputs, neighbors, mask, targets, contacts in validation_loader:
            changes = torch.zeros_like(targets)
            changes[:, :3] = inputs[:, 3:6] * dataset.horizon * 5e-6
            baseline_predictions = (changes - target_mean) / target_std
            gravity_prediction = (gravity_changes(inputs, dataset.horizon) - target_mean) / target_std
            inputs = (inputs-input_mean)/input_std
            neighbors = (neighbors - neighbor_mean) / neighbor_std
            targets = (targets - target_mean)/target_std
            corrections = model(inputs, neighbors, mask)
            predictions = gravity_prediction + corrections
            model_errors = (predictions - targets).square().mean(dim=1)
            baseline_errors = (baseline_predictions - targets).square().mean(dim=1)
            for group in [0, 1]:
                selected = contacts == bool(group)
                group_sums[group, 0] += model_errors[selected].sum()
                group_sums[group, 1] += baseline_errors[selected].sum()
                group_counts[group] += selected.sum()
            squared_error = (predictions - targets).square()
            loss = loss_fn(predictions,targets)
            val_loss += loss.item()*inputs.shape[0]
            val_count += inputs.shape[0]
            component_error += torch.stack([
                squared_error[:, :3].mean(),
                squared_error[:, 3:6].mean(),
                squared_error[:, 6:9].mean(),
            ]) * inputs.shape[0]
    current_val_loss = val_loss / val_count

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "input_mean": input_mean,
            "input_std": input_std,
            "neighbor_mean": neighbor_mean,
            "neighbor_std": neighbor_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "horizon": dataset.horizon,
            "train_indices": indices,
            "baseline_type": "gravity",
            "dem_dt": 5e-6,
            "validation_indices": val_indices,
            "torch_rng_state": torch.get_rng_state(),
        }, checkpoint_dir / "seed42_to43_gravity_modelseed123_best.pt")

        print("Saved best checkpoint")
    for group, label in enumerate(["No particle contact", "Particle contact"]):
        count = int(group_counts[group].item())

        if count:
            errors = group_sums[group] / group_counts[group]
            print(
                f"{label}: n={count}, "
                f"model={errors[0].item():.6f}, "
                f"baseline={errors[1].item():.6f}"
            )
    print(
        f"Epoch {epoch + 1}: "
        f"train={train_loss / train_count:.6f}, "
        f"val={val_loss / val_count:.6f}"
    )
    print("Position / velocity / spin:", component_error / val_count)


for name in ["Constant velocity", "Gravity only"]:
    component_error = torch.zeros(3)
    count = 0

    with torch.no_grad():
        for inputs, neighbors, mask, targets, contacts in validation_loader:
            if name == "Gravity only":
                changes = gravity_changes(inputs, dataset.horizon)
            else:
                changes = torch.zeros_like(targets)
                changes[:, :3] = (
                    inputs[:, 3:6] * dataset.horizon * 5e-6
                )

            squared_error = ((changes - targets) / target_std).square()

            component_error += torch.stack([
                squared_error[:, :3].sum(),
                squared_error[:, 3:6].sum(),
                squared_error[:, 6:9].sum(),
            ])

            count += inputs.shape[0]

    errors = component_error / (count * 3)

    print(f"{name} validation loss: {errors.mean().item():.6f}")
    print("Position / velocity / spin:", errors)
