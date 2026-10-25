import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from data import ParticleDataset, collate_particles
from pathlib import Path
from sklearn.model_selection import train_test_split
from model import NeighborhoodMLP

csv_path = (Path(__file__).resolve().parent.parent / "experiments/results/local_timestepping/pilot_10rpm.csv")
dataset = ParticleDataset(csv_path, horizon=10)
start_frames = range(dataset.num_frames - dataset.horizon)
train_frames, val_frames = train_test_split(start_frames,test_size=0.2,shuffle=False,)

#remove training examples where the future state is in validation
val_start = val_frames[0]
train_frames = [frame for frame in train_frames if frame + dataset.horizon < val_start]
num = dataset.num_particle
train_dataset = Subset(dataset, range(train_frames[0]*num,(train_frames[-1]+1)*num))
generator = torch.Generator().manual_seed(42)
indices = torch.randperm(len(train_dataset), generator=generator)[:100_000].tolist()
train_dataset = Subset(train_dataset, indices)
validation_dataset = Subset(dataset,range(val_frames[0]*num,(val_frames[-1]+1)*num))
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
torch.manual_seed(42)
model = NeighborhoodMLP()
nn.init.zeros_(model.predictor[-1].weight)
nn.init.zeros_(model.predictor[-1].bias)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(5):
    model.train()
    train_loss = 0.0
    train_count = 0
    for inputs,neighbors,mask,targets,contacts in train_loader:
        changes = torch.zeros_like(targets)
        changes[:, :3] = inputs[:, 3:6] * dataset.horizon * 5e-6
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
            inputs = (inputs-input_mean)/input_std
            neighbors = (neighbors - neighbor_mean) / neighbor_std
            targets = (targets - target_mean)/target_std
            corrections = model(inputs, neighbors, mask)
            predictions = baseline_predictions + corrections
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


baseline_loss = 0.0
component_error = torch.zeros(3)
count = 0
prediction_dt = dataset.horizon * 5e-6

with torch.no_grad():
    for inputs, neighbors, mask, targets, contacts in validation_loader:
        changes = torch.zeros_like(targets)
        changes[:, :3] = inputs[:, 3:6] * prediction_dt

        predictions = (changes - target_mean) / target_std
        targets = (targets - target_mean) / target_std

        batch_size = inputs.shape[0]
        loss = loss_fn(predictions, targets)
        baseline_loss += loss.item() * inputs.shape[0]
        squared_error = (predictions - targets).square()
        component_error += torch.stack([
            squared_error[:, :3].mean(),
            squared_error[:, 3:6].mean(),
            squared_error[:, 6:9].mean(),
        ]) * batch_size
        count += batch_size

print(f"Constant-velocity validation loss: {baseline_loss / count:.6f}")
print("Position / velocity / spin:", component_error / count)