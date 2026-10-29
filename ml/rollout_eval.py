import torch

def maximum_overlaps(state,radii):
    positions = state[:,:3].double()
    radii = radii.double()
    distances = torch.cdist(positions,positions,compute_mode="donot_use_mm_for_euclid_dist")
    distances.fill_diagonal_(float("inf"))
    pair_overlap = (radii[:,None]+radii[None,:]-distances).clamp_min(0).max()
    radial = positions[:,[0,2]].norm(dim=1)
    wall_overlap = torch.stack([radial+radii-0.02,-0.01+radii-positions[:,1],positions[:,1]+radii-0.01]).clamp_min(0).max()
    return torch.stack([pair_overlap,wall_overlap])

@torch.no_grad()
def reference_overlap_peak(dataset, radii, start, steps):
    stop = start + steps * dataset.horizon
    if start < 0 or stop >= dataset.num_frames:
        raise ValueError("Reference window exceeds available data")

    return torch.stack([maximum_overlaps(frame, radii) for frame in dataset.states[start:stop + 1]]).amax(dim=0)

@torch.no_grad()
def evaluate_rollouts(model,dataset,loss_scale,starts,steps=50,overlap_factor=2.0):
    was_training = model.training
    model.eval()
    reports = []
    try:
        for start in starts:
            horizon = dataset.horizon
            if (start < horizon or start + steps * horizon >= dataset.num_frames):
                raise ValueError("rollout window exceeds available data")
            reference = dataset.states[start:start + steps * horizon + 1:horizon]
            reference_peak = reference_overlap_peak(dataset,model.radii,start,steps)
            limits = (overlap_factor*reference_peak).clamp_min(1e-6)
            previous = dataset.states[start-horizon].clone()
            state = reference[0].clone()
            peak = maximum_overlaps(state,model.radii)
            total_error = 0.0
            completed = 0
            failure = None
            for step in range(1,steps+1):
                next_state = model(state,previous)
                if not torch.isfinite(next_state).all():
                    failure = "nonfinite state"
                    break
                previous, state = state, next_state
                overlaps = maximum_overlaps(state,model.radii)
                peak = torch.maximum(peak,overlaps)
                error = ((state.double()-reference[step].double())/loss_scale.double()).square().mean().item()
                if not torch.isfinite(torch.tensor(error)):
                    failure = "nonfinite error"
                    break
                total_error += error
                completed = step
                if (peak > limits).any():
                    failure = "overlap screening limit exceeded"
                    break
            reports.append({
                "start_frame": start,
                "completed": completed,
                "failure": failure,
                "score": (
                    total_error / steps
                    if failure is None else float("inf")
                ),
                "peak_overlap_m": peak.tolist(),
                "reference_peak_m": reference_peak.tolist(),
                "limits_m": limits.tolist(),
            })
    finally:
        model.train(was_training)

    eligible = all(report["failure"] is None for report in reports)
    return {
        "eligible": eligible,
        "score": (
            sum(report["score"] for report in reports) / len(reports)
            if eligible else float("inf")
        ),
        "windows": reports,
    }