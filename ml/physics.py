import torch


def gravity_changes(state, horizon, dem_dt=5e-6):
    elapsed = horizon * dem_dt
    gravity = state.new_tensor([0.0, 0.0, -9.81])
    changes = state.new_zeros((state.shape[0], 9))
    changes[:, :3] = (state[:, 3:6] * elapsed + 0.5 * gravity * elapsed * (elapsed + dem_dt))
    changes[:, 3:6] = gravity * elapsed

    return changes