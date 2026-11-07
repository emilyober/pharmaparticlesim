from pathlib import Path
import torch
import cProfile
import pstats
import subprocess
import re
from time import perf_counter
from statistics import median
from frame_data import FrameDataset
from processor_model import ProcessorModel
from rollout_eval import maximum_overlaps, reference_overlap_peak

root = Path(__file__).resolve().parent.parent
checkpoint = torch.load(
    root / "experiments/ml_runs/processor_surface_slip_gated_best.pt",
    map_location="cpu",
    weights_only=True,
)

model = ProcessorModel(
    masses=checkpoint["masses"],
    radii=checkpoint["radii"],
    **checkpoint["config"],
)
model.load_state_dict(checkpoint["model_state"])
model.eval()
model.early_gate = True
data = FrameDataset(
    root / "experiments/results/local_timestepping/"
    "pilot_seed44_10rpm_0p1s_dense.csv",
    horizon=checkpoint["horizon"],
    rollout_steps=2,
)

horizon = data.horizon
dt = checkpoint["config"]["step_dt"]
gravity = torch.tensor([0.0, 0.0, -9.81])
starts = checkpoint["rollout_starts"]

print("Checkpoint epoch:", checkpoint["epoch"])
print("Prediction interval:", dt)
print("RMSE order: position [m], velocity [m/s], spin [rad/s]")


def rmse(state, truth):
    return (
        (state.double() - truth.double())
        .square()
        .reshape(-1, 3, 3)
        .mean(dim=(0, 2))
        .sqrt()
        .tolist()
    )


#compare against classical timing
torch.set_num_threads(1)
start = 6000
steps = 200
repeats = 5

initial_previous = data.states[start-horizon].clone()
initial_state = data.states[start].clone()

@torch.inference_mode()
def run_window():
    previous = initial_previous.clone()
    state = initial_state.clone()
    for step in range(steps):
        prediction = model(state,previous)
        previous,state = state,prediction
    return state

def run_dem():
    result = subprocess.run(
        [str(root / "build/pharmaparticlesim")],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )

    match = re.search(
        r"DEM benchmark seconds:\s*([0-9.eE+-]+)",
        result.stdout,
    )
    if match is None:
        raise RuntimeError("DEM benchmark timing was not printed")

    return float(match.group(1))

run_dem()
run_window()

dem_times = []
times = []

for repeat in range(repeats):
    dem_elapsed = run_dem()
    begin = perf_counter()
    final_state = run_window()
    ml_elapsed = perf_counter() - begin
    dem_times.append(dem_elapsed)
    times.append(ml_elapsed)
    print(
        f"Pair {repeat + 1}: "
        f"DEM={dem_elapsed:.6f}s, ML={ml_elapsed:.6f}s"
    )

print("Median DEM [s]:", median(dem_times))
print("Median ML [s]:", median(times))
print("Speedup [DEM / ML]:", median(dem_times) / median(times))
profile = cProfile.Profile()
profile.runcall(run_window)

profile_path = root / "experiments/summaries/ml_inference_seed44.prof"
profile.dump_stats(str(profile_path))

print("\nLargest total function times:")
pstats.Stats(profile).strip_dirs().sort_stats("cumulative").print_stats(25)

print("\nLargest function self-times:")
pstats.Stats(profile).strip_dirs().sort_stats("tottime").print_stats(25)