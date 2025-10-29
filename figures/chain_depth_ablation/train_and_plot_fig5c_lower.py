from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
import os
import csv
import matplotlib.pyplot as plt
import re
from collections import defaultdict

"""
Trains a grid of 4-step DTM models with ACP (adaptive_threshold=0.016) and produces the lower panel of Figure 5c:
performance (FID^{-1}) vs. fraction of visible nodes, sweeping grid size and sampler schedule.

Training setup:
    -20 epochs with a relatively high learning rate (0.022) to approach peak performance within this budget.
    -Grid side lengths: [44, 50, 60, 70, 80, 90]; total nodes per model = (grid_side_len)^2.
    -Fixed graph degree = 12. At this degree, longer warmups tend to help slightly, but gains are modest.
    -Visible nodes are fixed: 784 image + (10 label spots × 10 one-hot) = 884. The visible-node fraction is 884 / (grid_side_len^2).
    -Trained on binarized fashion mnist.

Sampler schedule sweep:
    -Training warmup ∈ {400, 800, 1200}; steps_per_sample ∈ {8, 16, 24} (scaled proportionally).
    -num_samples = 50 for all runs.
    -Generation warmup = 2 × training warmup (i.e., {800, 1600, 2400}), mirroring “sample at the end of training” behavior.

What gets plotted:
    -For each (grid size, schedule) run, we take the best performance over the 20 epochs from either free or clamped
        generation (i.e., highest 1 / FID).
    -Results are aggregated by warmup and plotted against the fraction of visible nodes (color-coded: 400→green, 800→orange, 1200→blue).

Key takeaway:

    -With these 4-step DTM+ACP models, peak performance typically appears near ~25% visible nodes, though the exact optimum
        can shift with model scale. Intuitively, this ratio lets the latent field “hear” the data clearly while remaining large
        enough to interact expressively with it.

Implementation notes:

    -Each run saves a CSV named DTM_ACP_warmup_{warmup}_grid_{side}.csv containing per-epoch autocorrelations and FIDs.
    -The plotting script parses those files, computes the visible-node fraction, selects the best 1 / FID per run, and writes
        chain_depth_ablation.png.
"""

n_epochs = 20

GRID_SIDE_LENS = [44, 50, 60, 70, 80, 90]
SCHEDULE_FACTORS = [1, 2, 3]

for grid_side_len in GRID_SIDE_LENS:
    for schedule_factor in SCHEDULE_FACTORS:
        training_warmup = schedule_factor * 400
        training_steps_per_sample= schedule_factor * 8
        generation_warmup = schedule_factor * 800


        descript = f"DTM_ACP_warmup_{training_warmup}_grid_{grid_side_len}"

        cfg = make_cfg(
            exp=dict(seed=1, descriptor=descript),
            data=dict(dataset_name="fashion_mnist", target_classes=tuple(range(10))),
            graph=dict(graph_preset_architecture=int(f"{grid_side_len}12"), num_label_spots=10, grayscale_levels=1, torus=True),
            sampling=dict(batch_size=400, n_samples=50, steps_per_sample=training_steps_per_sample, steps_warmup=training_warmup, training_beta=1.0),
            generation=dict(generation_beta_start=0.8, generation_beta_end=1.2, steps_warmup=generation_warmup),
            diffusion_schedule=dict(num_diffusion_steps=4, kind="log", diffusion_offset=0.15),
            diffusion_rates=dict(image_rate=0.9, label_rate=0.2),
            optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(0.022,), n_epochs_for_lrd=20, alpha_cosine_decay=0.2),
            cp=dict(adaptive_cp=True, adaptive_threshold=0.016),
        )

        dtm = DTM(cfg)
        dtm.train(n_epochs, 1)

        # Save results to CSV
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_filename = os.path.join(script_dir, descript + ".csv")
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['epoch'] + [f'autocorr_step_{i}' for i in range(dtm.cfg.diffusion_schedule.num_diffusion_steps)] + ['free_fid', 'clamped_fid']
            writer.writerow(header)
            for epoch in range(0, n_epochs + 1):
                autocorrs = [step.autocorrelations[epoch] for step in dtm.steps]
                free_fid = dtm.fids_dict['free'][epoch]
                clamped_fid = dtm.fids_dict['clamped'][epoch]
                row = [epoch] + autocorrs + [free_fid, clamped_fid]
                writer.writerow(row)

# Hardcoded constants
VISIBLE_NODES = 884  # 784 image + 100 label (100 from 10 label spots * 10 labels)
WARMUPS = [s * 400 for s in SCHEDULE_FACTORS] 
COLORS = {400: 'green', 800: 'orange', 1200: 'blue'}

def parse_filename(filename):
    # Extract warmup and grid_side_len from filename like DTM_ACP_warmup_400_grid_44.csv
    warmup_match = re.search(r'warmup_(\d+)', filename)
    grid_match = re.search(r'grid_(\d+)', filename)
    if warmup_match and grid_match:
        warmup = int(warmup_match.group(1))
        grid_side_len = int(grid_match.group(1))
        if warmup in WARMUPS and grid_side_len in GRID_SIDE_LENS:
            return warmup, grid_side_len
    return None, None

# Collect data: dict of warmup -> list of (fraction, performance)
data = defaultdict(list)

# Process all CSV files in current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        warmup, grid_side_len = parse_filename(filename)
        if warmup is not None:
            # Read CSV to find the best (lowest) FID
            min_fid = float('inf')
            with open(filename, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        free_fid = float(row[-2]) if row[-2] else float('inf')
                        clamped_fid = float(row[-1]) if row[-1] else float('inf')
                        min_fid = min(min_fid, free_fid, clamped_fid)
            
            if min_fid != float('inf'):
                performance = 1 / min_fid
                fraction = VISIBLE_NODES / (grid_side_len ** 2)
                data[warmup].append((fraction, performance))

# Plotting
plt.figure(figsize=(10, 6))
for warmup in WARMUPS:
    if warmup in data:
        points = sorted(data[warmup], key=lambda x: x[0])  # Sort by fraction ascending
        fractions, performances = zip(*points)
        plt.plot(fractions, performances, marker='o', color=COLORS[warmup], label=f'Warmup {warmup}')

plt.xlabel('Fraction of Visible Nodes')
plt.ylabel('Performance [FID^{-1}]')
plt.title('Chain Depth Ablation')
plt.legend()
plt.grid(True)
plt.savefig('chain_depth_ablation.png')
plt.close()
