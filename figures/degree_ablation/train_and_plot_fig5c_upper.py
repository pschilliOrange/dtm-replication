import os
import csv
import matplotlib.pyplot as plt
import re
from collections import defaultdict

from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg

"""
Trains a grid of 4-step DTM models with ACP (adaptive_threshold=0.016) on binarized Fashion-MNIST and produces the
upper panel of Figure 5c: performance (FID^{-1}) vs. fraction of visible nodes, sweeping graph degree and grid size.

Training setup:
    -20 epochs with a relatively high learning rate (0.022) to approach peak performance within this budget.
    -Grid side lengths: [44, 50, 60, 70, 80, 90]; total nodes per model = (grid_side_len)^2.
    -Degrees: [8, 12, 16, 20] (average per-node degree). Higher degree generally improves performance.
    -Visible nodes are fixed: 784 image + (10 label spots × 10 one-hot) = 884. The visible-node fraction is 884 / (grid_side_len^2).
    -Sampler schedule: steps_per_sample=8, steps_warmup=400, num_samples=50; generation_warmup=800.

What gets plotted:
    -For each (degree, grid size) run, we take the best performance over the 20 epochs from either free or clamped
        generation (i.e., highest 1 / FID).
    -Results are grouped by degree and plotted against the fraction of visible nodes (color-coded per degree).

Analysis:
    -Increasing degree adds more local/contextual constraints, so each pixel doesn’t have to infer its value from a very
        thin neighborhood. Richer connectivity propagates information more effectively, which likely explains the strong
        degree–performance correlation observed here.
    -We again see that the optimal data node percentage is right around 25% as in the lower half of figure 5c.

Implementation notes:
    -Each run saves a CSV named DTM_ACP_degree_{degree}_grid_{side}.csv containing per-epoch autocorrelations and FIDs.
    -The plotting script parses those files, computes the visible-node fraction, selects the best 1 / FID per run, and writes
        degree_ablation.png.
"""

n_epochs = 20

DEGREES = [8, 12, 16, 20]
GRID_SIDE_LENS = [44, 50, 60, 70, 80, 90]

for degree in DEGREES:
    for grid_side_len in GRID_SIDE_LENS:
        descript = f"DTM_ACP_degree_{degree}_grid_{grid_side_len}"

        cfg = make_cfg(
            exp=dict(seed=1, descriptor=descript, compute_autocorr=True),
            data=dict(dataset_name="fashion_mnist", target_classes=tuple(range(10))),
            graph=dict(graph_preset_architecture=int(f"{grid_side_len}{degree}"), num_label_spots=10, grayscale_levels=1, torus=True),
            sampling=dict(batch_size=400, n_samples=50, steps_per_sample=8, steps_warmup=400, training_beta=1.0),
            generation=dict(generation_beta_start=0.8, generation_beta_end=1.2, steps_warmup=800),
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
VISIBLE_NODES = 884  # 784 image + 100 label
COLORS = {8: 'green', 12: 'red', 16: 'blue', 20: 'orange'}

def parse_filename(filename):
    # Extract degree and grid_side_len from filename like DTM_ACP_degree_8_grid_44.csv
    degree_match = re.search(r'degree_(\d+)', filename)
    grid_match = re.search(r'grid_(\d+)', filename)
    if degree_match and grid_match:
        degree = int(degree_match.group(1))
        grid_side_len = int(grid_match.group(1))
        if degree in DEGREES and grid_side_len in GRID_SIDE_LENS:
            return degree, grid_side_len
    return None, None

# Collect data: dict of degree -> list of (fraction, performance)
data = defaultdict(list)

# Process all CSV files in current directory
for filename in os.listdir('.'):
    if filename.endswith('.csv'):
        degree, grid_side_len = parse_filename(filename)
        if degree is not None:
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
                data[degree].append((fraction, performance))

# Plotting
plt.figure(figsize=(10, 6))
for degree in DEGREES:
    if degree in data:
        points = sorted(data[degree], key=lambda x: x[0])  # Sort by fraction ascending
        fractions, performances = zip(*points)
        plt.plot(fractions, performances, marker='o', color=COLORS[degree], label=f'Degree {degree}')

plt.xlabel('Fraction of Visible Nodes')
plt.ylabel('Performance [FID^{-1}]')
plt.title('Degree Ablation')
plt.legend()
plt.grid(True)
plt.savefig('degree_ablation.png')
plt.close()
