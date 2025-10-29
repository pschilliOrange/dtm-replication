from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
import os
import csv
import matplotlib.pyplot as plt
import numpy as np

"""
This script trains and plots the three models shown in Figure 5b (MEBM, DTM, and DTM+ACP) on Fashion-MNIST,
reproducing the training-stability curves: 1 / Free FID (top) and the maximum per-step autocorrelation (bottom)
as functions of epoch.

Only the green DTM+ACP curve uses the adaptive correlation penalty (ACP). Here we set adaptive_threshold=0.008,
which means the controller penalizes the loss whenever the measured autocorrelation at the CD lag exceeds 0.008.
In practice, the DTM+ACP stays below this threshold, as reflected in the autocorrelation plot.

Note on measurement: this codebase projects the image state into a lower-dimensional vector using a fixed random
projection before computing vector autocorrelation. The original Figure 5b used a different dimensionality-reduction
procedure, so absolute autocorrelation values may differ slightly; the qualitative trends match.

ACP is such an impressive regularizer because, while it lets the model form useful energy minima around data (yielding
    strong samples with FIDs well below 30), it keeps those minima extremely shallow so that the model can quickly jump
    out of the minima, driving the measured autocorrelation to roughly ~1/400 of what it would be without ACP!
"""

n_epochs = 150

# Hardcoded bits per run (descriptor, csv name, diffusion steps, cp settings)
specs = {
    "MEBM": dict(
        descriptor="MEBM_1",
        num_steps=1,
        cp=dict(adaptive_cp=False, adaptive_threshold=0.0),
    ),
    "DTM": dict(
        descriptor="DTM_1",
        num_steps=8,
        cp=dict(adaptive_cp=False, adaptive_threshold=0.0), # adaptive thresholds for models without ACP are not relevant
    ),
    "DTM_ACP": dict(
        descriptor="DTM_ACP_1",
        num_steps=8,
        cp=dict(adaptive_cp=True, adaptive_threshold=0.008),
    ),
}

descriptions = ["MEBM", "DTM", "DTM_ACP"]

for name in descriptions:
    s = specs[name]

    # ---- build cfg  ----
    cfg = make_cfg(
        exp=dict(seed=1, descriptor=s["descriptor"], compute_autocorr=True),
        data=dict(dataset_name="fashion_mnist", target_classes=tuple(range(10))),
        graph=dict(graph_preset_architecture=7012, num_label_spots=10, grayscale_levels=1, torus=True),
        sampling=dict(batch_size=400, n_samples=50, steps_per_sample=8, steps_warmup=400, training_beta=1.0),
        generation=dict(generation_beta_start=0.8, generation_beta_end=1.2, steps_warmup=800),
        diffusion_schedule=dict(num_diffusion_steps=s["num_steps"], kind="log", diffusion_offset=0.15),
        diffusion_rates=dict(image_rate=0.9, label_rate=0.2),
        optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(0.016,), n_epochs_for_lrd=20, alpha_cosine_decay=0.18),
        cp=s["cp"],
    )

    # ---- train ----
    dtm = DTM(cfg)
    dtm.train(n_epochs, 1)

    # ---- write CSV ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filename = os.path.join(script_dir, s["descriptor"] + ".csv")
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

# then plot from the csv's

def read_csv(file_path):
    data = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        for row in reader:
            data.append([float(val) for val in row])
    return np.array(data), headers

# Load the CSV files
data_mebm, headers_mebm = read_csv(specs["MEBM"]["descriptor"] + ".csv")
data_dtm, headers_dtm = read_csv(specs["DTM"]["descriptor"] + ".csv")
data_dtm_acp, headers_dtm_acp = read_csv(specs["DTM_ACP"]["descriptor"] + ".csv")

# Function to get indices
def get_indices(headers, file_name):
    epoch_idx = headers.index('epoch')
    free_fid_idx = headers.index('free_fid')
    autocorr_cols = [h for h in headers if h.startswith('autocorr_step_')]
    autocorr_idxs = [headers.index(h) for h in autocorr_cols]
    return epoch_idx, free_fid_idx, autocorr_idxs

# Get indices for each
epoch_idx_m, free_fid_idx_m, autocorr_idxs_m = get_indices(headers_mebm, specs["MEBM"]["descriptor"] + ".csv")
epoch_idx_d, free_fid_idx_d, autocorr_idxs_d = get_indices(headers_dtm, specs["DTM"]["descriptor"] + ".csv")
epoch_idx_da, free_fid_idx_da, autocorr_idxs_da = get_indices(headers_dtm_acp, specs["DTM_ACP"]["descriptor"] + ".csv")

# Calculate 1 / free_fid
inv_free_fid_mebm = 1 / data_mebm[:, free_fid_idx_m]
inv_free_fid_dtm = 1 / data_dtm[:, free_fid_idx_d]
inv_free_fid_dtm_acp = 1 / data_dtm_acp[:, free_fid_idx_da]

# Calculate max autocorr
max_autocorr_mebm = np.max(data_mebm[:, autocorr_idxs_m], axis=1)
max_autocorr_dtm = np.max(data_dtm[:, autocorr_idxs_d], axis=1)
max_autocorr_dtm_acp = np.max(data_dtm_acp[:, autocorr_idxs_da], axis=1)

# Epochs
epochs_mebm = data_mebm[:, epoch_idx_m]
epochs_dtm = data_dtm[:, epoch_idx_d]
epochs_dtm_acp = data_dtm_acp[:, epoch_idx_da]

# Create subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# First plot: 1 / free_fid
axs[0].plot(epochs_mebm, inv_free_fid_mebm, color='blue', label='MEBM')
axs[0].plot(epochs_dtm, inv_free_fid_dtm, color='orange', label='DTM')
axs[0].plot(epochs_dtm_acp, inv_free_fid_dtm_acp, color='green', label='DTM_ACP')
axs[0].set_ylabel('1 / Free FID')
axs[0].set_title('1 / Free FID vs Epoch')
axs[0].legend()
axs[0].grid(True)

# Second plot: max autocorr
axs[1].plot(epochs_mebm, max_autocorr_mebm, color='blue', label='MEBM')
axs[1].plot(epochs_dtm, max_autocorr_dtm, color='orange', label='DTM')
axs[1].plot(epochs_dtm_acp, max_autocorr_dtm_acp, color='green', label='DTM_ACP')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Max Autocorr')
axs[1].set_title('Max Autocorr vs Epoch')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.savefig('fig_5b.png')