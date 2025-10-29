from thrmlDenoising.DTM import DTM
from thrmlDenoising.utils import make_cfg
import os
import csv

"""
This is the script which generated images like those used in figure 5a. The model in the same folder as this script
    was trained using true binomial nodes, where the distribution for each pixel's grayscale levels was truly a 
    binomial distribution. This makes the image noisy as it is not able to create a sharp peak at the pixel's ideal
    grayscale level. Since training this model, this codebase has deprecated true binomial nodes in favor of poisson 
    binomial nodes, which allow the model to create sharp peaks around a grayscale level by allowing for different 
    ising probabilities across sub-trials.
"""

n_epochs = 100

cfg = make_cfg(
    exp=dict(seed=1, descriptor="DTM_ACP_gs"),
    data=dict(dataset_name="fashion_mnist", target_classes=tuple(range(10))),
    graph=dict(graph_preset_architecture=8048, num_label_spots=20, grayscale_levels=7, torus=True),
    sampling=dict(batch_size=400, n_samples=50, steps_per_sample=8, steps_warmup=400, training_beta=1.0),
    generation=dict(generation_beta_start=0.8, generation_beta_end=1.2, steps_warmup=800),
    diffusion_schedule=dict(num_diffusion_steps=8, kind="log", diffusion_offset=0.15),
    diffusion_rates=dict(image_rate=0.9, label_rate=0.2),
    optim=dict(momentum=0.9, b2_adam=0.999, step_learning_rates=(.012,), n_epochs_for_lrd=20, alpha_cosine_decay=0.2),
    cp=dict(adaptive_cp=True, adaptive_threshold=.06),
) 

dtm = DTM(cfg)
dtm.train(n_epochs, 1)

# Save results to CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_filename = os.path.join(script_dir, "DTM_ACP_gs.csv")
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
