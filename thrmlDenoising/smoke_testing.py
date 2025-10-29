import numpy as np
from typing import Optional
import jax.numpy as jnp
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
import os
import jax.tree_util as jtu

# Keys for smoke test data dict are (n_image_pixels, n_grayscale_levels, n_target_classes)
# To make your own smoke testing dataset, just add the key, 'images' and labels to this dictionary, 
#   and make sure to create the config with the right number of grayscale levels
smoke_test_data_dict = {(2, 2, 2): {"image": jnp.array([[0, 0], [2, 2]], dtype=jnp.uint8),
                                 "label": jnp.array([0,         1], dtype=jnp.int32)},

                        (3, 1, 3): {"image": jnp.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=jnp.bool_),
                                 "label": jnp.array([0,            1,         2], dtype=jnp.int32)},

                        (3, 2, 5): {"image": jnp.array([[0, 0, 0], [0, 0, 2], [0, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=jnp.bool_),
                                 "label": jnp.array([0,            1,         2,         3,         4], dtype=jnp.int32)},

                        (4, 1, 4): {"image": jnp.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=jnp.bool_),
                                 "label": jnp.array([0,               1,            2,            3], dtype=jnp.int32)},

                        (4, 4, 4): {"image": jnp.array([[0, 0, 0, 0], [2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0]], dtype=jnp.uint8),
                                 "label": jnp.array([0,               1,            2,            3], dtype=jnp.int32)}}


def _label_percentages(vec: jnp.ndarray, num_classes: int, num_spots: int) -> jnp.ndarray:
    """Return per-class percentages across all spots; shape (num_classes,), values in [0,1]."""
    v = jnp.asarray(vec).reshape(num_spots, num_classes)
    return v.astype(jnp.float32).mean(axis=0)  # votes per class / num_spots

def _pred_class_from_vec(vec: jnp.ndarray, num_classes: int, num_spots: int) -> int:
    """Argmax over the per-class percentages."""
    pcts = _label_percentages(vec, num_classes, num_spots)
    return int(jnp.argmax(pcts))

def _nonzero_pct_str(pcts: jnp.ndarray) -> str:
    """Compact string showing only nonzero-percentage classes, like '{1:75% 3:25%}'."""
    idx = jnp.where(pcts > 0)[0].tolist()
    if not idx:
        return "{}"
    parts = [f"{int(i)}:{float(pcts[int(i)])*100:.0f}%" for i in idx]
    return "{" + " ".join(parts) + "}"

def compute_conditional_accuracy(
    dtm, key, batch_size: int, n_print: Optional[int] = 10
) -> float:
    """
    Conditional generation accuracy.
    Printing (if n_print is set): up to n_print examples PER CLASS (by condition),
    showing reference image, reference label percentages (no arrow),
    output label percentages with → predicted class, and output image.
    """
    free = False
    image_samples, label_samples = dtm._run_denoising(key, free, batch_size, dtm.steps[0].generation_spec.schedule)


    n_digits = dtm.one_hot_target_labels.shape[0]
    label_size = dtm.one_hot_target_labels.shape[1]
    image_size = dtm.train_dataset["image"].shape[1]

    def take_last_sample_image(x):
        expected_shape = (n_digits, batch_size, dtm.steps[0].generation_spec.schedule.n_samples, image_size)
        assert x.shape == expected_shape, f"Expected image shape {expected_shape}, got {x.shape}"
        return x[:, :, -1, :]

    def take_last_sample_label(x):
        expected_shape = (n_digits, batch_size, dtm.steps[0].generation_spec.schedule.n_samples, label_size)
        assert x.shape == expected_shape, f"Expected label shape {expected_shape}, got {x.shape}"
        return x[:, :, -1, :]
    
    image_samples = jtu.tree_map(take_last_sample_image, image_samples)
    label_samples = jtu.tree_map(take_last_sample_label, label_samples)

    final_image = image_samples[-1]      # (num_conditions, B, img_len)
    final_label = label_samples[-1]      # (num_conditions, B, lbl_len), clamped to condition

    num_conditions, B = final_image.shape[:2]
    num_classes = dtm.one_hot_target_labels.shape[0]
    num_spots = dtm.cfg.graph.num_label_spots

    # Train lookup: label class -> set(images)
    label_to_images = defaultdict(set)
    for i in range(dtm.train_dataset["image"].shape[0]):
        c = _pred_class_from_vec(dtm.train_dataset["label"][i], num_classes, num_spots)
        label_to_images[c].add(tuple(dtm.train_dataset["image"][i].tolist()))

    # Conditions (the thing _run_denoising vmaps over)
    cond_classes = [
        _pred_class_from_vec(dtm.one_hot_target_labels[i], num_classes, num_spots)
        for i in range(num_conditions)
    ]

    total = num_conditions * B
    correct = 0

    per_class_budget = {c: (0 if n_print is None else int(n_print)) for c in set(cond_classes)}

    for li, cond_c in enumerate(cond_classes):
        true_set = label_to_images.get(cond_c, set())
        for b in range(B):
            gen_img = tuple(final_image[li, b].tolist())
            ok = gen_img in true_set
            correct += int(ok)

            # printing per class (by condition)
            if per_class_budget.get(cond_c, 0) > 0:

                out_pcts = _label_percentages(final_label[li, b], num_classes, num_spots)
                out_pred = int(jnp.argmax(out_pcts))

                mark = "✓" if ok else "✗"
                print(
                    f"[COND] cond={cond_c} | "
                    f"out_label {_nonzero_pct_str(out_pcts)} → {out_pred} | "
                    f"out_img={gen_img} {mark}"
                )
                per_class_budget[cond_c] -= 1

    return 100.0 * correct / total

def compute_free_accuracy(
    dtm, key, batch_size: int, n_print: Optional[int] = 10
) -> float:
    """
    Free generation accuracy.
    Printing (if n_print is set): up to n_print TOTAL examples,
    showing reference image, reference label percentages (no arrow),
    output label percentages with → predicted class, and output image.
    """
    free = True
    image_samples, label_samples = dtm._run_denoising(key, free, batch_size, dtm.steps[0].generation_spec.schedule)


    n_digits = dtm.one_hot_target_labels.shape[0]
    label_size = dtm.one_hot_target_labels.shape[1]
    image_size = dtm.train_dataset["image"].shape[1]

    def take_last_sample_image(x):
        expected_shape = (n_digits, batch_size, dtm.steps[0].generation_spec.schedule.n_samples, image_size)
        assert x.shape == expected_shape, f"Expected image shape {expected_shape}, got {x.shape}"
        return x[:, :, -1, :]

    def take_last_sample_label(x):
        expected_shape = (n_digits, batch_size, dtm.steps[0].generation_spec.schedule.n_samples, label_size)
        assert x.shape == expected_shape, f"Expected label shape {expected_shape}, got {x.shape}"
        return x[:, :, -1, :]
    
    image_samples = jtu.tree_map(take_last_sample_image, image_samples)
    label_samples = jtu.tree_map(take_last_sample_label, label_samples)

    final_image = image_samples[-1][0]   # (B, img_len)
    final_label = label_samples[-1][0]   # (B, lbl_len)

    B = final_image.shape[0]
    num_classes = dtm.one_hot_target_labels.shape[0]
    num_spots   = dtm.cfg.graph.num_label_spots

    # Train lookup: label class -> set(images)
    label_to_images = defaultdict(set)
    for i in range(dtm.train_dataset["image"].shape[0]):
        c = _pred_class_from_vec(dtm.train_dataset["label"][i], num_classes, num_spots)
        label_to_images[c].add(tuple(dtm.train_dataset["image"][i].tolist()))

    correct = 0
    remaining = 0 if n_print is None else int(n_print)

    for b in range(B):
        gen_img = tuple(final_image[b].tolist())
        out_pcts = _label_percentages(final_label[b], num_classes, num_spots)
        pred_c   = int(jnp.argmax(out_pcts))
        ok = gen_img in label_to_images.get(pred_c, set())
        correct += int(ok)

        if remaining > 0:
            mark = "✓" if ok else "✗"
            print(
                f"out_label {_nonzero_pct_str(out_pcts)} → {pred_c} | out_img={gen_img} {mark}"
            )
            remaining -= 1

    return 100.0 * correct / B

def save_accuracy_plot(dtm, filename: str = "accuracy_vs_epoch.png") -> str:
    """
    Save an 'epoch vs accuracy' plot for smoke testing to dtm.model_saving_path.
    Uses dtm.eval_epochs on the x-axis and dtm.fids_dict['free']/['clamped'] for y.
    Returns the path of the saved figure.
    """
    # Guard: nothing to plot yet
    xs = np.asarray(dtm.eval_epochs, dtype=int)
    ys_free = np.asarray(dtm.fids_dict.get("free", []), dtype=float)
    ys_cond = np.asarray(dtm.fids_dict.get("clamped", []), dtype=float)
    if xs.size == 0 or ys_free.size == 0 or ys_cond.size == 0:
        return ""

    # Ensure model_saving_path exists
    model_saving_path = getattr(dtm, "model_saving_path", "") or "."
    os.makedirs(model_saving_path, exist_ok=True)
    out_path = os.path.join(model_saving_path, filename)

    # Plot
    plt.figure()
    plt.plot(xs, ys_free, marker="o", label="free")
    plt.plot(xs, ys_cond, marker="o", label="conditional")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Epoch (smoke testing)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path