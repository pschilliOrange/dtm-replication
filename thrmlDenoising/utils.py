import numpy as np
from jaxtyping import Array
from typing import Optional, Sequence
import jax.numpy as jnp
import jax.random as jr
import re
from matplotlib import pyplot as plt
import os
import imageio.v2 as imageio
import jax
import tensorflow_datasets as tfds
from dataclasses import is_dataclass, replace, asdict

from thrmlDenoising.DTM_config import DTMConfig
from thrmlDenoising.smoke_testing import smoke_test_data_dict
from thrml.block_sampling import sample_with_observation

def extend_params_or_zeros(params, num_steps):
    if len(params) == 0:
        return jnp.array([0.0] * num_steps)
    params = list(params)[: num_steps] + [params[-1]] * (
        num_steps - len(params)
    )
    return jnp.array(params)

def batch_sample(
    key,
    init_states: list[Array],
    clamps: list[Array],
    program,
    schedule,
    observer,
):
    """
    Function to perform batched sampling over a batch of examples.
    Assumes init_states is a list of arrays, each shaped (batch_size, ...).
    Assumes clamps is a list of arrays, each shaped (batch_size, ...).
    Returns the samples as a list of arrays, each shaped (batch_size, n_samples, output_dim).
    """
    batch_size = clamps[0].shape[0]
    sample_keys = jr.split(key, batch_size)
    obs_init = observer.init()

    def single_sample(k, init_state, clamp):
        return sample_with_observation(
            k,
            program,
            schedule,
            init_state,
            clamp,
            obs_init,
            observer,
        )
    # The sample_with_observation returns n_samples as leading dim
    _, samples = jax.vmap(single_sample)(sample_keys, init_states, clamps)
    for s in samples:
        assert s.shape[:2] == (batch_size, schedule.n_samples)
    return samples

def one_hot_repeated_from_ints(
    labels: jnp.ndarray,
    num_classes: int,
    num_label_spots: int
) -> jnp.ndarray:
    """
    Turn integer labels (N,) into repeated one-hot vectors (N, num_classes*num_label_spots).
    """
    labels = jnp.asarray(labels).astype(jnp.int32)           # (N,)
    base = (labels[:, None] == jnp.arange(num_classes)[None, :])  # (N, K) bool
    base = base.astype(jnp.bool_)                             # ensure jnp.bool_
    return jnp.tile(base, (1, int(num_label_spots)))  
    
def one_hot(x, digits, num_label_spots, dtype=jnp.int32):
    digits = jnp.array(digits)
    one_hot = jnp.array(x[:, None] == digits, dtype)
    one_hot_repeated = jnp.concatenate([one_hot] * num_label_spots, axis=-1)
    return jnp.array(one_hot_repeated, dtype=jnp.bool)

def load_dataset(
    dataset_name: str,
    n_grayscale_levels: int,
    target_classes: Sequence[int],
    num_label_spots: int,
    threshold: float = 0.1,
    max_samples: Optional[int] = None,
):
    # If smoke_testing load from smoke testing data dictionary.
    #   Smoke test data dictionary keys are hackily appended to the dataset name set in config
    #   which should be handed to load_dataset like "smoke_testing_{n_image_pixels}_{n_grayscale_levels}_{n_target_classes}"
    if (m := re.fullmatch(r"smoke_testing_(\d+)_(\d+)_(\d+)", dataset_name)):
        n1, n2, n3 = map(int, m.groups())
        if (n1, n2, n3) not in smoke_test_data_dict:
            raise ValueError(
                f"Smoke test data key ({n1}, {n2}, {n3}) specified by datset name {dataset_name} not found in smoke_test_data_dict. "
            )
        target_classes=tuple(range(n3))
        train_data = smoke_test_data_dict[(n1, n2, n3)]
        label_train_data = one_hot(train_data["label"], target_classes, num_label_spots)
        new_train_data = {"image": train_data["image"], "label": label_train_data}
        return new_train_data, {}, one_hot(jnp.array(target_classes), target_classes, num_label_spots)
    elif "smoke_testing" in dataset_name:
        raise ValueError("Smoke testing passed as dataset with invalid appended key.")
    else:
        data = tfds.load(
            name=dataset_name, batch_size=-1, data_dir="datasets/"
        )
        data = tfds.as_numpy(data)

    def process_images(data):
        # Select only the target classes
        if target_classes is not None:
            mask = np.squeeze(np.isin(data["label"], target_classes))
            data = {k: v[mask] for k, v in data.items()}

        images = data["image"]
        images = np.asarray(images, dtype=np.float32) / 255.0

        if n_grayscale_levels == 1:
            images = np.asarray(images > threshold, dtype=bool)
        else:
            images = jnp.astype(
                np.rint(images * n_grayscale_levels), np.min_scalar_type(n_grayscale_levels) if n_grayscale_levels > 1 else jnp.bool
            )

        images = images.reshape(images.shape[0], -1)
        if max_samples is not None:
            images = images[:max_samples]
            labels = data["label"][:max_samples]
        else:
            labels = data["label"]
        labels = one_hot(labels, target_classes, num_label_spots)
        assert labels.shape == (images.shape[0], num_label_spots * len(target_classes)), (
            f"Labels shape {labels.shape} does not match expected shape {(images.shape[0], num_label_spots * len(target_classes))}"
        )
        return {"image": jnp.array(images), "label": jnp.array(labels)}

    data = {k: process_images(v) for k, v in data.items()}
    # Only keep 1000 test images and labels
    data["test"] = {k: v[:1000] for k, v in data["test"].items()}
    return data["train"], data["test"], one_hot(jnp.array(target_classes), target_classes, num_label_spots)

def adapt_param(ac, prev_ac, param, correlation_threshold):
    if ac < correlation_threshold:
        param *= 0.6 + 0.2 * (ac / correlation_threshold)
    elif ac > prev_ac:
        param *= 1.2 if ac < 1.8 * correlation_threshold else 1.5
    else:
        param *= 0.95
    return param

def _tuples_to_lists(x):
    if isinstance(x, dict):
        return {k: _tuples_to_lists(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return [_tuples_to_lists(v) for v in x]   # list, not tuple
    if isinstance(x, list):
        return [_tuples_to_lists(v) for v in x]
    return x

def _lists_to_tuples(x):
    if isinstance(x, dict):
        return {k: _lists_to_tuples(v) for k, v in x.items()}
    if isinstance(x, list):
        return tuple(_lists_to_tuples(v) for v in x)
    return x

def config_to_yaml_dict(cfg):
    return _tuples_to_lists(asdict(cfg))

def load_yaml_config_from_dict(base_cfg, data):
    normalized = _lists_to_tuples(data)
    cfg = base_cfg
    for section, values in normalized.items():
        if not hasattr(cfg, section):
            raise ValueError("Section in data that is not present in dtm config.")
        cur = getattr(cfg, section)
        if is_dataclass(cur) and isinstance(values, dict):
            cfg = replace(cfg, **{section: replace(cur, **values)})
        else:
            cfg = replace(cfg, **{section: values})
    return cfg

def make_cfg(**overrides) -> DTMConfig:
    cfg = DTMConfig()
    for section, vals in overrides.items():
        cfg = replace(cfg, **{section: replace(getattr(cfg, section), **vals)})
    return cfg

def write(text: str, log_path: Optional[str] = None):
    if log_path is not None:
        with open(log_path, "a") as log:
            log.write(text)
    print(text, end="", flush=True)

def draw_single_image(image, ax, image_side_len):
    assert image.shape == (image_side_len * image_side_len,)
    image = image.reshape(image_side_len, image_side_len)
    ax.imshow(image, cmap="gray")
    ax.axis("off")

def draw_image_batch(
    images: Array,
    h: int,
    w: int,
    super_columns: Optional[int] = None,
    title: Optional[str] = None,
    image_side_len: int = 28,
):
    assert images.shape == (
        h * w,
        (image_side_len**2),
    ), f"{images.shape} != ({h} * {w}, {image_side_len}**2)"
    if super_columns is None:
        super_columns = 1
    assert h % super_columns == 0, f"{h} % {super_columns} != 0"
    rows = h // super_columns
    cols = w * super_columns

    fig, axs = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for i in range(h * w):
        y = i // w
        x = i % w
        super_col = y // rows
        row = y % rows
        col = x + super_col * w
        draw_single_image(images[i], axs[row, col], image_side_len)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    return fig

def denoise_arrays_to_gif(
    image_readout_list: list[np.ndarray],
    out_path: str,
    *,
    n_grayscale_levels: int,
    runs_per_label: int,
    frame_stride: int,
    fps: int,
    image_side_len: int,
    pad: int = 1,
    label_readout_list: list[np.ndarray] | None = None,   # list over steps; each (n_labels, rp_avail, n_samples, label_size)
    enable_label_bars: bool = False,
    steps_per_sample: int = 1,
):
    """
    Rows = steps (noisiest -> least noisy). Columns = images ordered by label-major
    (all images for class 0, then class 1, ...), with `runs_per_label` per class.

    Animation:
      * Only one row evolves at a time. Finished rows are frozen at their final output.
        Not-yet-started rows are all-black (images) with baseline-only bars.
      * Left gutter shows an updating counter "s/N" for each row.
      * Optional per-image class bars (1 px per class) to the right of each image,
        height up to 27 px, always with a 1-px white baseline (even at 0%).
        Bars update each frame like the images do.

    Args
      image_readout_list: list over steps; each array (n_labels, runs_per_label_avail, n_samples, image_size)
      out_path: gif path
      n_grayscale_levels: grayscale levels to normalize image pixels to [0,1]
      runs_per_label: columns per class to display (<= available runs)
      frame_stride: keep 1 of every `frame_stride` samples
      fps: animation fps
      image_side_len: sqrt(image_size)
      pad: vertical pixels between step rows (>=1 recommended)
      label_readout_list: (optional) list over steps; each (n_labels, runs_per_label_avail, n_samples, label_size)
      enable_label_bars: whether to draw bars (default False; set True for clamped)
    """
    assert len(image_readout_list) > 0, "empty image_readout_list"
    n_steps = len(image_readout_list)
    n_labels, rp_avail, n_samples, image_size = image_readout_list[0].shape
    assert rp_avail >= runs_per_label, "runs_per_label exceeds available runs"
    assert image_side_len * image_side_len == image_size, "image_size must be a perfect square"

    use_bars = enable_label_bars and (label_readout_list is not None)
    if use_bars:
        assert len(label_readout_list) == n_steps
        assert label_readout_list[0].shape[0] == n_labels
        label_size = label_readout_list[0].shape[-1]
        assert label_size % n_labels == 0, "label_size must be multiple of n_labels"
        num_label_spots = label_size // n_labels
    else:
        num_label_spots = 0

    # ---- layout constants ----
    spacer_img_to_bars = 2           # px between image and bars
    spacer_after_bars = 6            # px after bars before next column
    bar_height_px = min(27, image_side_len)   # max bar height
    baseline_white = 1               # keep 1px white baseline at bottom
    bars_width = n_labels if use_bars else 0  # 1 px per class

    tile_w = image_side_len + (spacer_img_to_bars + bars_width + spacer_after_bars if use_bars else spacer_after_bars)
    n_cols = n_labels * runs_per_label

    # ---- simple 3x5 pixel font for counters, scaled to row height ----
    FONT_3x5 = {
        "0": ["111",
              "101",
              "101",
              "101",
              "111"],
        "1": ["010",
              "110",
              "010",
              "010",
              "111"],
        "2": ["111",
              "001",
              "111",
              "100",
              "111"],
        "3": ["111",
              "001",
              "111",
              "001",
              "111"],
        "4": ["101",
              "101",
              "111",
              "001",
              "001"],
        "5": ["111",
              "100",
              "111",
              "001",
              "111"],
        "6": ["111",
              "100",
              "111",
              "101",
              "111"],
        "7": ["111",
              "001",
              "010",
              "010",
              "010"],
        "8": ["111",
              "101",
              "111",
              "101",
              "111"],
        "9": ["111",
              "101",
              "111",
              "001",
              "111"],
        "/": ["001",
              "001",
              "010",
              "100",
              "100"],
        " ": ["000",
              "000",
              "000",
              "000",
              "000"],
    }
    def _render_counter_bitmap(text: str, H: int) -> np.ndarray:
        # scale factor (fit height as close as possible, center vertically)
        base_h, base_w = 5, 3
        s = max(1, H // base_h)
        glyph_h = base_h * s
        glyph_w = base_w * s
        gap = max(1, s // 1)  # inter-char spacing

        # compose base bitmap (unscaled) row-major
        bitmaps = []
        for ch in text:
            patt = FONT_3x5.get(ch, FONT_3x5[" "])
            g = np.array([[1.0 if c == "1" else 0.0 for c in row] for row in patt], dtype=np.float32)
            # upscale with nearest-neighbor
            g_up = np.kron(g, np.ones((s, s), dtype=np.float32))
            bitmaps.append(g_up)
        if len(bitmaps) == 0:
            canvas = np.zeros((H, 1), dtype=np.float32)
            return canvas

        # place with 1 glyph gap between chars
        total_w = len(bitmaps) * glyph_w + (len(bitmaps) - 1) * gap
        canvas = np.zeros((H, total_w), dtype=np.float32)
        y0 = (H - glyph_h) // 2
        x = 0
        for i, g in enumerate(bitmaps):
            h, w = g.shape
            canvas[y0:y0 + h, x:x + w] = np.maximum(canvas[y0:y0 + h, x:x + w], g)
            x += w
            if i < len(bitmaps) - 1:
                x += gap
        return canvas
    
    total_steps = n_samples * max(1, int(steps_per_sample))

    # max width needed for left gutter: "N/N" with full digits
    sample_text = f"{total_steps}/{total_steps}"
    # approximate width from renderer
    left_gutter = _render_counter_bitmap(sample_text, image_side_len)
    left_w = left_gutter.shape[1]

    # ---- helpers ----
    def _norm_img(x: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=np.float32) / float(n_grayscale_levels), 0.0, 1.0)

    def _img_tile(step_arr: np.ndarray, lbl_idx: int, run_idx: int, s: int) -> np.ndarray:
        flat = step_arr[lbl_idx, run_idx, s, :]  # (image_size,)
        tile = _norm_img(flat).reshape(image_side_len, image_side_len)
        return tile

    def _bars_tile(label_step_arr: np.ndarray, lbl_idx: int, run_idx: int, s: int) -> np.ndarray:
        """
        Returns (image_side_len, n_labels) with 1px-wide bars per class.
        Bottom row is always white (baseline).
        """
        if not use_bars:
            return None
        vec = np.asarray(label_step_arr[lbl_idx, run_idx, s, :], dtype=np.int32)  # (label_size,)
        bars = np.zeros((image_side_len, n_labels), dtype=np.float32)
        # ensure baseline white at the bottom
        bars[-1, :] = 1.0
        # fill by class
        for cls in range(n_labels):
            # indices for this class across repeated spots: cls, cls+n_labels, ...
            idxs = np.arange(cls, cls + n_labels * num_label_spots, n_labels, dtype=np.int32)
            # percentage in [0,1]
            p = float(np.sum(vec[idxs] != 0)) / float(num_label_spots)
            fill = baseline_white + int(round(p * (bar_height_px - baseline_white)))
            # fill from bottom up
            y_bottom = image_side_len - 1
            y_top = max(0, y_bottom - (fill - 1))
            bars[y_top:y_bottom + 1, cls] = 1.0
        return bars

    # precompute final (frozen) tiles/rows per step for speed
    final_images = image_readout_list  # alias
    final_rows_images = []  # list of (n_cols, image_side_len, image_side_len)
    final_rows_bars = []    # list of (n_cols, image_side_len, n_labels) or None
    for k in range(n_steps):
        imgs_k = []
        bars_k = []
        for col in range(n_cols):
            lbl_idx = col // runs_per_label
            run_idx = col % runs_per_label
            imgs_k.append(_img_tile(final_images[k], lbl_idx, run_idx, n_samples - 1))
            if use_bars:
                bars_k.append(_bars_tile(label_readout_list[k], lbl_idx, run_idx, n_samples - 1))
        final_rows_images.append(imgs_k)
        final_rows_bars.append(bars_k if use_bars else None)

    # canvas size
    frame_h = n_steps * image_side_len + pad * (n_steps - 1)
    frame_w = left_w + n_cols * tile_w - spacer_after_bars  # no trailing spacer after last col

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    stride = max(1, int(frame_stride))

    with imageio.get_writer(out_path, mode="I", fps=fps, loop=0) as writer:
        # animate each step k
        for k in range(n_steps):
            for s in range(0, n_samples, stride):
                frame = np.zeros((frame_h, frame_w), dtype=np.float32)

                # draw each row r
                for r in range(n_steps):
                    y0 = r * (image_side_len + pad)

                    # --- 1) left counter ---
                    if r < k:
                        cur_steps = total_steps
                    elif r == k:
                        cur_steps = min((s + 1) * steps_per_sample, total_steps)
                    else:
                        cur_steps = 0
                    txt = f"{cur_steps}/{total_steps}" 
                    bmp = _render_counter_bitmap(txt, image_side_len)
                    frame[y0:y0 + image_side_len, 0:bmp.shape[1]] = np.maximum(
                        frame[y0:y0 + image_side_len, 0:bmp.shape[1]],
                        bmp
                    )

                    # --- 2) columns (images + bars) ---
                    for col in range(n_cols):
                        x0 = left_w + col * tile_w
                        lbl_idx = col // runs_per_label
                        run_idx = col % runs_per_label

                        if r < k:
                            # frozen final
                            img_tile = final_rows_images[r][col]
                            frame[y0:y0 + image_side_len, x0:x0 + image_side_len] = img_tile
                            if use_bars:
                                bars = final_rows_bars[r][col]
                                xb = x0 + image_side_len + spacer_img_to_bars
                                frame[y0:y0 + image_side_len, xb:xb + n_labels] = np.maximum(
                                    frame[y0:y0 + image_side_len, xb:xb + n_labels], bars
                                )
                        elif r == k:
                            # current evolving sample
                            img_tile = _img_tile(image_readout_list[r], lbl_idx, run_idx, s)
                            frame[y0:y0 + image_side_len, x0:x0 + image_side_len] = img_tile
                            if use_bars:
                                bars = _bars_tile(label_readout_list[r], lbl_idx, run_idx, s)
                                xb = x0 + image_side_len + spacer_img_to_bars
                                frame[y0:y0 + image_side_len, xb:xb + n_labels] = np.maximum(
                                    frame[y0:y0 + image_side_len, xb:xb + n_labels], bars
                                )
                        else:
                            # not started: black image; bars baseline only if enabled
                            if use_bars:
                                # draw baseline bars (just bottom white row)
                                xb = x0 + image_side_len + spacer_img_to_bars
                                frame[y0 + image_side_len - 1, xb:xb + n_labels] = 1.0
                            # image area remains black

                writer.append_data((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))

        # final frame (everything frozen complete)
        frame = np.zeros((frame_h, frame_w), dtype=np.float32)
        for r in range(n_steps):
            y0 = r * (image_side_len + pad)
            # counter at N/N
            bmp = _render_counter_bitmap(f"{total_steps}/{total_steps}", image_side_len)
            frame[y0:y0 + image_side_len, 0:bmp.shape[1]] = np.maximum(
                frame[y0:y0 + image_side_len, 0:bmp.shape[1]], bmp
            )
            for col in range(n_cols):
                x0 = left_w + col * tile_w
                frame[y0:y0 + image_side_len, x0:x0 + image_side_len] = final_rows_images[r][col]
                if use_bars:
                    xb = x0 + image_side_len + spacer_img_to_bars
                    frame[y0:y0 + image_side_len, xb:xb + n_labels] = np.maximum(
                        frame[y0:y0 + image_side_len, xb:xb + n_labels], final_rows_bars[r][col]
                    )
        writer.append_data((np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8))
