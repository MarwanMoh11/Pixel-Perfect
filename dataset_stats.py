import argparse
import os
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


def parse_label(label_str):
    label_str = str(label_str).replace("[", "").replace("]", "").strip()
    return np.array([float(x) for x in label_str.split()], dtype=np.float32)


def valid_image_name(name):
    lower = name.lower()
    return (
        lower.endswith((".jpeg", ".jpg", ".png"))
        and "__macosx" not in lower
        and not os.path.basename(name).startswith("._")
    )


def iter_images_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if valid_image_name(name):
                with zf.open(name) as f:
                    img = Image.open(f).convert("RGB")
                    yield name, img.copy()


def iter_images_from_folder(folder_path):
    folder = Path(folder_path)
    for path in sorted(folder.rglob("*")):
        if path.is_file() and valid_image_name(str(path)):
            img = Image.open(path).convert("RGB")
            yield str(path), img.copy()


def build_figure(csv_path, image_source, output_dir):
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_vectors = np.vstack(df["Label"].apply(parse_label).to_numpy())
    label_counts = label_vectors.sum(axis=0).astype(int)

    size_counter = Counter()
    quantized_color_counter = Counter()

    if str(image_source).lower().endswith(".zip"):
        image_iter = iter_images_from_zip(image_source)
    else:
        image_iter = iter_images_from_folder(image_source)

    total_images = 0
    for _, img in image_iter:
        total_images += 1
        size_counter[img.size] += 1
        arr = np.array(img)
        quantized = (arr // 16) * 16
        quantized_color_counter.update(map(tuple, quantized.reshape(-1, 3)))

    train_count = int(total_images * 0.8)
    val_count = int(total_images * 0.1)
    test_count = total_images - train_count - val_count

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    summary_text = (
        f"Total images: {total_images:,}\n"
        f"Scale factor: 2\n"
        f"Resize-back: True\n"
        f"Train/Val/Test: {train_count:,} / {val_count:,} / {test_count:,}\n"
        f"Classes: {label_vectors.shape[1]}"
    )
    ax0.text(0.02, 0.95, summary_text, va="top", ha="left", fontsize=14, family="monospace")
    ax0.set_title("Dataset Summary", fontsize=16, pad=10)

    ax1 = fig.add_subplot(gs[0, 1])
    size_labels = [f"{w}×{h}" for (w, h), _ in size_counter.items()]
    size_counts = [count for _, count in size_counter.items()]
    ax1.bar(size_labels, size_counts)
    ax1.set_title("Sprite Size Distribution")
    ax1.set_ylabel("Count")

    ax2 = fig.add_subplot(gs[1, 0])
    categories = ["Train", "Validation", "Test"]
    split_counts = [train_count, val_count, test_count]
    ax2.bar(categories, split_counts)
    ax2.set_title("80/10/10 Split Counts")
    ax2.set_ylabel("Images")

    ax3 = fig.add_subplot(gs[1, 1])
    top_colors = quantized_color_counter.most_common(12)
    swatch = np.zeros((1, len(top_colors), 3), dtype=np.uint8)
    for i, (color, _) in enumerate(top_colors):
        swatch[0, i] = np.array(color, dtype=np.uint8)
    ax3.imshow(np.repeat(np.repeat(swatch, 40, axis=0), 40, axis=1), interpolation="nearest")
    ax3.set_title("Dominant Quantized Colors (Top 12)")
    ax3.set_xticks(range(len(top_colors)))
    ax3.set_xticklabels([f"{count:,}" for _, count in top_colors], rotation=45, ha="right", fontsize=8)
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(False)

    fig.suptitle("Dataset Statistics", fontsize=18)
    fig.tight_layout()

    fig_path = output_dir / "dataset_statistics_figure.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "dataset_statistics_summary.csv"
    pd.DataFrame(
        {
            "metric": [
                "total_images",
                "train_count",
                "val_count",
                "test_count",
                "num_classes",
            ],
            "value": [
                total_images,
                train_count,
                val_count,
                test_count,
                label_vectors.shape[1],
            ],
        }
    ).to_csv(summary_path, index=False)

    label_path = output_dir / "label_distribution.csv"
    pd.DataFrame(
        {
            "class_index": list(range(label_vectors.shape[1])),
            "count": label_counts,
        }
    ).to_csv(label_path, index=False)

    print(f"Saved figure to: {fig_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved label distribution to: {label_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to the original dataset CSV.")
    parser.add_argument(
        "--image_source",
        required=True,
        help="Path to the extracted images folder or the image ZIP file."
    )
    parser.add_argument(
        "--output_dir",
        default="dataset_analysis_outputs",
        help="Folder to save the statistics outputs."
    )
    args = parser.parse_args()

    build_figure(args.csv_path, args.image_source, args.output_dir)