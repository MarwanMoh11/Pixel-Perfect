import argparse
import os
import random
import zipfile
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt


def valid_image_name(name):
    lower = name.lower()
    return (
        lower.endswith((".jpeg", ".jpg", ".png"))
        and "__macosx" not in lower
        and not os.path.basename(name).startswith("._")
    )


def load_samples_from_zip(zip_path, num_samples, seed):
    with zipfile.ZipFile(zip_path) as zf:
        image_names = [name for name in zf.namelist() if valid_image_name(name)]
        rng = random.Random(seed)
        selected = rng.sample(image_names, num_samples)

        samples = []
        for name in selected:
            with zf.open(name) as f:
                img = Image.open(f).convert("RGB")
                samples.append((Path(name).stem, img.copy()))
        return samples


def load_samples_from_folder(folder_path, num_samples, seed):
    folder = Path(folder_path)
    image_paths = [p for p in folder.rglob("*") if p.is_file() and valid_image_name(str(p))]
    rng = random.Random(seed)
    selected = rng.sample(image_paths, num_samples)

    samples = []
    for path in selected:
        img = Image.open(path).convert("RGB")
        samples.append((path.stem, img.copy()))
    return samples


def preprocess_sprite(img, scale=2):
    w, h = img.size
    new_w = w - (w % scale)
    new_h = h - (h % scale)

    left = (w - new_w) // 2
    top = (h - new_h) // 2

    hr = img.crop((left, top, left + new_w, top + new_h))
    lr = hr.resize((new_w // scale, new_h // scale), resample=Image.Resampling.NEAREST)
    lr_resize_back = lr.resize((new_w, new_h), resample=Image.Resampling.NEAREST)

    return hr, lr, lr_resize_back


def build_figure(image_source, output_path, num_samples=8, seed=42):
    if str(image_source).lower().endswith(".zip"):
        samples = load_samples_from_zip(image_source, num_samples, seed)
    else:
        samples = load_samples_from_folder(image_source, num_samples, seed)

    fig, axes = plt.subplots(len(samples), 3, figsize=(8, 2 * len(samples)))

    for row_idx, (name, img) in enumerate(samples):
        hr, lr, lr_resize_back = preprocess_sprite(img, scale=2)

        hr_big = hr.resize((hr.width * 8, hr.height * 8), resample=Image.Resampling.NEAREST)
        lr_big = lr.resize((hr.width * 8, hr.height * 8), resample=Image.Resampling.NEAREST)
        lrb_big = lr_resize_back.resize((lr_resize_back.width * 8, lr_resize_back.height * 8), resample=Image.Resampling.NEAREST)

        images = [
            (hr_big, "Raw HR"),
            (lr_big, "NN Downsampled (8×8)"),
            (lrb_big, "Processed LR (resize-back)")
        ]

        for col_idx, (show_img, title) in enumerate(images):
            ax = axes[row_idx, col_idx]
            ax.imshow(show_img)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(title, fontsize=12)

        axes[row_idx, 0].set_ylabel(name.replace("image_", "img "), rotation=0, labelpad=30, va="center", fontsize=9)

    fig.suptitle("Sample Sprites Before and After Preprocessing", fontsize=16)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_source",
        required=True,
        help="Path to the extracted images folder or the image ZIP file."
    )
    parser.add_argument(
        "--output_path",
        default="dataset_analysis_outputs/sample_visualizations_figure.png",
        help="Path to save the figure."
    )
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_figure(
        image_source=args.image_source,
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed
    )