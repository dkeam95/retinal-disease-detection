"""
EDA (Exploratory Data Analysis) для DDR Retinal Disease Detection датасета.

Скрипт анализирует датасет диабетической ретинопатии и строит 6 графиков:
  1. Распределение классов по train/valid/test
  2. Imbalance Ratio - naskolko datset nesbalansirovan
  3. Размеры файлов (проксируют разрешение/качество снимков)
  4. Распределение размеров изображений (разрешения)
  5. Средняя яркость изображений по каналам (R, G, B)
  6. Train / Valid / Test split пропорции

Запуск:
  cd D:\\python_projects\\retinal-disease-detection
  .venv\\Scripts\\python notebooks\\eda_full.py
"""

from __future__ import annotations

import os
import random
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # для headless рендеринга, сохраняем в файлы
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from PIL import Image
from tqdm import tqdm

# ─── Настройки ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "reports" / "eda"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative",
}

SPLITS = ["train", "valid", "test"]

# Палитра
COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]
SPLIT_COLORS = ["#3498db", "#2ecc71", "#e74c3c"]

# ─── Парсинг аннотаций ───────────────────────────────────────────────────────


def parse_annotations(split: str) -> list[tuple[str, int]]:
    """Парсим файл аннотаций: <filename> <label>."""
    ann_path = DATA_DIR / f"{split}.txt"
    records = []
    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(maxsplit=1)
            if len(parts) == 2:
                filename, label = parts[0], int(parts[1])
                records.append((filename, label))
    return records


# ─── Загрузка данных ─────────────────────────────────────────────────────────

print("=" * 60)
print("  EDA - DDR Retinal Disease Detection Dataset")
print("=" * 60)

all_data: dict[str, list[tuple[str, int]]] = {}
for split in SPLITS:
    all_data[split] = parse_annotations(split)
    print(f"  [{split:>5}] {len(all_data[split]):>5} images")

total = sum(len(v) for v in all_data.values())
print(f"  {'TOTAL':>7}: {total:>5} images")
print()

# ─── ГРАФИК 1: Распределение классов по каждому сплиту ────────────────────────

print(">> [1/6] Class distribution across train / valid / test...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle(
    "Распределение степеней тяжести DR по сплитам",
    fontsize=15,
    fontweight="bold",
    y=1.02,
)

for ax, split in zip(axes, SPLITS):
    labels = [rec[1] for rec in all_data[split]]
    counter = Counter(labels)
    x_labels = [CLASS_NAMES[i] for i in range(5)]
    counts = [counter.get(i, 0) for i in range(5)]

    bars = ax.bar(
        x_labels, counts, color=COLORS, edgecolor="black", linewidth=0.5, alpha=0.9
    )

    # Подписи с числами над столбцами
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.02,
            f"{count}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title(f"{split.upper()} ({len(all_data[split])} img)", fontsize=12)
    ax.set_xlabel("Класс DR")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

axes[0].set_ylabel("Количество снимков")
plt.tight_layout()
plt.savefig(REPORT_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   [OK] Saved: reports/eda/01_class_distribution.png")


# ─── ГРАФИК 2: Imbalance Ratio (bar + числовая аннотация) ────────────────────

print(">> [2/6] Imbalance Ratio (train)...")

train_labels = [rec[1] for rec in all_data["train"]]
train_counter = Counter(train_labels)
max_class_count = max(train_counter.values())

fig, ax = plt.subplots(figsize=(10, 5))
ratios = []
x_labels = []
for cls_id in range(5):
    count = train_counter.get(cls_id, 1)
    ratio = max_class_count / count
    ratios.append(ratio)
    x_labels.append(CLASS_NAMES[cls_id])

bars = ax.barh(
    x_labels, ratios, color=COLORS, edgecolor="black", linewidth=0.5, alpha=0.9
)

for bar, ratio, cls_id in zip(bars, ratios, range(5)):
    count = train_counter.get(cls_id, 0)
    ax.text(
        bar.get_width() + 0.1,
        bar.get_y() + bar.get_height() / 2,
        f"x{ratio:.1f}  ({count} img)",
        va="center",
        fontsize=10,
    )

ax.set_xlabel("Коэффициент дисбаланса (max_class / current_class)")
ax.set_title("Imbalance Ratio по классам (train)", fontsize=13, fontweight="bold")
ax.grid(axis="x", linestyle="--", alpha=0.4)
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(REPORT_DIR / "02_imbalance_ratio.png", dpi=150, bbox_inches="tight")
plt.close()
print("   [OK] Saved: reports/eda/02_imbalance_ratio.png")


# ─── ГРАФИК 3: Распределение размеров файлов ─────────────────────────────────

print(">> [3/6] File size distribution (train)...")

file_sizes_kb = []
for filename, label in tqdm(all_data["train"], desc="   File sizes", ncols=80):
    fpath = DATA_DIR / "train" / filename
    if fpath.exists():
        file_sizes_kb.append(fpath.stat().st_size / 1024)

fig, ax = plt.subplots(figsize=(12, 5))
ax.hist(file_sizes_kb, bins=80, color="#3498db", edgecolor="white", alpha=0.85)
ax.axvline(
    np.median(file_sizes_kb),
    color="#e74c3c",
    linestyle="--",
    linewidth=2,
    label=f"Медиана: {np.median(file_sizes_kb):.0f} KB",
)
ax.axvline(
    np.mean(file_sizes_kb),
    color="#2ecc71",
    linestyle="--",
    linewidth=2,
    label=f"Среднее: {np.mean(file_sizes_kb):.0f} KB",
)
ax.set_xlabel("Размер файла (KB)")
ax.set_ylabel("Количество изображений")
ax.set_title("Распределение размеров файлов (train)", fontsize=13, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(REPORT_DIR / "03_file_size_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("   [OK] Saved: reports/eda/03_file_size_distribution.png")


# ─── ГРАФИК 4: Распределение разрешений (Width x Height) ─────────────────────

print(">> [4/6] Image resolutions (sample 500 from train)...")

# Сэмплируем для скорости
sample_size = min(500, len(all_data["train"]))
sample_indices = random.sample(range(len(all_data["train"])), sample_size)

widths = []
heights = []
broken_count = 0

for idx in tqdm(sample_indices, desc="   Resolutions", ncols=80):
    filename, label = all_data["train"][idx]
    fpath = DATA_DIR / "train" / filename
    try:
        with Image.open(fpath) as img:
            w, h = img.size
            widths.append(w)
            heights.append(h)
    except Exception:
        broken_count += 1

if broken_count > 0:
    print(f"   [WARN] Broken files in sample: {broken_count}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    "Разрешения изображений (сэмпл 500 из train)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

# Scatter width vs height
axes[0].scatter(widths, heights, alpha=0.4, s=15, color="#8e44ad", edgecolors="none")
axes[0].set_xlabel("Ширина (px)")
axes[0].set_ylabel("Высота (px)")
axes[0].set_title("Width x Height")
axes[0].grid(linestyle="--", alpha=0.4)

# Histogram ширин
unique_resolutions = Counter(zip(widths, heights))
res_labels = [f"{w}x{h}" for (w, h), _ in unique_resolutions.most_common(10)]
res_counts = [c for _, c in unique_resolutions.most_common(10)]
axes[1].barh(
    res_labels,
    res_counts,
    color="#e67e22",
    edgecolor="black",
    linewidth=0.5,
    alpha=0.85,
)
axes[1].set_xlabel("Количество (из 500)")
axes[1].set_title("Топ-10 разрешений")
axes[1].invert_yaxis()
axes[1].grid(axis="x", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(REPORT_DIR / "04_image_resolutions.png", dpi=150, bbox_inches="tight")
plt.close()
print("   [OK] Saved: reports/eda/04_image_resolutions.png")


# ─── ГРАФИК 5: Средняя яркость по каналам (R, G, B) ──────────────────────────

print(">> [5/6] Channel brightness R/G/B (sample 300 from train)...")

brightness_sample = min(300, len(all_data["train"]))
brightness_indices = random.sample(range(len(all_data["train"])), brightness_sample)

r_means = []
g_means = []
b_means = []
class_labels_brightness = []

for idx in tqdm(brightness_indices, desc="   Brightness", ncols=80):
    filename, label = all_data["train"][idx]
    fpath = DATA_DIR / "train" / filename
    try:
        with Image.open(fpath) as img:
            img_rgb = img.convert("RGB")
            arr = np.array(img_rgb, dtype=np.float32)
            r_means.append(arr[:, :, 0].mean())
            g_means.append(arr[:, :, 1].mean())
            b_means.append(arr[:, :, 2].mean())
            class_labels_brightness.append(label)
    except Exception:
        pass

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
fig.suptitle(
    "Распределение средней яркости по каналам (сэмпл 300)",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

channel_data = [
    (r_means, "Red", "#e74c3c"),
    (g_means, "Green", "#2ecc71"),
    (b_means, "Blue", "#3498db"),
]

for ax, (data, name, color) in zip(axes, channel_data):
    ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.8)
    mean_val = np.mean(data)
    ax.axvline(
        mean_val,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {mean_val:.1f}",
    )
    ax.set_xlabel(f"{name} channel (0-255)")
    ax.set_title(name)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

axes[0].set_ylabel("Кол-во изображений")
plt.tight_layout()
plt.savefig(REPORT_DIR / "05_channel_brightness.png", dpi=150, bbox_inches="tight")
plt.close()
print("   [OK] Saved: reports/eda/05_channel_brightness.png")


# ─── ГРАФИК 6: Train / Valid / Test split пропорции ───────────────────────────

print(">> [6/6] Train / Valid / Test split proportions...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Разбиение датасета: Train / Valid / Test", fontsize=13, fontweight="bold")

# Pie chart
split_sizes = [len(all_data[s]) for s in SPLITS]
split_labels_pie = [
    f"{s.upper()}\n{n} img\n({n / total * 100:.1f}%)"
    for s, n in zip(SPLITS, split_sizes)
]
wedges, texts = axes[0].pie(
    split_sizes,
    labels=split_labels_pie,
    colors=SPLIT_COLORS,
    startangle=90,
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=10),
)
axes[0].set_title("Общее соотношение сплитов")

# Stacked bar per class showing split proportions
x = np.arange(5)
width = 0.6

bottoms = np.zeros(5)
for split, color in zip(SPLITS, SPLIT_COLORS):
    labels = [rec[1] for rec in all_data[split]]
    counter = Counter(labels)
    counts = np.array([counter.get(i, 0) for i in range(5)], dtype=float)
    axes[1].bar(
        x,
        counts,
        width,
        bottom=bottoms,
        color=color,
        edgecolor="white",
        linewidth=0.5,
        label=split.upper(),
    )
    bottoms += counts

axes[1].set_xticks(x)
axes[1].set_xticklabels([CLASS_NAMES[i] for i in range(5)], rotation=20)
axes[1].set_ylabel("Количество снимков")
axes[1].set_title("Распределение по классам в каждом сплите")
axes[1].legend()
axes[1].grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(REPORT_DIR / "06_split_proportions.png", dpi=150, bbox_inches="tight")
plt.close()
print("   [OK] Saved: reports/eda/06_split_proportions.png")


# ─── Итоговая сводка ─────────────────────────────────────────────────────────

print()
print("=" * 60)
print("  SUMMARY")
print("=" * 60)

for split in SPLITS:
    labels = [rec[1] for rec in all_data[split]]
    counter = Counter(labels)
    print(f"\n  {split.upper()} ({len(labels)} img):")
    for cls_id in range(5):
        cnt = counter.get(cls_id, 0)
        pct = cnt / len(labels) * 100 if len(labels) > 0 else 0
        print(f"    {CLASS_NAMES[cls_id]:<14}: {cnt:>5} ({pct:>5.1f}%)")

print(f"\n  Median file size (train): {np.median(file_sizes_kb):.0f} KB")
print(f"  Mean file size (train): {np.mean(file_sizes_kb):.0f} KB")
print(
    f"  Min/Max file size (train): {np.min(file_sizes_kb):.0f} / {np.max(file_sizes_kb):.0f} KB"
)

if widths:
    uw = sorted(set(widths))
    uh = sorted(set(heights))
    print(f"\n  Unique widths (sample): {len(uw)}, range: {min(uw)}-{max(uw)} px")
    print(f"  Unique heights (sample): {len(uh)}, range: {min(uh)}-{max(uh)} px")

print(f"\n  Mean brightness R: {np.mean(r_means):.1f}")
print(f"  Mean brightness G: {np.mean(g_means):.1f}")
print(f"  Mean brightness B: {np.mean(b_means):.1f}")

print()
print("=" * 60)
print(f"  [OK] All 6 charts saved to: {REPORT_DIR}")
print("=" * 60)
