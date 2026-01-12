import matplotlib
matplotlib.use("Agg")  # nodig op Snellius (geen scherm)

import matplotlib.pyplot as plt
from src.ml_core.data.pcam import PCAMDataset

DATA_PATH = "/scratch-shared/scur2292/pcam"

ds = PCAMDataset(
    f"{DATA_PATH}/camelyonpatch_level_2_split_train_x.h5",
    f"{DATA_PATH}/camelyonpatch_level_2_split_train_y.h5",
)

# ---------- Plot 1: label distribution ----------
labels = [ds[i][1].item() for i in range(100)]

plt.figure()
plt.hist(labels, bins=[-0.5, 0.5, 1.5])
plt.xticks([0, 1])
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Label distribution (subset)")
plt.tight_layout()
plt.savefig("eda_label_distribution.png", dpi=200)
plt.close()

# ---------- Plot 2: example images ----------
fig, axes = plt.subplots(1, 4, figsize=(8, 3))
for i, ax in enumerate(axes):
    x, y = ds[i]
    ax.imshow(x.permute(1, 2, 0))
    ax.set_title(f"Class {y.item()}")
    ax.axis("off")

plt.tight_layout()
plt.savefig("eda_example_patches.png", dpi=200)
plt.close()

print("EDA plots saved.")

