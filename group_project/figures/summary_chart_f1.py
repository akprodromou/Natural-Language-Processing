import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
models = [
    "Zero-shot",
    "RAG + Vreg",
    "RAG + Vreg + Cert\n(Concat)",
    "RAG + Vreg + Cert\n(Avg)"
]

data = pd.DataFrame({
    "Model": models * 2,
    "Class": ["Non-Conspiracy"] * 4 + ["Conspiracy"] * 4,
    "F1-score": [
        0.5085, 0.8697, 0.8687, 0.8672,
        0.4407, 0.7339, 0.7194, 0.7621
    ]
})

# Color palette
palette = {
    "Non-Conspiracy": "#8c4053",
    "Conspiracy": "#40798C"
}

# Plot
plt.figure(figsize=(8, 5))
ax = sns.barplot(
    data=data,
    x="Model",
    y="F1-score",
    hue="Class",
    palette=palette,
    width=0.6
)

# Remove spines
sns.despine(ax=ax, left=True, bottom=True)

# Axis formatting
ax.set_ylim(0, 1.0)
ax.set_ylabel("F1-score", fontsize=11)
ax.set_xlabel("Model variant", fontsize=11)
ax.tick_params(axis='both', labelsize=11)
# ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.legend(title=None, fontsize=11)

# Add labels inside bars
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.3f",
        label_type="edge",
        padding=-18,
        color="white",
        fontsize=11,
        fontweight="bold"
    )

plt.tight_layout()
plt.show()
