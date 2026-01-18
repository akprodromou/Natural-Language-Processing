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

# Color palette (unchanged)
palette = {
    "Non-Conspiracy": "#8c4053",
    "Conspiracy": "#40798C"
}

# Font setup (from first chart)
plt.rcParams['font.family'] = 'DM Sans Light'

# Plot
plt.figure(figsize=(18, 9))
ax = sns.barplot(
    data=data,
    x="F1-score",
    y="Model",
    hue="Class",
    palette=palette,
    width=0.7
)

# Remove spines
sns.despine(ax=ax, left=True, bottom=True)

# Axis formatting (matched to first chart)
ax.set_xlim(0, 1.0)
ax.set_xlabel("F1-score", fontsize=24)
ax.set_ylabel("")
ax.tick_params(axis='both', labelsize=24)

# Legend formatting
ax.legend(title=None, fontsize=24)

# Add value labels (style matched to first chart)
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.3f",
        label_type="edge",
        padding=-81,
        color="white",
        fontsize=24
    )

plt.tight_layout()
plt.show()
