import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# data
models = [
    "Zero-shot",
    "RAG + Vreg",
    "RAG + Vreg + Cert\n(Concat)",
    "RAG + Vreg + Cert\n(Avg)"
]

recall_conspiracy = [
    0.5948,
    0.6961,
    0.6618,
    0.7876
]

data = pd.DataFrame({
    "Model": models,
    "Recall": recall_conspiracy
})

color = "#40798C"
plt.rcParams['font.family'] = 'Arial'

# plot
plt.figure(figsize=(8, 3.5))
ax = sns.barplot(
    data=data,
    y="Model",
    x="Recall",
    color=color,
    width=0.6
)

# remove spines
sns.despine(ax=ax, left=True, bottom=True)

# axis formatting
ax.set_xlim(0, 1.0)
ax.set_xlabel("Recall", fontsize=11)
ax.set_ylabel("")
ax.tick_params(axis='both', labelsize=11)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.3f",
        label_type="edge",
        padding=-35,
        color="white",
        fontsize=11,
        fontweight="bold"
    )

plt.tight_layout()
plt.show()
