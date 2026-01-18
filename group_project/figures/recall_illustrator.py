import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 1. Data Setup
models = [
    "Zero-shot",
    "RAG + Vreg",
    "RAG + Vreg + Cert\n(Concat)",
    "RAG + Vreg + Cert\n(Avg)"
]
recall_conspiracy = [0.5948, 0.6961, 0.6618, 0.7876]
data = pd.DataFrame({"Model": models, "Recall": recall_conspiracy})

# 2. Define Colors
# First three are '#a170a9', the last one is '#70A9A1'
custom_palette = ['#40798C', '#40798C', '#40798C', '#8bbaca']

# 3. Font Setup
# Specify the exact font variant name
plt.rcParams['font.family'] = 'DM Sans Light'

# 4. Plot
plt.figure(figsize=(18, 7))
ax = sns.barplot(
    data=data,
    y="Model",
    x="Recall",
    palette=custom_palette, # Apply the colors here
    width=0.6
)

# remove spines
sns.despine(ax=ax, left=True, bottom=True)

# axis formatting
ax.set_xlim(0, 1.0)
ax.set_xlabel("Recall", fontsize=24)
ax.set_ylabel("")
ax.tick_params(axis='both', labelsize=24)

# Add value labels (keeping these bold as in your original)
for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.3f",
        label_type="edge",
        padding=-90.5,
        color="white",
        fontsize=24
    )

plt.tight_layout()
plt.show()