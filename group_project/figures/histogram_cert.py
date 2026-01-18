import pandas as pd
import utils
import matplotlib.pyplot as plt
import seaborn as sns


# load data
df = pd.read_csv("./data/COCO-add-aff.csv")
df_cleaned = utils.clean_sentiment_data(df)

# create grouping variable
df_cleaned["Group"] = df_cleaned["label"].str.contains(
    "CONSPIRACY", case=False, na=False
).map({True: "Conspiracy", False: "Non-Conspiracy"})

color_palette = ['#8c4053', '#40798C']
plt.rcParams['font.family'] = 'Arial'


# histograms
fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

groups = ["Conspiracy", "Non-Conspiracy"]
colors = ['#8c4053', '#40798C']

for ax, group, color in zip(axes, groups, colors):
    data = df_cleaned.loc[df_cleaned["Group"] == group, "Cert_clean"]

    sns.histplot(
        data,
        bins=15,
        kde=False,
        ax=ax,
        color=color
    )

    mean_val = data.mean()
    ax.axvline(
        mean_val,
        linestyle="--",
        linewidth=1,
        color="black",
        label=f"Mean = {mean_val:.3f}"
    )

    ax.set_title(f"{group} Certainty Distribution", fontsize=11, loc='left')
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=9)
    sns.despine(ax=ax, left=True, bottom=True)

axes[-1].set_xlabel("Certainty Score", fontsize=9)

plt.tight_layout()
plt.show()
