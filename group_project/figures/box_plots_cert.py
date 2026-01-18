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


# box plot
plt.figure(figsize=(5, 3.5))

sns.boxplot(
    data=df_cleaned,
    x="Group",
    y="Cert_clean",
    palette=color_palette,
    showmeans=True,
    meanprops={
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 5
    }
)

plt.ylabel("Certainty Score", fontsize=9)
plt.xlabel("")
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.show()

