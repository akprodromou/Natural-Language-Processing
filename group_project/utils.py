import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# clean the column
def clean_sentiment_data(df):
    df = df.copy()

    df['Cert_clean'] = (
        df['Cert']
        .str.extract(r'([-+]?\d*\.?\d+)')
        .astype(float)
    )

    df = df[
        (df['Cert_clean'] >= 0) & (df['Cert_clean'] <= 1)
    ]

    return df

# returns sample size, mean, and variance of Cert_clean
# for conspiracy vs non-conspiracy labels
def cert_summary_by_label(df):    
    is_conspiracy = df['label'].str.contains(
        'CONSPIRACY', case=False, na=False
    )
    summary = {
        "conspiracy": {
            "n": is_conspiracy.sum(),
            "mean": df.loc[is_conspiracy, 'Cert_clean'].mean(),
            "variance": df.loc[is_conspiracy, 'Cert_clean'].var(ddof=1),
        },
        "non_conspiracy": {
            "n": (~is_conspiracy).sum(),
            "mean": df.loc[~is_conspiracy, 'Cert_clean'].mean(),
            "variance": df.loc[~is_conspiracy, 'Cert_clean'].var(ddof=1),
        }
    }

    return summary

# Runs a two-sided Welch's t-test comparing Cert_clean
# between conspiracy and non-conspiracy labels
def welch_cert_test(df):
    is_conspiracy = df['label'].str.contains(
        'CONSPIRACY', case=False, na=False
    )

    x = df.loc[is_conspiracy, 'Cert_clean']
    y = df.loc[~is_conspiracy, 'Cert_clean']

    t_stat, p_value = stats.ttest_ind(
        x, y, equal_var=False
    )

    return {
        "t_statistic": t_stat,
        "p_value": p_value
    }

