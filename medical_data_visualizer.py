import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('medical_examination.csv')

# Compute BMI and classify overweight
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['overweight'] = (df['BMI'] > 25).astype(int)  # 1 if BMI > 25, else 0

# Normalize cholesterol and glucose levels
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)


def draw_cat_plot():
    """Draws a categorical count plot comparing different health indicators by 'cardio' status."""

    # Melt the dataframe
    df_cat = pd.melt(
        df,
        id_vars=["cardio"],
        value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"],
        var_name="variable",
        value_name="value"
    )

    # Create categorical count plot
    fig = sns.catplot(
        data=df_cat,
        x="variable",
        hue="value",
        kind="count",
        col="cardio",
        height=5,  # Adjusted height
        aspect=1.1
    )

    fig.set_axis_labels("Health Indicator", "Count")
    fig.set_titles("Cardio Status: {col_name}")

    plt.tight_layout()
    fig.savefig('catplot.png')  # Save plot
    plt.close(fig.figure)  # Close to free memory

    return fig


def draw_heat_map():
    """Draws a heatmap to visualize correlations between medical indicators after cleaning data."""

    # Clean the data (remove incorrect blood pressure readings)
    df_heat = df[df['ap_lo'] <= df['ap_hi']]

    # Remove height and weight outliers (only keep data within 2.5th and 97.5th percentiles)
    df_heat = df_heat[
        (df_heat['height'] >= df_heat['height'].quantile(0.025)) &
        (df_heat['height'] <= df_heat['height'].quantile(0.975)) &
        (df_heat['weight'] >= df_heat['weight'].quantile(0.025)) &
        (df_heat['weight'] <= df_heat['weight'].quantile(0.975))
    ]

    # Compute correlation matrix (excluding ID if present)
    if "id" in df_heat.columns:
        df_heat = df_heat.drop(columns=["id"])

    corr = df_heat.corr()

    # Create heatmap figure
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".1f", cmap="RdBu_r", linewidths=0.5, ax=ax, center=0)

    plt.title("Correlation Heatmap of Medical Data")
    plt.tight_layout()
    fig.savefig("heatmap.png")  # Save heatmap
    plt.close(fig)  # Close to free memory

    return fig


# Call the functions to generate and save the plots
draw_cat_plot()
draw_heat_map()
