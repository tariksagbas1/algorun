import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_sales_over_time(df):
    df_grouped = df.groupby("date")["sales_quantity_sum"].sum().reset_index()
    plt.figure(figsize=(12,5))
    plt.plot(df_grouped["date"], df_grouped["sales_quantity_sum"])
    plt.title("Total Sales Quantity Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.grid(True)
    plt.show()


def plot_revenue_over_time(df):
    df_grouped = df.groupby("date")["sales_revenue_sum"].sum().reset_index()
    plt.figure(figsize=(12,5))
    plt.plot(df_grouped["date"], df_grouped["sales_revenue_sum"])
    plt.title("Total Revenue Over Time")
    plt.xlabel("Date")
    plt.ylabel("Revenue")
    plt.grid(True)
    plt.show()


def plot_product_sales(df, product_id):
    df_prod = df[df["product_id"] == product_id]
    plt.figure(figsize=(12,5))
    plt.plot(df_prod["date"], df_prod["sales_quantity_sum"])
    plt.title(f"Sales Trend for Product {product_id}")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.grid(True)
    plt.show()


def plot_channel_trends(df, clip_percentile=99):
    # Aggregate per day per channel
    df_agg = (
        df.groupby(["date", "channel_type"])["sales_quantity_sum"]
          .sum()
          .reset_index()
    )

    # Clip extreme outliers (optional)
    max_val = df_agg["sales_quantity_sum"].quantile(clip_percentile / 100)
    df_agg["sales_quantity_sum"] = np.clip(df_agg["sales_quantity_sum"], 0, max_val)

    plt.figure(figsize=(12, 6))

    for ch, d in df_agg.groupby("channel_type"):
        plt.plot(d["date"], d["sales_quantity_sum"], label=ch)

    plt.title("Daily Total Sales per Channel (Clipped Outliers)")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def plot_weekly_sales(df):
    df['week'] = df['date'].dt.to_period('W').apply(lambda r: r.start_time)
    weekly = df.groupby('week')["sales_quantity_sum"].sum()
    weekly.plot(figsize=(12,5), title='Weekly Sales')


def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()


def plot_revenue_vs_store(df):
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="store_count_max", y="sales_revenue_sum")
    plt.title("Store Count vs Revenue")
    plt.xlabel("Store Count")
    plt.ylabel("Revenue")
    plt.grid(True)
    plt.show()

def plot_sales_vs_stock(df):
    plt.figure(figsize=(8,5))
    sns.scatterplot(data=df, x="stock_days", y="sales_quantity_sum")
    plt.title("Stock Days vs Sales Quantity")
    plt.xlabel("Stock Days")
    plt.ylabel("Sales Quantity")
    plt.grid(True)
    plt.show()


def plot_channel_trends(df):
    plt.figure(figsize=(12,5))
    for ch, d in df.groupby("channel_type"):
        plt.plot(d["date"], d["sales_quantity_sum"], label=ch)
    plt.title("Sales Trend by Channel")
    plt.xlabel("Date")
    plt.ylabel("Sales Quantity")
    plt.legend()
    plt.grid(True)
    plt.show()
