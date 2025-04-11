import pandas as pd
import matplotlib.pyplot as plt
import ast
import re

def parse_dimension(dim_str):
    dim_str = dim_str.strip()
    pattern = r"\((\d+),\s*(\d+)\)"
    match = re.match(pattern, dim_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def main():
    parquet_path = "DATA/train-00001-of-00002-823ac5dae71e0e87.parquet"
    df = pd.read_parquet(parquet_path)

    print("Initial DataFrame shape:", df.shape)
    print("Sample of raw data:\n", df.head(5))

    # 1) Parse (width, height) from the string
    df[["width", "height"]] = df["dimensions"].apply(parse_dimension).tolist()

    print("\nAfter parsing 'dimensions', shape:", df.shape)
    print(df[["width", "height"]].head(5))

    # 2) Convert to numeric and drop invalids
    df["width"] = pd.to_numeric(df["width"], errors="coerce")
    df["height"] = pd.to_numeric(df["height"], errors="coerce")

    df.dropna(subset=["width", "height"], inplace=True)
    df = df[df["height"] != 0]  # remove zero-height if any
    df["width"] = df["width"].astype(int)
    df["height"] = df["height"].astype(int)

    print("\nAfter cleaning, shape:", df.shape)
    print(df[["width", "height"]].head(5))

    if df.empty:
        print("No valid data remains! Exiting.")
        return

    # 3) Build top-10 counts
    width_counts = df["width"].value_counts().head(10)
    height_counts = df["height"].value_counts().head(10)

    # For dimension pairs: "(width, height)"
    df["dim_str"] = "(" + df["width"].astype(str) + ", " + df["height"].astype(str) + ")"
    pair_counts = df["dim_str"].value_counts().head(10)

    # Area
    df["area"] = df["width"] * df["height"]
    area_counts = df["area"].value_counts().head(10)

    # Aspect ratio
    df["aspect_ratio"] = (df["width"] / df["height"]).round(2)
    ratio_counts = df["aspect_ratio"].value_counts().head(10)

    # Quick print of the counts
    print("\nTop 10 Widths:\n", width_counts)
    print("\nTop 10 Heights:\n", height_counts)
    print("\nTop 10 Dimension Pairs:\n", pair_counts)
    print("\nTop 10 Areas:\n", area_counts)
    print("\nTop 10 Aspect Ratios:\n", ratio_counts)

    # 4) Plot if non-empty
    if not width_counts.empty:
        plt.figure()
        plt.bar(width_counts.index.astype(str), width_counts.values)
        plt.title("Top 10 Most Frequent Widths")
        plt.xlabel("Width (pixels)")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("OUTPUTS/top_widths.png")
        plt.close()

    if not height_counts.empty:
        plt.figure()
        plt.bar(height_counts.index.astype(str), height_counts.values)
        plt.title("Top 10 Most Frequent Heights")
        plt.xlabel("Height (pixels)")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("OUTPUTS/top_heights.png")
        plt.close()

    if not pair_counts.empty:
        plt.figure(figsize=(10, 5))
        x_positions = range(len(pair_counts))
        plt.bar(x_positions, pair_counts.values)
        plt.title("Top 10 Most Frequent Dimension Pairs")
        plt.xlabel("Dimension Pair (width, height)")
        plt.ylabel("Count")
        plt.xticks(ticks=x_positions, labels=pair_counts.index, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("OUTPUTS/top_dimensions.png")
        plt.close()

    if not area_counts.empty:
        plt.figure()
        x_positions = range(len(area_counts))
        plt.bar(x_positions, area_counts.values)
        plt.title("Top 10 Most Frequent Areas")
        plt.xlabel("Area (width * height)")
        plt.ylabel("Count")
        plt.xticks(ticks=x_positions, labels=area_counts.index.astype(str), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("OUTPUTS/top_areas.png")
        plt.close()

    if not ratio_counts.empty:
        plt.figure()
        x_positions = range(len(ratio_counts))
        plt.bar(x_positions, ratio_counts.values)
        plt.title("Top 10 Most Frequent Aspect Ratios")
        plt.xlabel("Aspect Ratio (width / height)")
        plt.ylabel("Count")
        plt.xticks(ticks=x_positions, labels=ratio_counts.index.astype(str), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("OUTPUTS/top_aspect_ratios.png")
        plt.close()

    print("\nAll plots done.")

if __name__ == "__main__":
    main()
