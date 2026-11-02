import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "crashes_weather_traffic_speed.parquet"

OUTPUT = ROOT / "data" / "processed" / "crashes_sample_40.csv"

def main():
    df = pd.read_parquet(INPUT)

    df_sample = df.head(40)

    df_sample.to_csv(OUTPUT, index=False)

if __name__ == "__main__":
    main()