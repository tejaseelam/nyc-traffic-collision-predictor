"""Fit historical-average baselines and a logistic regression for has_crash.

Only the training (2021-2022) and validation (2023) files are read. The 2024
test file is intentionally never opened here.

Models compared on the validation file:

* cell_rate:       each cell's crash rate over the training years
* cell_hour_rate:  each cell's crash rate for the same six-hour slot
* logistic C=...:  logistic regression on calendar, weather, land, speed-limit,
                   and past-only crash-history features

History features for a period use only outcomes from earlier periods, so they
can be computed on the training and validation years together without leaking
the label being predicted. Negatives are subsampled for fitting and the
predicted probabilities are corrected back to the full-data base rate.
"""

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = "model_panel_train_2021_2022.parquet"
VALIDATION_FILE = "model_panel_validation_2023.parquet"
TARGET = "has_crash"

PANEL_COLUMNS = [
    "cell_id",
    "time_bin",
    TARGET,
    "borough",
    "hour",
    "day_of_week",
    "month",
    "land_fraction",
    "speed_road_length_m",
    "speed_road_segments",
    "speed_limit_weighted_avg_mph",
    "speed_limit_coverage",
    "weather_temp_midpoint_c",
    "weather_precip_mm",
    "weather_snowfall_mm",
    "weather_snow_depth_mm",
    "weather_wind_mps",
    "weather_precip_day",
    "weather_snow_day",
    "weather_any_missing",
]
CATEGORICAL_FEATURES = ["hour", "day_of_week", "month", "borough"]
NUMERIC_FEATURES = [
    "weather_temp_midpoint_c",
    "weather_precip_mm",
    "weather_snowfall_mm",
    "weather_snow_depth_mm",
    "weather_wind_mps",
    "weather_precip_day",
    "weather_snow_day",
    "weather_any_missing",
    "land_fraction",
    "log_speed_road_length_m",
    "log_speed_road_segments",
    "speed_limit_weighted_avg_mph",
    "speed_limit_coverage",
    "history_cell_logit",
    "history_cell_hour_logit",
    "history_recent_logit",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
RATE_FLOOR = 1e-5


def require_columns(frame, columns, label):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def to_float(series):
    return series.astype("Float64").to_numpy(dtype=np.float32, na_value=np.nan)


def prepare_panel(frame, label):
    """Keep the modeling columns and convert nullable types for scikit-learn."""
    require_columns(frame, PANEL_COLUMNS, label)
    panel = frame[PANEL_COLUMNS].copy()
    panel["time_bin"] = pd.to_datetime(panel["time_bin"])
    panel[TARGET] = panel[TARGET].astype(bool)
    panel["borough"] = panel["borough"].astype("string").fillna("UNKNOWN").astype(str)
    for column in ["hour", "day_of_week", "month"]:
        panel[column] = panel[column].astype(np.int16)
    for column in [
        "land_fraction",
        "speed_limit_weighted_avg_mph",
        "weather_temp_midpoint_c",
        "weather_precip_mm",
        "weather_snowfall_mm",
        "weather_snow_depth_mm",
        "weather_wind_mps",
        "weather_precip_day",
        "weather_snow_day",
        "weather_any_missing",
    ]:
        panel[column] = to_float(panel[column])

    # Cells without mapped roads have no coverage rather than unknown coverage.
    panel["speed_limit_coverage"] = np.nan_to_num(to_float(panel["speed_limit_coverage"]))
    panel["log_speed_road_length_m"] = np.log1p(
        np.nan_to_num(to_float(panel["speed_road_length_m"]))
    )
    panel["log_speed_road_segments"] = np.log1p(
        np.nan_to_num(to_float(panel["speed_road_segments"]))
    )
    return panel.drop(columns=["speed_road_length_m", "speed_road_segments"])


def load_split(path, label):
    if not path.is_file():
        raise FileNotFoundError(f"{label} file not found: {path}")
    return prepare_panel(pd.read_parquet(path, columns=PANEL_COLUMNS), label)


def logit(rate):
    rate = np.clip(rate, RATE_FLOOR, 1 - RATE_FLOOR)
    return np.log(rate) - np.log1p(-rate)


def prior_cumulative(outcomes):
    """Event totals from all periods strictly before each row."""
    totals = np.zeros(outcomes.shape, dtype=np.float64)
    np.cumsum(outcomes[:-1], axis=0, out=totals[1:])
    return totals


def smoothed_rate(events, periods, cells, prior_strength):
    """Shrink each cell's past rate toward the past city-wide rate."""
    city_rate = events.sum(axis=1, keepdims=True) / np.maximum(periods * cells, 1)
    return (events + prior_strength * city_rate) / (periods + prior_strength)


def expanding_rate(outcomes, prior_strength):
    """Smoothed crash rate from every earlier period, for each (period, cell)."""
    period_count, cell_count = outcomes.shape
    events = prior_cumulative(outcomes)
    periods = np.arange(period_count, dtype=np.float64)[:, None]
    return smoothed_rate(events, periods, cell_count, prior_strength)


def recent_rate(outcomes, window, prior_strength):
    """Smoothed crash rate over the previous `window` periods only."""
    period_count, cell_count = outcomes.shape
    events = prior_cumulative(outcomes)
    window_events = events.copy()
    window_events[window:] -= events[:-window]
    periods = np.minimum(np.arange(period_count), window).astype(np.float64)[:, None]
    return smoothed_rate(window_events, periods, cell_count, prior_strength)


def add_history_features(frames, prior_strength, recent_window_days):
    """Add past-only crash-history features to chronologically ordered frames.

    The frames are combined so later splits can use history from earlier ones.
    Every feature for period t is computed only from periods before t.
    """
    combined = pd.concat(
        [frame[["cell_id", "time_bin", TARGET]] for frame in frames],
        ignore_index=True,
    )
    times = np.sort(combined["time_bin"].unique())
    cells = np.sort(combined["cell_id"].unique())
    time_index = np.searchsorted(times, combined["time_bin"].to_numpy())
    cell_index = np.searchsorted(cells, combined["cell_id"].to_numpy())

    filled = np.zeros((len(times), len(cells)), dtype=np.int32)
    np.add.at(filled, (time_index, cell_index), 1)
    if (filled != 1).any():
        raise ValueError(
            "History features need exactly one row per cell and period; "
            f"found {int((filled == 0).sum()):,} gaps and "
            f"{int((filled > 1).sum()):,} duplicates"
        )

    outcomes = np.zeros(filled.shape, dtype=np.float32)
    outcomes[time_index, cell_index] = combined[TARGET].to_numpy(dtype=np.float32)

    period_length = pd.Series(times).diff().min()
    if len(times) < 2 or pd.Timedelta(days=1) % period_length != pd.Timedelta(0):
        raise ValueError("Periods must evenly divide one day")
    window = max(1, int(recent_window_days * (pd.Timedelta(days=1) / period_length)))

    cell_rate = expanding_rate(outcomes, prior_strength)
    recent = recent_rate(outcomes, window, prior_strength)
    cell_hour_rate = np.empty_like(cell_rate)
    hours = pd.DatetimeIndex(times).hour
    for hour in np.unique(hours):
        rows = np.flatnonzero(hours == hour)
        cell_hour_rate[rows] = expanding_rate(outcomes[rows], prior_strength)

    features = {
        "history_cell_logit": logit(cell_rate[time_index, cell_index]),
        "history_cell_hour_logit": logit(cell_hour_rate[time_index, cell_index]),
        "history_recent_logit": logit(recent[time_index, cell_index]),
    }

    results = []
    start = 0
    for frame in frames:
        stop = start + len(frame)
        frame = frame.copy()
        for name, values in features.items():
            frame[name] = values[start:stop].astype(np.float32)
        results.append(frame)
        start = stop
    return results


def training_rate(train, validation, keys, prior_strength):
    """Validation predictions from smoothed training-period crash rates."""
    city_rate = train[TARGET].mean()
    grouped = train.groupby(keys, observed=True)[TARGET].agg(["sum", "count"])
    rates = (grouped["sum"] + prior_strength * city_rate) / (
        grouped["count"] + prior_strength
    )
    index = pd.MultiIndex.from_frame(validation[keys]) if len(keys) > 1 else validation[keys[0]]
    return rates.reindex(index).fillna(city_rate).to_numpy(dtype=np.float64)


def evaluate(y_true, probability, top_fraction=0.05):
    y_true = np.asarray(y_true, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)

    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    denominator = precision[:-1] + recall[:-1]
    f1 = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best = int(np.argmax(f1))

    top_count = max(1, int(round(top_fraction * len(probability))))
    top_rows = np.argsort(-probability, kind="stable")[:top_count]
    top_hits = int(y_true[top_rows].sum())

    return {
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "mean_predicted_probability": float(probability.mean()),
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "log_loss": float(log_loss(y_true, np.clip(probability, 1e-7, 1 - 1e-7))),
        "brier_score": float(brier_score_loss(y_true, probability)),
        "top_fraction": top_fraction,
        "precision_at_top_fraction": top_hits / top_count,
        "recall_at_top_fraction": top_hits / max(int(y_true.sum()), 1),
        "best_f1": float(f1[best]),
        "best_f1_threshold": float(thresholds[best]),
        "precision_at_best_f1": float(precision[best]),
        "recall_at_best_f1": float(recall[best]),
    }


def build_model(c_value):
    preprocess = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
        (
            "numeric",
            make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
            ),
            NUMERIC_FEATURES,
        ),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("model", LogisticRegression(C=c_value, max_iter=2000)),
    ])


def sample_training_rows(y_true, negative_rate, seed):
    """Keep every crash row and a random share of zero-crash rows."""
    if not 0 < negative_rate <= 1:
        raise ValueError("Negative sample rate must be in (0, 1]")
    keep = np.asarray(y_true, dtype=bool).copy()
    random = np.random.default_rng(seed)
    keep |= random.random(len(keep)) < negative_rate
    return np.flatnonzero(keep)


def corrected_probability(model, features, negative_rate):
    """Undo the log-odds shift caused by subsampling negatives."""
    return expit(model.decision_function(features) + np.log(negative_rate))


def coefficient_table(model):
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_.ravel()
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["absolute_coefficient"] = table["coefficient"].abs()
    return table.sort_values("absolute_coefficient", ascending=False).reset_index(drop=True)


def fit_and_evaluate(
    train,
    validation,
    c_values=(0.01, 0.1, 1.0),
    negative_rate=0.1,
    burn_in_days=28,
    prior_strength=20.0,
    recent_window_days=28,
    seed=42,
    top_fraction=0.05,
):
    """Compare baselines and logistic models on validation and return the best."""
    train, validation = add_history_features(
        [train, validation], prior_strength, recent_window_days
    )
    y_validation = validation[TARGET].to_numpy()

    results = {"baselines": {}, "logistic": {}}
    print("Evaluating historical-average baselines...")
    for name, keys in [("cell_rate", ["cell_id"]), ("cell_hour_rate", ["cell_id", "hour"])]:
        probability = training_rate(train, validation, keys, prior_strength)
        results["baselines"][name] = evaluate(y_validation, probability, top_fraction)

    burn_in_end = train["time_bin"].min() + pd.Timedelta(days=burn_in_days)
    eligible = train.loc[train["time_bin"] >= burn_in_end]
    if eligible.empty:
        raise ValueError("No training rows remain after the burn-in period")
    rows = sample_training_rows(eligible[TARGET].to_numpy(), negative_rate, seed)
    fit_frame = eligible.iloc[rows]
    print(
        f"Fitting on {len(fit_frame):,} sampled rows "
        f"({int(fit_frame[TARGET].sum()):,} with crashes)"
    )

    best_name = None
    best_model = None
    for c_value in c_values:
        name = f"C={c_value:g}"
        print(f"Fitting logistic regression {name}...")
        model = build_model(c_value)
        model.fit(fit_frame[FEATURES], fit_frame[TARGET].to_numpy())
        probability = corrected_probability(model, validation[FEATURES], negative_rate)
        results["logistic"][name] = evaluate(y_validation, probability, top_fraction)
        score = results["logistic"][name]["average_precision"]
        if best_name is None or score > results["logistic"][best_name]["average_precision"]:
            best_name = name
            best_model = model

    results["selected_logistic"] = best_name
    results["selection_metric"] = "validation average_precision"
    results["settings"] = {
        "c_values": list(c_values),
        "negative_sample_rate": negative_rate,
        "burn_in_days": burn_in_days,
        "prior_strength": prior_strength,
        "recent_window_days": recent_window_days,
        "seed": seed,
        "training_rows_after_burn_in": int(len(eligible)),
        "fitted_rows": int(len(fit_frame)),
        "features": FEATURES,
    }
    return results, best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--c-values", default="0.01,0.1,1")
    parser.add_argument("--negative-sample-rate", type=float, default=0.1)
    parser.add_argument("--burn-in-days", type=int, default=28)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--recent-window-days", type=int, default=28)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    output_directory = args.output_directory or args.root / "reports" / "baseline_logreg"
    model_path = args.model_path or processed / "models" / "logreg_has_crash.joblib"
    c_values = [float(value) for value in args.c_values.split(",") if value.strip()]
    if not c_values or any(value <= 0 for value in c_values):
        raise ValueError("C values must be positive")

    print("Loading training and validation files...")
    train = load_split(processed / TRAIN_FILE, "Training file")
    validation = load_split(processed / VALIDATION_FILE, "Validation file")

    results, model = fit_and_evaluate(
        train,
        validation,
        c_values=c_values,
        negative_rate=args.negative_sample_rate,
        burn_in_days=args.burn_in_days,
        prior_strength=args.prior_strength,
        recent_window_days=args.recent_window_days,
        seed=args.seed,
        top_fraction=args.top_fraction,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "validation_metrics.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    coefficient_table(model).to_csv(output_directory / "coefficients.csv", index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "negative_sample_rate": args.negative_sample_rate,
            "features": FEATURES,
            "selected": results["selected_logistic"],
        },
        model_path,
    )

    print("\nVALIDATION RESULTS (2023)")
    rows = {**results["baselines"], **{f"logistic {k}": v for k, v in results["logistic"].items()}}
    for name, metrics in rows.items():
        print(
            f"{name:>22}: AP={metrics['average_precision']:.4f} "
            f"ROC-AUC={metrics['roc_auc']:.4f} "
            f"log-loss={metrics['log_loss']:.4f} "
            f"recall@top{metrics['top_fraction']:.0%}={metrics['recall_at_top_fraction']:.3f}"
        )
    print(f"Selected logistic model: {results['selected_logistic']}")
    print(f"Metrics: {output_directory / 'validation_metrics.json'}")
    print(f"Coefficients: {output_directory / 'coefficients.csv'}")
    print(f"Model: {model_path}")


if __name__ == "__main__":
    main()
