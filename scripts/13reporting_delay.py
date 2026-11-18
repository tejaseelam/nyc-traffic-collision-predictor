"""Measure how late crash reports change each model's 2023 validation results.

A crash is not in the data the moment it happens. If reports arrive N days
late, the crash-history features for a six-hour period can only use outcomes
from at least N days before it. This script rebuilds those features with a
0-, 3-, 7- and 14-day delay and scores three models for each delay:

* baseline:  the rolling cell + time-of-day rate
* logistic:  logistic regression with the C selected by scripts/8baseline_logreg.py
* lightgbm:  LightGBM with the location features (the full model)

Each model is refit on the 2021-2022 training file with the delayed features
and scored on 2023 with the same delay, as a deployed model facing that delay
would be ("refit"). The models fitted without a delay are also scored on the
delayed features ("frozen"), to show what ignoring the delay in training costs.

Everything else matches scripts/8baseline_logreg.py and scripts/10tree_model.py:
the same burn-in, sampled training rows, settings, and measurements, so the
0-day rows reproduce those scripts' results. Each change from the 0-day delay
gets a 95% interval from the day-block bootstrap of scripts/12compare_models.py.

Longer delays can be passed with --delays. A delay longer than 14 days would
leave the first fitted rows with almost no visible crash history, so those rows
are dropped: every fitted row keeps at least 14 days of history.

Only 2023 rows are scored. The training file supplies the rows the models are
fitted on and the crash history before 2023. The 2024 test file is never opened.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import importlib.util

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import union_categoricals
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TREE = load_script("tree_model", "10tree_model.py")
COMPARE = load_script("compare_models", "12compare_models.py")
LOGREG = TREE.LOGREG
PLOTS = COMPARE.PLOTS

TARGET = LOGREG.TARGET
VALIDATION_YEAR = COMPARE.VALIDATION_YEAR
DELAYS_DAYS = (0, 3, 7, 14)
# The least crash history a fitted row may have left once the delay hides the
# newest reports. It is what the 14-day delay leaves after the 28-day burn-in.
MINIMUM_HISTORY_DAYS = 14
HISTORY_COLUMNS = TREE.FEATURE_GROUPS["Crash history"]
BASELINE_COLUMN = "history_cell_hour_logit"
TREE_FEATURES = TREE.FEATURE_SETS["core_plus_location"]
TREE_SETTINGS = TREE.CANDIDATES["shallow"]
# (name in the shared CSV, plot label, color), as in scripts/12compare_models.py.
MODELS = [(name, label, color) for name, _, _, label, color in COMPARE.MODELS[:3]]
MARKERS = {"baseline": "o", "logistic": "s", "lightgbm": "D"}
SHARED_CSV = "yashwant_reporting_delay.csv"
DETAILS_CSV = "reporting_delay_details.csv"
FIGURE = "reporting_delay.png"


def delay_history(frames, delay_days):
    """Give every period the crash-history features it had `delay_days` earlier.

    A history feature for a period summarizes every outcome before it, so its
    value `delay_days` earlier is exactly what is known when reports take that
    long to arrive. Periods in the first `delay_days` of the panel have nothing
    earlier and get the first period's value, which is the no-history value.
    """
    if delay_days < 0:
        raise ValueError("The delay cannot be negative")
    if delay_days == 0:
        return list(frames)

    time_index, times = pd.factorize(
        pd.concat([frame["time_bin"] for frame in frames], ignore_index=True), sort=True
    )
    cell_index, cells = pd.factorize(
        union_categoricals([frame["cell_id"].astype("category") for frame in frames])
    )
    times = times.to_numpy()
    earlier = times - np.timedelta64(delay_days, "D")
    source = np.searchsorted(times, earlier)
    in_panel = earlier >= times[0]
    if (times[source[in_panel]] != earlier[in_panel]).any():
        raise ValueError(f"Some periods have no period exactly {delay_days} days earlier")
    source_index = source[time_index]

    delayed = {}
    for column in HISTORY_COLUMNS:
        grid = np.full((len(times), len(cells)), np.nan, dtype=np.float32)
        grid[time_index, cell_index] = np.concatenate(
            [frame[column].to_numpy(dtype=np.float32) for frame in frames]
        )
        delayed[column] = grid[source_index, cell_index]
        if np.isnan(delayed[column]).any():
            raise ValueError("History features need exactly one row per cell and period")

    results = []
    start = 0
    for frame in frames:
        stop = start + len(frame)
        frame = frame.copy(deep=False)
        for column, values in delayed.items():
            frame[column] = values[start:stop]
        results.append(frame)
        start = stop
    return results


def fitting_rows(train, delay_days, negative_rate, burn_in_days, minimum_history_days, seed):
    """The sampled training rows of scripts 8 and 10, without rows the delay leaves blind.

    Returns the rows for the logistic regression, and the same rows split into
    LightGBM's fitting and early-stopping parts. Every delay starts from the
    same sample and the same split. A row is then dropped when the delay hides
    all but its first `minimum_history_days` of crash history, so a long delay
    is not fitted on rows whose history features are still empty. With the
    default 28-day burn-in this changes nothing for delays up to 14 days.
    """
    _, sampled = TREE.sampled_training_rows(train, negative_rate, burn_in_days, seed)
    fit_frame, stop_frame = TREE.early_stopping_split(sampled, seed)
    first_fitted = train["time_bin"].min() + pd.Timedelta(days=delay_days + minimum_history_days)
    return [
        frame.loc[frame["time_bin"] >= first_fitted] for frame in (sampled, fit_frame, stop_frame)
    ]


def fit_scorers(
    train, levels, delay_days, logistic_c, negative_rate, burn_in_days, minimum_history_days, seed
):
    """Fit the logistic regression and LightGBM as scripts 8 and 10 do.

    Returns a function per model that turns a frame into crash probabilities.
    """
    sampled, fit_frame, stop_frame = fitting_rows(
        train, delay_days, negative_rate, burn_in_days, minimum_history_days, seed
    )
    print(f"  fitting on {len(sampled):,} sampled rows from {sampled['time_bin'].min():%Y-%m-%d}")

    logistic = LOGREG.build_model(logistic_c)
    logistic.fit(sampled[LOGREG.FEATURES], sampled[TARGET].to_numpy())

    tree = TREE.build_model(TREE_SETTINGS, seed)
    TREE.fit_model(
        tree,
        TREE.model_inputs(fit_frame, TREE_FEATURES, levels),
        fit_frame[TARGET].to_numpy(),
        TREE.model_inputs(stop_frame, TREE_FEATURES, levels),
        stop_frame[TARGET].to_numpy(),
    )
    return {
        "baseline": lambda frame: expit(frame[BASELINE_COLUMN].to_numpy(dtype=np.float64)),
        "logistic": TREE.logistic_scorer({
            "model": logistic,
            "features": LOGREG.FEATURES,
            "negative_sample_rate": negative_rate,
        }),
        "lightgbm": TREE.tree_scorer(tree, TREE_FEATURES, levels, negative_rate),
    }


def score_delays(
    train,
    validation,
    delays=DELAYS_DAYS,
    logistic_c=0.1,
    negative_rate=0.1,
    burn_in_days=28,
    minimum_history_days=MINIMUM_HISTORY_DAYS,
    prior_strength=20.0,
    recent_window_days=28,
    seed=42,
):
    """Validation probabilities for every model and delay.

    Returns {(model, delay_days, fit): probabilities}. `fit` is "refit" when the
    model was fitted with the same delay, or "frozen" when the model fitted
    without a delay is scored on the delayed features.
    """
    delays = sorted(set(delays))
    if not delays or delays[0] != 0:
        raise ValueError("The delays must include 0, the reference for every change")
    years = sorted(validation["time_bin"].dt.year.unique())
    if years != [VALIDATION_YEAR]:
        raise ValueError(f"Expected only {VALIDATION_YEAR} validation rows, found years {years}")

    train, validation = LOGREG.add_history_features(
        [train, validation], prior_strength, recent_window_days
    )
    levels = TREE.category_levels(train)
    scores = {}
    frozen = None
    for delay in delays:
        print(f"Fitting and scoring with a {delay}-day delay...")
        delayed_train, delayed_validation = delay_history([train, validation], delay)
        scorers = fit_scorers(
            delayed_train, levels, delay, logistic_c, negative_rate, burn_in_days,
            minimum_history_days, seed,
        )
        for model, score in scorers.items():
            scores[(model, delay, "refit")] = score(delayed_validation)
        if delay == 0:
            frozen = scorers
        else:
            for model in ["logistic", "lightgbm"]:
                scores[(model, delay, "frozen")] = frozen[model](delayed_validation)
    return scores


def bootstrap_changes(y_true, days, scores, repeats, seed):
    """Day-block bootstrap of every score's change from the same model's 0-day score.

    Returns {key: array of (average precision change, top-5% capture change)}.
    Both delays are measured on the same resampled days, so the interval is for
    the change itself and not for two separate estimates.
    """
    y_true = np.asarray(y_true, dtype=bool)
    day_codes, unique_days = pd.factorize(days)
    day_codes = day_codes.astype(np.int16)
    ranked = {}
    for key, probability in scores.items():
        order, group_end = COMPARE.ranked(probability, y_true)
        ranked[key] = (y_true[order], day_codes[order], group_end)

    def resampled_scores(key, day_weights):
        y_sorted, day_sorted, group_end = ranked[key]
        return COMPARE.weighted_scores(y_sorted, day_weights[day_sorted], group_end)

    random = np.random.default_rng(seed)
    draws = {key: np.empty((repeats, 2)) for key in scores}
    # Threads help because the time goes to large NumPy sums, which release the lock.
    with ThreadPoolExecutor() as pool:
        for repeat in range(repeats):
            day_weights = np.bincount(
                random.integers(0, len(unique_days), len(unique_days)), minlength=len(unique_days)
            ).astype(np.float64)
            results = pool.map(lambda key: resampled_scores(key, day_weights), ranked)
            for key, result in zip(ranked, results):
                draws[key][repeat] = result
    return {key: draws[key] - draws[(key[0], 0, "refit")] for key in scores}


def measure(y_true, days, scores, repeats, seed):
    """One row per model, delay, and fit: the shared metrics and their change from 0 days."""
    metrics = {key: LOGREG.evaluate(y_true, probability) for key, probability in scores.items()}
    print(f"Drawing {repeats} day-block bootstrap samples...")
    changes = bootstrap_changes(y_true, days, scores, repeats, seed)
    rows = []
    for (model, delay, fit), result in metrics.items():
        reference = metrics[(model, 0, "refit")]
        low, high = np.quantile(changes[(model, delay, fit)], [0.025, 0.975], axis=0)
        rows.append({
            "model": model,
            "delay_days": delay,
            "fit": fit,
            "n_rows": result["rows"],
            "n_positive": int(np.sum(y_true)),
            "average_precision": result["average_precision"],
            "roc_auc": result["roc_auc"],
            "top5_capture": result["recall_at_top_fraction"],
            "average_precision_change": result["average_precision"] - reference["average_precision"],
            "average_precision_change_low_95": low[0],
            "average_precision_change_high_95": high[0],
            "roc_auc_change": result["roc_auc"] - reference["roc_auc"],
            "top5_capture_change": result["recall_at_top_fraction"] - reference["recall_at_top_fraction"],
            "top5_capture_change_low_95": low[1],
            "top5_capture_change_high_95": high[1],
        })
    order = {name: position for position, (name, _, _) in enumerate(MODELS)}
    details = pd.DataFrame(rows)
    details["model_order"] = details["model"].map(order)
    details = details.sort_values(["fit", "model_order", "delay_days"], ascending=[False, True, True])
    return details.drop(columns="model_order").reset_index(drop=True)


def shared_csv_rows(details, repeats, burn_in_days=28, minimum_history_days=MINIMUM_HISTORY_DAYS):
    """Refit rows in the format of reports/evaluation/evaluation_results_template.csv."""
    labels = {name: label for name, label, _ in MODELS}
    frozen = details[details["fit"] == "frozen"].set_index(["model", "delay_days"])
    rows = []
    for row in details[details["fit"] == "refit"].itertuples():
        notes = [labels[row.model]]
        if row.delay_days == 0:
            notes.append("no delay: history uses every earlier six-hour period; reference for the other delays")
        else:
            notes.append(f"crash-history features use only outcomes at least {row.delay_days} days old")
            if row.model == "baseline":
                notes.append("no fitted model")
            else:
                notes.append("refit on 2021-22 with the same delay")
                first_fitted = row.delay_days + minimum_history_days
                if first_fitted > burn_in_days:
                    notes.append(
                        f"fitted rows start {first_fitted} days into 2021 instead of {burn_in_days} "
                        f"so each keeps at least {minimum_history_days} days of visible history"
                    )
            notes.append(
                f"AP change vs 0-day delay {row.average_precision_change:+.5f} "
                f"(95% CI {row.average_precision_change_low_95:+.5f} to "
                f"{row.average_precision_change_high_95:+.5f}; {repeats} day-block bootstrap samples)"
            )
            if (row.model, row.delay_days) in frozen.index:
                unrefit = frozen.loc[(row.model, row.delay_days), "average_precision"]
                notes.append(f"AP {unrefit:.6f} if the 0-day model is not refit")
        rows.append({
            "experiment": "reporting_delay",
            "model": row.model,
            "group": f"{row.delay_days}_day_delay",
            "average_precision": round(row.average_precision, 6),
            "roc_auc": round(row.roc_auc, 6),
            "top5_capture": round(row.top5_capture, 6),
            "top5_scope": "global",
            "n_rows": row.n_rows,
            "n_positive": row.n_positive,
            "split": "2023_validation",
            "status": "complete",
            "notes": "; ".join(notes),
        })
    return pd.DataFrame(rows)


def plot_delays(details, path):
    """Average precision by delay, and its change from no delay with the 95% interval."""
    refit = details[details["fit"] == "refit"]
    delays = sorted(refit["delay_days"].unique())
    later = [delay for delay in delays if delay > 0]
    figure, (levels, changes) = plt.subplots(1, 2, figsize=(10.0, 4.8), facecolor=PLOTS.SURFACE)
    marker_style = {"markersize": 7, "markeredgecolor": PLOTS.SURFACE, "markeredgewidth": 1.5}
    spacing = 0.2
    largest_loss = 0.0
    for position, (name, label, color) in enumerate(MODELS):
        rows = refit[refit["model"] == name].sort_values("delay_days")
        precision = rows["average_precision"].to_numpy()
        levels.plot(rows["delay_days"], precision, color=color, linewidth=2,
                    marker=MARKERS[name], label=label, **marker_style)
        levels.annotate(f"{precision[-1]:.4f}", (delays[-1], precision[-1]),
                        textcoords="offset points", xytext=(9, 0), va="center",
                        fontsize=9, color=PLOTS.INK_SECONDARY)

        # Changes as a share of the model's own no-delay average precision.
        delayed = rows[rows["delay_days"] > 0]
        percent = {
            column: delayed[f"average_precision_change{column}"].to_numpy() / precision[0] * 100
            for column in ["", "_low_95", "_high_95"]
        }
        x = np.arange(len(later)) + (position - (len(MODELS) - 1) / 2) * spacing
        changes.vlines(x, percent["_low_95"], percent["_high_95"], color=color,
                       linewidth=3, alpha=0.45, capstyle="round")
        changes.plot(x, percent[""], color=color, linestyle="none", marker=MARKERS[name], **marker_style)
        largest_loss = max(largest_loss, -percent[""].min())

    levels.set_xticks(delays)
    # Room on the right for the value labels at the longest delay.
    levels.set_xlim(-0.06 * delays[-1], 1.19 * delays[-1])
    changes.axhline(0, color=PLOTS.INK_MUTED, linewidth=1)
    changes.set_xticks(range(len(later)))
    changes.set_xticklabels(later)
    changes.set_xlim(-0.5, len(later) - 0.5)
    changes.yaxis.set_major_formatter(
        lambda value, _: "0%" if abs(value) < 1e-9 else f"{value:+.1f}%".replace("-", "−")
    )
    # Relative, so a reader cannot take 0.2% for 0.2 points of average precision.
    changes.set_ylabel("% of no-delay average precision")
    for axes, title, subtitle in [
        (levels, "Average precision", "Vertical axis does not start at zero"),
        (changes, "Relative change from no delay", "Bar: 95% interval from resampling whole days"),
    ]:
        axes.set_xlabel("Reporting delay (days)")
        PLOTS.style_axes(axes, title, subtitle)
        axes.set_title(title, loc="left", color=PLOTS.INK, fontsize=10.5, fontweight="bold", pad=22)
        axes.grid(axis="x", visible=False)

    figure.legend(*levels.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.005, 0.86),
                  ncol=len(MODELS), frameon=False, fontsize=9, labelcolor=PLOTS.INK_SECONDARY,
                  handlelength=2.2)
    figure.suptitle(
        f"Crash reports up to {delays[-1]} days late cost at most {largest_loss:.2f}% of average precision",
        x=0.012, y=0.97, ha="left", fontsize=12, fontweight="bold", color=PLOTS.INK,
    )
    figure.text(
        0.012, 0.885,
        "2023 validation. Each model is refit with crash history that stops the given number of days "
        "before the period it predicts.",
        fontsize=9, color=PLOTS.INK_SECONDARY,
    )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.68, bottom=0.12, wspace=0.27)
    figure.savefig(path, dpi=200, facecolor=PLOTS.SURFACE)
    plt.close(figure)
    return path


def print_summary(details):
    print("\nREPORTING DELAY (2023 validation)")
    for row in details.itertuples():
        print(
            f"{row.model:>9} {row.delay_days:>2}-day {row.fit:>6}: AP={row.average_precision:.5f} "
            f"({row.average_precision_change:+.5f}, 95% {row.average_precision_change_low_95:+.5f} "
            f"to {row.average_precision_change_high_95:+.5f}) ROC-AUC={row.roc_auc:.4f} "
            f"top5={row.top5_capture:.4f} ({row.top5_capture_change:+.4f})"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--delays", default=",".join(str(delay) for delay in DELAYS_DAYS))
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--negative-sample-rate", type=float, default=0.1)
    parser.add_argument("--burn-in-days", type=int, default=28)
    parser.add_argument("--minimum-history-days", type=int, default=MINIMUM_HISTORY_DAYS)
    parser.add_argument("--prior-strength", type=float, default=20.0)
    parser.add_argument("--recent-window-days", type=int, default=28)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Redraw the figure from the saved details CSV without refitting.",
    )
    args = parser.parse_args()

    processed = args.root / "data" / "processed"
    output = args.output_directory or args.root / "reports" / "reporting_delay"
    (output / "figures").mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        details = pd.read_csv(output / DETAILS_CSV)
    else:
        delays = [int(value) for value in args.delays.split(",") if value.strip()]
        print("Loading training and validation files...")
        train = TREE.load_split(processed / LOGREG.TRAIN_FILE, "Training file")
        validation = TREE.load_split(processed / LOGREG.VALIDATION_FILE, "Validation file")
        y_true = validation[TARGET].to_numpy()
        days = validation["time_bin"].dt.normalize()

        scores = score_delays(
            train,
            validation,
            delays=delays,
            logistic_c=args.logistic_c,
            negative_rate=args.negative_sample_rate,
            burn_in_days=args.burn_in_days,
            minimum_history_days=args.minimum_history_days,
            prior_strength=args.prior_strength,
            recent_window_days=args.recent_window_days,
            seed=args.seed,
        )
        del train, validation
        details = measure(y_true, days, scores, args.bootstrap_repeats, args.seed)
        details.to_csv(output / DETAILS_CSV, index=False)
        shared_csv_rows(
            details, args.bootstrap_repeats, args.burn_in_days, args.minimum_history_days
        ).to_csv(output / SHARED_CSV, index=False)

    plot_delays(details, output / "figures" / FIGURE)
    print_summary(details)
    print(f"\nOutputs: {output}")


if __name__ == "__main__":
    main()
