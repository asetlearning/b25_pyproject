# Created by: Alex Myasnikov
# Date: June 8, 2026

"""Marimo dashboard for PatternBoost hyperparameter sweep analysis."""

import marimo

__generated_with = "0.9.0"
app = marimo.App(width="medium")


@app.cell
def _imports():
    import json
    import yaml
    from datetime import datetime
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import plotly.express as px

    return datetime, json, mo, pd, px, yaml, Path


@app.cell
def _inputs(mo):
    folder_input = mo.ui.text(
        value="output/pb_sweep/sweep_20260522_013909",
        label="Sweep root directory",
        full_width=True,
    )
    config_input = mo.ui.text(
        value="configs/pattern_boost_sweep_1.yaml",
        label="Sweep config YAML (defines swept parameters)",
        full_width=True,
    )
    mo.vstack([folder_input, config_input])
    return (folder_input, config_input)


@app.cell
def _parse(folder_input, config_input, json, yaml, datetime, pd, Path):
    def _get_nested(d, dotted_key):
        for k in dotted_key.split("."):
            d = d[k]
        return d

    # Load swept param keys from config YAML
    _param_keys = []
    _cfg_path = Path(config_input.value.strip()) if config_input.value.strip() else None
    if _cfg_path and _cfg_path.exists():
        with _cfg_path.open() as f:
            _sweep_cfg = yaml.safe_load(f)
        _param_keys = list(_sweep_cfg.get("params", {}).keys())

    # Short column name = last segment of dot-path (e.g. "local_search.beam_width" → "beam_width")
    param_cols = [k.split(".")[-1] for k in _param_keys]

    def _parse_trial(trial_dir: Path):
        """Returns (matched, first_iter, time_s, scores).
        scores: list of all match scores in this trial (empty if no matches)."""
        match_file = trial_dir / "run.matches.jsonl"
        if not match_file.exists() or match_file.stat().st_size == 0:
            return False, None, None, []
        matches = [json.loads(ln) for ln in match_file.read_text().splitlines() if ln.strip()]
        if not matches:
            return False, None, None, []
        with (trial_dir / "run.jsonl").open() as f:
            start_rec = json.loads(f.readline())
        first_iter = matches[0]["iteration"]
        time_s = (
            datetime.fromisoformat(matches[0]["timestamp"])
            - datetime.fromisoformat(start_rec["timestamp"])
        ).total_seconds()
        return True, first_iter, time_s, [m["score"] for m in matches]

    def _parse_sweep(root: Path) -> pd.DataFrame:
        rows = []
        for combo_dir in sorted(root.iterdir()):
            if not combo_dir.is_dir() or not combo_dir.name.startswith("combo_"):
                continue
            idx = int(combo_dir.name.split("_")[1])
            trial_dirs = [d for d in sorted(combo_dir.iterdir())
                          if d.is_dir() and d.name.startswith("trial_")]
            if not trial_dirs:
                continue

            # Read param values from first trial's pb_start event
            _params = {}
            if _param_keys:
                with (trial_dirs[0] / "run.jsonl").open() as f:
                    _start_rec = json.loads(f.readline())
                _cfg = _start_rec.get("config", {})
                for pk, col in zip(_param_keys, param_cols):
                    try:
                        _params[col] = _get_nested(_cfg, pk)
                    except (KeyError, TypeError):
                        _params[col] = None

            label = (
                "_".join(f"{c}={_params[c]}" for c in param_cols)
                if _params else combo_dir.name
            )

            n_trials = len(trial_dirs)
            matched_iters, matched_times, all_scores = [], [], []
            for td in trial_dirs:
                ok, it, ts, scores = _parse_trial(td)
                if ok:
                    matched_iters.append(it)
                    matched_times.append(ts)
                    all_scores.extend(scores)
            n_matched = len(matched_iters)
            total_matches = len(all_scores)
            rows.append({
                "combo_idx": idx,
                "combo_label": label,
                **_params,
                "n_trials": n_trials,
                "n_matched": n_matched,
                "success_rate_pct": round(100 * n_matched / n_trials, 1) if n_trials else float("nan"),
                "avg_matches_per_trial": round(total_matches / n_trials, 2) if n_trials else float("nan"),
                "avg_match_score": round(sum(all_scores) / total_matches, 4) if all_scores else float("nan"),
                "avg_iter_to_match": round(sum(matched_iters) / len(matched_iters), 2) if matched_iters else float("nan"),
                "avg_time_to_match_s": round(sum(matched_times) / len(matched_times), 1) if matched_times else float("nan"),
            })
        return pd.DataFrame(rows)

    _root = Path(folder_input.value.strip())
    df = _parse_sweep(_root) if _root.exists() else pd.DataFrame()
    return (df, param_cols)


@app.cell
def _table(df, param_cols, mo):
    _base_cols = ["combo_label"] + param_cols + [
        "n_trials", "n_matched", "success_rate_pct",
        "avg_matches_per_trial", "avg_match_score",
        "avg_iter_to_match", "avg_time_to_match_s",
    ]
    _cols = [c for c in _base_cols if c in df.columns]
    mo.vstack([
        mo.md("## Summary table"),
        mo.ui.table(df[_cols] if not df.empty else df),
    ])


@app.cell
def _color_selector(param_cols, mo):
    color_selector = mo.ui.dropdown(
        options=["(none)"] + param_cols,
        value=param_cols[-1] if param_cols else "(none)",
        label="Color bars by",
    )
    color_selector
    return (color_selector,)


@app.cell
def _chart_success(df, color_selector, px, mo):
    if df.empty:
        _out = mo.md("No data loaded.")
    else:
        _color = color_selector.value if color_selector.value != "(none)" else None
        _plot_df = df.sort_values("combo_idx").copy()
        _order = {"combo_label": _plot_df["combo_label"].tolist()}
        if _color:
            _plot_df[_color] = _plot_df[_color].astype(str)
        _fig = px.bar(
            _plot_df,
            x="combo_label",
            y="success_rate_pct",
            color=_color,
            category_orders=_order,
            title="Success rate % by parameter combo",
            labels={"combo_label": "Combo", "success_rate_pct": "Success rate (%)"},
            text_auto=True,
        )
        _fig.update_layout(xaxis_tickangle=-45)
        _out = mo.ui.plotly(_fig)
    _out
    return


@app.cell
def _chart_iterations(df, color_selector, px, mo):
    if df.empty:
        _out = mo.md("No data.")
    else:
        _color = color_selector.value if color_selector.value != "(none)" else None
        _plot_df = df.sort_values("combo_idx").copy()
        _order = {"combo_label": _plot_df["combo_label"].tolist()}
        if _color:
            _plot_df[_color] = _plot_df[_color].astype(str)
        _fig = px.bar(
            _plot_df,
            x="combo_label",
            y="avg_iter_to_match",
            color=_color,
            category_orders=_order,
            title="Avg iterations to first match (matched trials only)",
            labels={"combo_label": "Combo", "avg_iter_to_match": "Avg iterations"},
            text_auto=True,
        )
        _fig.update_layout(xaxis_tickangle=-45)
        _out = mo.ui.plotly(_fig)
    _out
    return


@app.cell
def _chart_time(df, color_selector, px, mo):
    if df.empty:
        _out = mo.md("No data.")
    else:
        _color = color_selector.value if color_selector.value != "(none)" else None
        _plot_df = df.sort_values("combo_idx").copy()
        _order = {"combo_label": _plot_df["combo_label"].tolist()}
        if _color:
            _plot_df[_color] = _plot_df[_color].astype(str)
        _fig = px.bar(
            _plot_df,
            x="combo_label",
            y="avg_time_to_match_s",
            color=_color,
            category_orders=_order,
            title="Avg time to first match, seconds (matched trials only)",
            labels={"combo_label": "Combo", "avg_time_to_match_s": "Avg time (s)"},
            text_auto=True,
        )
        _fig.update_layout(xaxis_tickangle=-45)
        _out = mo.ui.plotly(_fig)
    _out
    return


@app.cell
def _ranking_controls(mo):
    threshold_slider = mo.ui.slider(
        start=50, stop=100, step=5, value=90,
        label="Min success rate to include in ranking (%)",
        show_value=True,
    )
    threshold_slider
    return (threshold_slider,)


@app.cell
def _ranking(threshold_slider, df, param_cols, px, mo, pd):
    if df.empty:
        _out = mo.md("No data.")
    else:
        def _pareto_mask(df_in):
            """Non-dominated: no other point has >= success AND <= time (strictly better in one)."""
            result = []
            for i, ri in df_in.iterrows():
                si, ti = ri.success_rate_pct, ri.avg_time_to_match_s
                if pd.isna(ti):
                    result.append(False)
                    continue
                dominated = any(
                    rj.success_rate_pct >= si
                    and rj.avg_time_to_match_s <= ti
                    and (rj.success_rate_pct > si or rj.avg_time_to_match_s < ti)
                    for j, rj in df_in.iterrows()
                    if i != j and not pd.isna(rj.avg_time_to_match_s)
                )
                result.append(not dominated)
            return result

        _df = df.copy()
        _df["pareto"] = _pareto_mask(_df)
        _df["frontier"] = _df["pareto"].map({True: "Pareto-optimal", False: "Dominated"})

        _hover = [c for c in param_cols if c in _df.columns] + ["n_matched", "n_trials", "avg_iter_to_match"]

        _scatter = px.scatter(
            _df.dropna(subset=["avg_time_to_match_s"]),
            x="avg_time_to_match_s",
            y="success_rate_pct",
            color="frontier",
            color_discrete_map={"Pareto-optimal": "#27ae60", "Dominated": "#bdc3c7"},
            text="combo_label",
            title="Success rate vs avg time — upper-left is best; green = Pareto-optimal",
            labels={
                "avg_time_to_match_s": "Avg time to first match (s)",
                "success_rate_pct": "Success rate (%)",
            },
            hover_data=_hover,
        )
        _scatter.update_traces(textposition="top center", marker_size=10)
        _scatter.update_layout(legend_title_text="")

        _thresh = threshold_slider.value
        _ranked = (
            _df[_df["success_rate_pct"] >= _thresh]
            .sort_values("avg_time_to_match_s", ascending=False)
            .reset_index(drop=True)
        )
        _ranked.insert(0, "rank", range(len(_ranked), 0, -1))
        _rank_cols = (
            ["rank", "combo_label"]
            + [c for c in param_cols if c in _ranked.columns]
            + ["success_rate_pct", "avg_time_to_match_s", "avg_iter_to_match", "pareto"]
        )

        _out = mo.vstack([
            mo.md("---\n## Optimal parameter analysis"),
            threshold_slider,
            mo.md(f"### Ranked combos — success rate ≥ {_thresh}% (rank 1 = fastest)"),
            mo.ui.table(_ranked[_rank_cols].round(2)),
            mo.md("### Pareto frontier"),
            mo.ui.plotly(_scatter),
        ])
    _out
    return


if __name__ == "__main__":
    app.run()
