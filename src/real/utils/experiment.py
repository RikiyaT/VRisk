"""
Generic experiment runner: loads saved .npy datasets, builds rankings
with utils.algo.*, evaluates CVaR / IA-M / raw M using utils.eval.*,
and writes a CSV with raw stats and normalized CIs.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

from utils import algo, eval


_DATA_PATHS = {
    "INTENT-1DRC": ("../../data/ntcir/iprobs_1drc.npy", "../../data/ntcir/qrels_1drc.npy"),
    "INTENT-1DR":  ("../../data/ntcir/iprobs_1dr.npy",  "../../data/ntcir/qrels_1dr.npy"),
    "INTENT-2DRC": ("../../data/ntcir/iprobs_2drc.npy", "../../data/ntcir/qrels_2drc.npy"),
    "INTENT-2DR":  ("../../data/ntcir/iprobs_2dr.npy",  "../../data/ntcir/qrels_2dr.npy"),
    "MOVIELENS2": ("../../data/ml/movielens_iprobs_full2.npy", "../../data/ml/movielens_qrels_full2.npy"),
    "MOVIELENS": ("../../data/ml/movielens_iprobs.npy", "../../data/ml/movielens_qrels.npy"),
    "WEB": ("../../data/web/web_iprobs.npy", "../../data/web/web_qrels.npy"),
}


def get_stats(scores: np.ndarray) -> tuple[float, float]:
    """
    Calculates the mean and the 95% confidence interval for a given array of scores.
    """
    n = len(scores)
    if n < 2:
        return np.mean(scores) if n == 1 else 0.0, 0.0

    mean = np.mean(scores)
    sem = st.sem(scores)
    ci = sem * st.t.ppf((1 + 0.95) / 2., n - 1)
    return mean, ci


def binarize_qrels(qrels: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Binarize qrels based on a threshold relative to the min-max range.
    
    Args:
        qrels: Original relevance judgments
        threshold: Threshold between 0 and 1 (default 0.5 for midpoint)
                  0.5 means halfway between min and max
                  0.3 means 30% of the way from min to max (more lenient)
                  0.7 means 70% of the way from min to max (more strict)
    
    Returns:
        Binary qrels (0 or 1)
    """
    # Find global min and max across all values
    min_val = qrels.min()
    max_val = qrels.max()
    
    # Calculate the threshold value
    if max_val > min_val:
        threshold_val = min_val + threshold * (max_val - min_val)
        # Binarize: 1 if above threshold, 0 otherwise
        binary_qrels = (qrels > threshold_val).astype(float)
    else:
        # If all values are the same, return zeros
        binary_qrels = np.zeros_like(qrels)
    
    return binary_qrels


def run_experiment(
    sweep_key: str,
    sweep_values,
    methods_cfg,
    global_cfg,
    out_dir: str,
    map_threshold: float = 0.5,  # Add parameter for MAP binarization threshold
):
    # -------- load data once --------------------------------------
    ip_path, qr_path = _DATA_PATHS[global_cfg["dataset"]]
    cprob = np.load(ip_path)
    qrels_original = np.load(qr_path)
    qrels_original = qrels_original.astype(float)
    
    if "MOVIELENS" in global_cfg["dataset"]:
        max_q = qrels_original.max()
        if max_q > 0:
            qrels_original /= max_q

    output_rows = []
    for val in sweep_values:
        cfg = global_cfg.copy()
        cfg[sweep_key] = val
        
        # -------- Binarize qrels if metric is MAP ----------------
        if cfg.get("metric") in ["map", "prec"]:
            qrels = binarize_qrels(qrels_original, threshold=map_threshold)
            print(f"[INFO] Binarizing qrels for MAP with threshold={map_threshold}")
            print(f"       Original range: [{qrels_original.min():.3f}, {qrels_original.max():.3f}]")
            print(f"       Binary stats: {(qrels == 1).sum()} relevant, {(qrels == 0).sum()} non-relevant")
        else:
            qrels = qrels_original.copy()

        per_query_results = {}

        for m_name in methods_cfg:
            # --- Start: Noise model logic ---
            c_noise_std = cfg.get("c_noise", 0.0)
            cprob_for_ranking = cprob
            if c_noise_std > 0:
                noise = np.random.normal(loc=0.0, scale=c_noise_std, size=cprob.shape)
                cprob_for_ranking = np.maximum(0, cprob + noise)
                row_sums = cprob_for_ranking.sum(axis=1, keepdims=True)
                np.divide(cprob_for_ranking, row_sums, out=cprob_for_ranking, where=row_sums!=0)
            # --- End: Noise model logic ---
            
            ranker = getattr(algo, m_name)
            rnk = ranker(
                cprob_for_ranking, qrels, k=cfg["k"], metric=cfg["metric"],
                beta=cfg.get("beta"), lamb=cfg.get("lamb"), tgt_lvl=cfg.get("tgt_lvl"),
            )

            # --- Get per-query scores for this method ---
            per_query_results[m_name] = {
                'CVaR': eval.cvar_vec(
                    rnk, cprob, qrels, metric=cfg["metric"], k=cfg["k"],
                    beta=cfg.get("beta", 0.1), tgt_lvl=cfg.get("tgt_lvl")),
                'IA_M': eval.ia_m_vec(
                    rnk, cprob, qrels, metric=cfg["metric"], k=cfg["k"]),
                'RAW_M': eval.m_raw_vec(
                    rnk, cprob, qrels, metric=cfg["metric"], k=cfg["k"]),
            }
        
        # --- Process results after all methods are run for this sweep value ---
        if 'naive_rank' not in per_query_results:
            print("[Warning] 'naive_rank' method not found in config. Cannot compute normalized CI.")
            baseline_scores = None
        else:
            baseline_scores = per_query_results['naive_rank']

        for m_name, method_scores in per_query_results.items():
            row = {'sweep': sweep_key, 'value': val, 'method': m_name}
            
            for metric in ['CVaR', 'IA_M', 'RAW_M']:
                # 1. Calculate and store stats on the raw scores
                raw_mean, raw_ci = get_stats(method_scores[metric])
                row[f'{metric}_mean'] = raw_mean
                row[f'{metric}_ci'] = raw_ci

                # 2. Calculate and store the CI of the normalized scores (ratios)
                if baseline_scores:
                    baseline = baseline_scores[metric]
                    ratios = np.divide(method_scores[metric], baseline, 
                                       out=np.ones_like(baseline), where=baseline!=0)
                    _, norm_ci = get_stats(ratios)
                    row[f'{metric}_norm_ci'] = norm_ci
                else:
                    row[f'{metric}_norm_ci'] = 0.0
            
            output_rows.append(row)

    df = pd.DataFrame(output_rows)
    
    # -------- ensure output dir and save CSV ---------------------
    out_dir = Path(out_dir) / f"{global_cfg['metric']}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename_part = f"{sweep_key}_rel" if sweep_key != 'metric' else "metric_sweep"
    csv_path = out_dir / f"{filename_part}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[✓] CSV with raw means, raw CIs, and normalized CIs saved: {csv_path}")

    # -------- Generate Raw Plots with Confidence Intervals --------
    metric_suffix = f"_{global_cfg['metric']}" if sweep_key != 'metric' else ""

    for metric_name, y_label_str in [
        ('CVaR', 'Mean CVaR (with 95% CI) ↓'),
        ('IA_M', f"Mean IA-{global_cfg['metric'].upper()} (with 95% CI) ↑"),
        ('RAW_M', f"Mean RAW-{global_cfg['metric'].upper()} (with 95% CI) ↑")
    ]:
        plt.figure(figsize=(8, 6))
        pivot_mean = df.pivot(index="value", columns="method", values=f"{metric_name}_mean")
        pivot_ci = df.pivot(index="value", columns="method", values=f"{metric_name}_ci")

        for method in pivot_mean.columns:
            means = pivot_mean[method]
            cis = pivot_ci[method]
            plt.plot(means.index, means, "o-", label=method, linewidth=2, markersize=8)
            plt.fill_between(means.index, means - cis, means + cis, alpha=0.2)
        
        plt.xlabel(sweep_key.capitalize())
        plt.ylabel(y_label_str)
        plt.title(f"Absolute Performance | {global_cfg['dataset']}")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        
        plot_path = out_dir / f"{metric_name.lower()}{metric_suffix}.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"[✓] Raw {metric_name} plot with CI saved: {plot_path}")



def run_map_threshold_experiment(
    sweep_key: str,
    sweep_values,
    methods_cfg,
    global_cfg,
    out_dir: str,
    thresholds: list = [0.3, 0.4, 0.5, 0.6, 0.7],
):
    """
    Run experiments with different MAP binarization thresholds.
    This is useful for finding the optimal threshold for your dataset.
    """
    results = {}
    
    for threshold in thresholds:
        print(f"\n[EXPERIMENT] Running with MAP threshold = {threshold}")
        
        # Create a unique output directory for this threshold
        threshold_out_dir = f"{out_dir}_threshold_{threshold:.1f}"
        
        # Run the experiment with this threshold
        run_experiment(
            sweep_key=sweep_key,
            sweep_values=sweep_values,
            methods_cfg=methods_cfg,
            global_cfg=global_cfg,
            out_dir=threshold_out_dir,
            map_threshold=threshold
        )
        
        results[threshold] = threshold_out_dir
    
    print(f"\n[COMPLETE] All threshold experiments completed.")
    print("Results saved in:")
    for thresh, dir_path in results.items():
        print(f"  Threshold {thresh:.1f}: {dir_path}")
    
    return results
