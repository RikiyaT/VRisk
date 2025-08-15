import inspect
import numpy as np
from typing import Optional, Dict, Any, Callable
from .eval import _cvar_one
from tqdm.auto import tqdm
import pulp
def tie_break_by_mean_score(is_min: np.ndarray, mean_scores: np.ndarray, **kwargs: Dict[str, Any]) -> int:
    """
    Tie-break by selecting the candidate with the highest mean score.
    This is the original method.
    """
    tied_scores = np.where(is_min, mean_scores, -np.inf)
    return np.argmax(tied_scores)

def tie_break_randomly(is_min: np.ndarray, **kwargs: Dict[str, Any]) -> int:
    """
    Tie-break by selecting a random candidate from the tied set.
    """
    tied_indices = np.where(is_min)[0]
    return np.random.choice(tied_indices)


def _rel_2d(rels: np.ndarray) -> np.ndarray:
    """Vectorized mean relevance for a 2D array."""
    if rels.shape[1] == 0:
        return np.zeros(rels.shape[0])
    return rels.mean(axis=1)
def _sumrel_2d(rels: np.ndarray) -> np.ndarray:
    """Vectorized sum of relevance for a 2D array."""
    return rels.sum(axis=1)
def _prec_2d(rels: np.ndarray) -> np.ndarray:
    """Vectorized Precision@k for a 2D array."""
    if rels.shape[1] == 0:
        return np.zeros(rels.shape[0])
    return (rels > 0).mean(axis=1)
def _mrr_2d(rels: np.ndarray) -> np.ndarray:
    """Vectorized Mean Reciprocal Rank for a 2D array."""
    bin_rel = rels >= 0.5
    # Find the index of the first relevant document for each row (intent)
    ranks = np.argmax(bin_rel, axis=1)
    # If no relevant doc is found for a row, argmax returns 0.
    # We need to check if a relevant doc actually exists at that position.
    has_rel = bin_rel[np.arange(len(ranks)), ranks]
    # Calculate reciprocal rank, which is 0 if no relevant doc was found.
    reciprocal_ranks = has_rel / (ranks + 1.0)
    return reciprocal_ranks
def _err_2d(rels: np.ndarray, max_rels: np.ndarray) -> np.ndarray:
    """Vectorized Expected Reciprocal Rank for a 2D array of rankings."""
    if rels.shape[1] == 0:
        return np.zeros(rels.shape[0])
    # max_rels should be (C, 1) to broadcast correctly
    prob_stop = (np.power(2.0, rels) - 1.0) / (2.0 ** max_rels)
    prob_stop = np.clip(prob_stop, 0, 1)
    prob_continue_all = np.cumprod(1 - prob_stop, axis=1)
    prob_continue = np.insert(prob_continue_all[:, :-1], 0, 1.0, axis=1)
    
    ranks = np.arange(1, rels.shape[1] + 1)
    err = (prob_continue * prob_stop / ranks).sum(axis=1)
    return err
def _rbp_2d(rels: np.ndarray, p: float, max_rels: np.ndarray) -> np.ndarray:
    """Vectorized Rank-Biased Precision for a 2D array."""
    if rels.shape[1] == 0:
        return np.zeros(rels.shape[0])
    
    gains = np.divide(rels, max_rels, out=np.zeros_like(rels, dtype=float), where=max_rels!=0)
    weights = (1 - p) * (p ** np.arange(rels.shape[1]))
    return (gains * weights).sum(axis=1)
def _dcg_2d(rels: np.ndarray, disc: np.ndarray) -> np.ndarray:
    """Vectorized DCG for a 2D array."""
    k = rels.shape[1]
    return (rels * disc[:k]).sum(axis=1)
def _dcg_vec(rels: np.ndarray, disc: np.ndarray, **kwargs) -> float:
    """Linear DCG. rels length <= len(disc)"""
    return float((rels * disc[: len(rels)]).sum())
def _edcg_2d(rels: np.ndarray, disc: np.ndarray) -> np.ndarray:
    """Vectorized Exponential DCG for a 2D array."""
    k = rels.shape[1]
    exp_gains = np.power(2.0, rels) - 1.0
    return (exp_gains * disc[:k]).sum(axis=1)
def _ndcg_2d(rels: np.ndarray, disc: np.ndarray, idcgs: np.ndarray) -> np.ndarray:
    """Vectorized nDCG. Takes a vector of pre-computed IDCGs."""
    dcgs = _dcg_2d(rels, disc)
    return np.divide(dcgs, idcgs, out=np.zeros_like(dcgs), where=idcgs!=0)
def _endcg_2d(rels: np.ndarray, disc: np.ndarray, eidcgs: np.ndarray) -> np.ndarray:
    """Vectorized eNDCG. Takes a vector of pre-computed eIDCGs."""
    edcgs = _edcg_2d(rels, disc)
    return np.divide(edcgs, eidcgs, out=np.zeros_like(edcgs), where=eidcgs!=0)
def _cvar_one(loss: np.ndarray, w: np.ndarray, beta: float) -> float:
    """Helper to calculate CVaR for one set of losses."""
    if beta >= 1.0: return float(np.average(loss, weights=w))
    if beta <= 0.0: return float(loss.max())
    
    q = np.quantile(loss, 1 - beta)
    tail_losses = loss[loss >= q]
    tail_weights = w[loss >= q]
    
    if tail_weights.sum() == 0: return q
    return np.average(tail_losses, weights=tail_weights)
def _map_2d(rels: np.ndarray) -> np.ndarray:
    """
    Vectorized Mean Average Precision for binary relevance.
    
    Assumes rels are already binary (0 or 1).
    """
    # No normalization needed - already binary
    bin_rel = rels.astype(bool)
    num_rel = bin_rel.sum(axis=1)
    
    csum = np.cumsum(bin_rel, axis=1)
    ranks = np.arange(1, rels.shape[1] + 1)
    prec = csum / ranks
    
    ap = (prec * bin_rel).sum(axis=1)
    return np.divide(ap, num_rel, out=np.zeros_like(ap, dtype=float), where=num_rel!=0)

def _oracle_score_intent(
    rel_row: np.ndarray, metric: str, k: int, disc: np.ndarray, p: float = 0.8
) -> float:
    """
    Best possible score for one intent, top-k docs.
    
    For MAP and Precision, assumes rel_row is already binary.
    """
    if rel_row.size < k: k = rel_row.size
    top_k_indices = np.argpartition(-rel_row, k - 1)[:k] if k > 0 else []
    rels = rel_row[top_k_indices]
    rels_sorted = np.sort(rels)[::-1]
    
    rels_2d = rels_sorted[None, :]
    if metric == "map": 
        return _map_2d(rels_2d)[0]
    if metric == "prec":
        # Simplified: no min/max needed for binary relevance
        return _prec_2d(rels_2d)[0]
    
    # Other metrics remain the same
    max_rel_metric = rel_row.max()
    if max_rel_metric == 0: max_rel_metric = 4.0
    max_rels_2d = np.array([[max_rel_metric]])
    
    if metric == "rel": return _rel_2d(rels_2d)[0]
    if metric == "sumrel": return _sumrel_2d(rels_2d)[0]
    if metric == "mrr": return _mrr_2d(rels_2d)[0]
    if metric == "dcg": return _dcg_2d(rels_2d, disc)[0]
    if metric == "ndcg": return 1.0 if _dcg_2d(rels_2d, disc)[0] > 0 else 0.0
    if metric == "endcg": return 1.0 if _edcg_2d(rels_2d, disc)[0] > 0 else 0.0
    if metric == "err": return _err_2d(rels_2d, max_rels_2d)[0]
    if metric == "rbp": return _rbp_2d(rels_2d, p, max_rels_2d)[0]
    return 0.0

def ours(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    metric: str = "rel",
    k: int,
    beta: float = 0.1,
    p: float = 0.8,
    baseline_rnk: Optional[np.ndarray] = None,
    tgt_lvl: Optional[float] = None,
    filter_docs: bool = False,
    tie_breaker: Callable[..., int] = tie_break_by_mean_score,
    **kwargs,
) -> np.ndarray:
    """
    Greedy MINIMISER of CVaR of the loss L(c) = [M0(c) − M(rank | c)]_+
    
    Now accepts a `tie_breaker` function to resolve ties when multiple
    documents yield the same minimum CVaR.
    
    For MAP and Precision, assumes qrels are already binary.
    """
    # Robustness Fix: Check for and correct qrels dimension order
    if qrels.ndim == 3 and cprob.ndim == 2 and qrels.shape[0] == cprob.shape[0] and qrels.shape[2] == cprob.shape[1]:
        qrels = qrels.transpose(0, 2, 1)
    if beta is None:
        beta = 0.1
    T, C, D = qrels.shape
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ranking = -np.ones((T, k), dtype=np.int32)
    # Process each query
    for t in tqdm(range(T), desc=f"Ours (Vectorized CVaR, {metric})"):
        w_c = cprob[t] / cprob[t].sum() if cprob[t].sum() > 0 else np.full(C, 1.0 / C)
        r_tc = qrels[t]  # Shape: (C, D)
        if filter_docs:
            # Filter out non-relevant documents
            non_zero_mask = np.any(r_tc > 0, axis=0)  # Check along intent axis
            valid_docs = np.where(non_zero_mask)[0]
            
            if len(valid_docs) == 0:
                ranking[t, :] = np.arange(min(k, D))
                continue
        else:
            valid_docs = np.arange(D)
        
        n_valid = len(valid_docs)
        r_tc_filtered = r_tc[:, valid_docs]  # Shape: (C, n_valid)
        # Pre-computation for M0 (Target Score)
        M0 = np.empty(C, dtype=np.float32)
        max_rels_t = r_tc.max(axis=1, keepdims=True)
        max_rels_t[max_rels_t.squeeze() == 0] = 4.0
        # NDCG: Match the first version exactly - use full sort on unfiltered r_tc
        if metric == 'ndcg':
            idcgs_t = np.ones(C, dtype=np.float32)  # Default to 1 to avoid division by zero
            for c in range(C):
                ideal_oracle_rels = np.sort(r_tc[c])[::-1][:k]
                idcgs_t[c] = _dcg_vec(ideal_oracle_rels, disc)
        elif metric == 'endcg': 
            top_k_rels = np.partition(r_tc, -min(k, r_tc.shape[1]), axis=1)[:, -min(k, r_tc.shape[1]):]
            eidcgs_t = _edcg_2d(np.sort(top_k_rels, axis=1)[:, ::-1], disc)
        
        # M0 calculation
        if baseline_rnk is not None:
            rels_b = r_tc[:, baseline_rnk[t, :k]]
            if metric == 'ndcg': M0 = _ndcg_2d(rels_b, disc, idcgs_t)
            elif metric == 'endcg': M0 = _endcg_2d(rels_b, disc, eidcgs_t)
            elif metric == 'map':
                M0 = _map_2d(rels_b)
            elif metric == 'prec':
                # Simplified: no normalization needed for binary qrels
                M0 = _prec_2d(rels_b)
            else: M0 = globals()[f"_{metric}_2d"](rels_b, p=p, max_rels=max_rels_t, disc=disc) if metric in ['rbp','err','dcg','edcg'] else globals()[f"_{metric}_2d"](rels_b)
        else:
            for c in range(C): 
                M0[c] = _oracle_score_intent(r_tc[c], metric, k, disc, p)
            if tgt_lvl is not None and 0 < tgt_lvl <= 1.0: M0 *= tgt_lvl
        # Ranking State Initialization
        available = np.ones(n_valid, dtype=bool)
        current_scores_c = np.zeros(C, dtype=np.float32)
        
        # Metric-specific state
        if metric in ['rel', 'sumrel']: 
            state = {'sum': np.zeros(C, dtype=np.float32)}
        elif metric == 'prec': 
            state = {'count': np.zeros(C, dtype=np.float32)}
        elif metric == 'err': 
            state = {'prob_continue': np.ones(C, dtype=np.float32)}
        elif metric == 'map': 
            num_rel = r_tc_filtered.sum(axis=1)  # Already binary
            state = {
                'csum_rel': np.zeros(C), 
                'ap_sum': np.zeros(C), 
                'num_rel': num_rel
            }
        else: 
            state = {}
        # Greedy Selection Loop
        for pos in range(min(k, n_valid)):
            candidate_indices = np.where(available)[0]
            candidate_rels = r_tc_filtered[:, candidate_indices] # Shape: (C, num_candidates)
            n_candidates = len(candidate_indices)
            # Vectorized score calculation for all candidates
            if metric in ['dcg', 'edcg', 'ndcg', 'endcg']:
                if metric == 'dcg': delta = candidate_rels * disc[pos]
                elif metric == 'edcg': delta = (np.power(2.0, candidate_rels) - 1.0) * disc[pos]
                elif metric == 'ndcg': delta = np.divide(candidate_rels * disc[pos], idcgs_t[:, np.newaxis], out=np.zeros_like(candidate_rels), where=idcgs_t[:, np.newaxis]!=0)
                else: delta = np.divide((np.power(2.0, candidate_rels) - 1.0) * disc[pos], eidcgs_t[:, np.newaxis], out=np.zeros_like(candidate_rels), where=eidcgs_t[:, np.newaxis]!=0)
                trial_scores_all = current_scores_c[:, np.newaxis] + delta
            
            elif metric == 'rbp':
                gain = np.divide(candidate_rels, max_rels_t, out=np.zeros_like(candidate_rels), where=max_rels_t!=0)
                delta = (1 - p) * gain * (p ** pos)
                trial_scores_all = current_scores_c[:, np.newaxis] + delta
            
            elif metric == 'err':
                prob_stop_d = np.divide(np.power(2.0, candidate_rels) - 1.0, np.power(2.0, max_rels_t) - 1.0, out=np.zeros_like(candidate_rels), where=max_rels_t!=0)
                prob_stop_d = np.clip(prob_stop_d, 0, 1)
                delta = (state['prob_continue'][:, np.newaxis] * prob_stop_d) / (pos + 1.0)
                trial_scores_all = current_scores_c[:, np.newaxis] + delta
            elif metric == 'rel': 
                trial_scores_all = (state['sum'][:, np.newaxis] + candidate_rels) / (pos + 1.0)
            elif metric == 'sumrel': 
                trial_scores_all = state['sum'][:, np.newaxis] + candidate_rels
            elif metric == 'prec':
                # Simplified: qrels are already binary, so `> 0` is not needed.
                trial_scores_all = (state['count'][:, np.newaxis] + candidate_rels.astype(bool)) / (pos + 1.0)
            
            elif metric == 'mrr':
                is_rel_all = (candidate_rels > 0)
                trial_scores_all = np.where(current_scores_c[:, np.newaxis] > 0, current_scores_c[:, np.newaxis], is_rel_all / (pos + 1.0))
            elif metric == 'map':
                is_rel_all = candidate_rels.astype(bool)
                trial_csum_rel = state['csum_rel'][:, np.newaxis] + is_rel_all
                prec_at_pos = trial_csum_rel / (pos + 1.0)
                delta_ap_sum = prec_at_pos * is_rel_all
                trial_ap_sum = state['ap_sum'][:, np.newaxis] + delta_ap_sum
                trial_scores_all = np.divide(trial_ap_sum, state['num_rel'][:, np.newaxis], out=np.zeros_like(trial_ap_sum), where=state['num_rel'][:, np.newaxis]!=0)
            
            # Vectorized CVaR computation
            loss_all = np.maximum(0, M0[:, np.newaxis] - trial_scores_all)  # Shape: (C, n_candidates)
            
            # Vectorized CVaR computation for all candidates at once
            if beta >= 1.0:
                risks = np.average(loss_all, weights=w_c, axis=0)
            elif beta <= 0.0:
                risks = loss_all.max(axis=0)
            else:
                threshold = 1.0 - beta
                w_c_expanded = w_c[:, np.newaxis]
                
                sort_indices = np.argsort(loss_all, axis=0)
                sorted_losses = np.take_along_axis(loss_all, sort_indices, axis=0)
                sorted_weights = np.take_along_axis(np.broadcast_to(w_c_expanded, (C, n_candidates)), sort_indices, axis=0)
                
                cumsum_weights = np.cumsum(sorted_weights, axis=0)
                threshold_idx = np.argmax(cumsum_weights >= threshold, axis=0)
                quantiles = sorted_losses[threshold_idx, np.arange(n_candidates)]
                
                tail_masks = loss_all >= quantiles
                numerator = np.sum(loss_all * tail_masks * w_c_expanded, axis=0)
                denominator = np.sum(tail_masks * w_c_expanded, axis=0)
                risks = np.where(denominator > 0, numerator / denominator, quantiles)
            
            mean_scores = np.sum(w_c[:, np.newaxis] * trial_scores_all, axis=0)
            # --- Selection ---
            min_risk = np.min(risks)
            tolerance = 1e-9 * np.abs(min_risk) if min_risk != 0 else 1e-9
            is_min = np.abs(risks - min_risk) < tolerance
            
            best_local_idx_in_candidates = tie_breaker(
                is_min=is_min, 
                mean_scores=mean_scores
            )
            
            best_doc_local_idx = candidate_indices[best_local_idx_in_candidates]
            
            # Update State with Best Document
            available[best_doc_local_idx] = False
            ranking[t, pos] = valid_docs[best_doc_local_idx]
            current_scores_c = trial_scores_all[:, best_local_idx_in_candidates]
            
            best_rels_d = r_tc_filtered[:, best_doc_local_idx]
            if metric in ['rel', 'sumrel']: 
                state['sum'] += best_rels_d
            elif metric == 'prec':
                # Simplified: qrels are already binary
                state['count'] += best_rels_d.astype(bool)
            elif metric == 'err':
                prob_stop_d = np.divide(np.power(2.0, best_rels_d) - 1.0, np.power(2.0, max_rels_t.squeeze()) - 1.0, out=np.zeros(C), where=max_rels_t.squeeze()!=0)
                state['prob_continue'] *= (1.0 - np.clip(prob_stop_d, 0, 1))
            elif metric == 'map':
                is_rel_d = best_rels_d.astype(bool)
                state['csum_rel'] += is_rel_d
                prec_at_pos = state['csum_rel'] / (pos + 1.0)
                state['ap_sum'] += (prec_at_pos * is_rel_d)
        # Fill remaining ranks if not enough valid docs were found
        if filter_docs and n_valid < k:
            non_zero_mask = np.any(r_tc > 0, axis=0)
            zero_docs = np.where(~non_zero_mask)[0]
            ranking[t, n_valid:] = zero_docs[:k - n_valid]
    return ranking.astype(np.int32)

def optimal_milp(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    metric: str,
    k: int,
    beta: float,
    tgt_lvl: Optional[float] = None,
    **kwargs
) -> np.ndarray:
    """
    Finds the truly optimal ranking by solving a Mixed-Integer Linear Program (MILP)
    for each query. This is the most robust and often most efficient way to find
    the true global optimum over all relevant documents.
    
    Updated to handle binary qrels for MAP and Precision metrics.
    """
    T, C, D = qrels.shape
    final_ranking = -np.ones((T, k), dtype=np.int32)
    disc = 1.0 / np.log2(np.arange(2, k + 2))

    for t in tqdm(range(T), desc="Optimal (MILP)"):
        qrels_t = qrels[t]
        cprob_t = cprob[t]

        # --- Pre-computation ---
        # 1. Only consider documents with rel > 0
        viable_docs_mask = qrels_t.sum(axis=0) > 0
        doc_indices = np.where(viable_docs_mask)[0]
        if len(doc_indices) == 0:
            final_ranking[t, :k] = np.arange(min(k, D))
            continue
        
        num_viable_docs = len(doc_indices)
        qrels_viable = qrels_t[:, doc_indices]

        # 2. Calculate baseline score M0(c)
        M0_c = np.empty(C)
        for c in range(C):
            M0_c[c] = _oracle_score_intent(qrels_t[c], metric, k, disc)
        if tgt_lvl is not None:
            M0_c *= tgt_lvl

        # --- MILP Problem Definition ---
        prob = pulp.LpProblem(f"CVaR_Optimal_Ranking_Q{t}", pulp.LpMinimize)

        # Decision variables
        x = pulp.LpVariable.dicts("x", (range(k), range(num_viable_docs)), cat='Binary')
        zeta = pulp.LpVariable("zeta", cat='Continuous')
        u = pulp.LpVariable.dicts("u", range(C), lowBound=0, cat='Continuous')

        # Objective function
        prob += zeta + (1.0 / beta) * pulp.lpSum([cprob_t[c] * u[c] for c in range(C)])

        # Constraints
        for c in range(C):
            # Calculate ranking score M(R|c)
            if metric == 'rel':
                m_rank_c = (1/k) * pulp.lpSum([
                    qrels_viable[c, d_idx] * x[i][d_idx] 
                    for i in range(k) for d_idx in range(num_viable_docs)
                ])
            
            elif metric == 'prec':
                # For binary precision: mean of binary relevance values
                # Since qrels are already binary, just compute mean
                m_rank_c = (1/k) * pulp.lpSum([
                    qrels_viable[c, d_idx] * x[i][d_idx] 
                    for i in range(k) for d_idx in range(num_viable_docs)
                ])
            
            elif metric == 'map':
                # WARNING: MAP cannot be exactly formulated as a linear program
                # This is an approximation. For exact MAP optimization, use 
                # branch-and-bound or exhaustive search for small problems.
                
                # Count total relevant documents for this intent
                num_rel_c = qrels_viable[c, :].sum()
                
                if num_rel_c > 0:
                    # Approximation: treat as weighted sum of relevances
                    # with position-based weights
                    weights = 1.0 / np.arange(1, k + 1)
                    m_rank_c = pulp.lpSum([
                        qrels_viable[c, d_idx] * weights[i] * x[i][d_idx] 
                        for i in range(k) for d_idx in range(num_viable_docs)
                    ]) / num_rel_c
                else:
                    m_rank_c = 0
            
            elif metric == 'ndcg':
                idcg_c = _dcg_vec(np.sort(qrels_t[c])[::-1][:k], disc)
                if idcg_c > 0:
                    m_rank_c = (1/idcg_c) * pulp.lpSum([
                        qrels_viable[c, d_idx] * disc[i] * x[i][d_idx] 
                        for i in range(k) for d_idx in range(num_viable_docs)
                    ])
                else:
                    m_rank_c = 0
            
            elif metric == 'endcg':
                eidcg_c = _edcg_vec(np.sort(qrels_t[c])[::-1][:k], disc)
                if eidcg_c > 0:
                    # For eNDCG, using pre-computed exponential gains
                    exp_gains = (2.0 ** qrels_viable[c, :] - 1.0)
                    m_rank_c = (1/eidcg_c) * pulp.lpSum([
                        exp_gains[d_idx] * disc[i] * x[i][d_idx] 
                        for i in range(k) for d_idx in range(num_viable_docs)
                    ])
                else:
                    m_rank_c = 0
            
            elif metric == 'mrr':
                # MRR: reciprocal rank of first relevant document
                # This requires auxiliary variables to track if we've seen a relevant doc
                # Simplified approximation: sum of (relevance * 1/rank) for first relevant
                m_rank_c = pulp.lpSum([
                    (qrels_viable[c, d_idx] > 0) * (1.0 / (i + 1)) * x[i][d_idx]
                    for i in range(k) for d_idx in range(num_viable_docs)
                ])
            
            elif metric == 'err':
                # ERR requires linearization of exponential stopping probabilities
                # Using approximation with pre-computed gains
                max_rel_c = qrels_t[c].max()
                if max_rel_c == 0:
                    max_rel_c = 4.0
                
                # Pre-compute stopping probabilities for each document
                prob_stop = (2.0 ** qrels_viable[c, :] - 1.0) / (2.0 ** max_rel_c)
                prob_stop = np.clip(prob_stop, 0, 1)
                
                # Simplified linear approximation for ERR
                m_rank_c = pulp.lpSum([
                    prob_stop[d_idx] * (1.0 / (i + 1)) * x[i][d_idx]
                    for i in range(k) for d_idx in range(num_viable_docs)
                ])
            
            elif metric == 'rbp':
                # RBP with persistence parameter p
                p = kwargs.get('p', 0.8)
                max_rel_c = qrels_t[c].max()
                if max_rel_c == 0:
                    max_rel_c = 4.0
                
                # Pre-compute gains
                gains = qrels_viable[c, :] / max_rel_c
                
                m_rank_c = pulp.lpSum([
                    gains[d_idx] * (1 - p) * (p ** i) * x[i][d_idx]
                    for i in range(k) for d_idx in range(num_viable_docs)
                ])
            
            else:  # Default to rel metric
                m_rank_c = (1/k) * pulp.lpSum([
                    qrels_viable[c, d_idx] * x[i][d_idx] 
                    for i in range(k) for d_idx in range(num_viable_docs)
                ])

            # CVaR linearization constraint
            prob += u[c] >= M0_c[c] - m_rank_c - zeta

        # Ranking constraints
        for i in range(k):
            # Each position must have exactly one document
            prob += pulp.lpSum([x[i][d_idx] for d_idx in range(num_viable_docs)]) == 1
        
        for d_idx in range(num_viable_docs):
            # Each document can appear at most once
            prob += pulp.lpSum([x[i][d_idx] for i in range(k)]) <= 1

        # --- Solve the problem ---
        # solver = pulp.GUROBI_CMD(msg=0)  # If Gurobi is available
        prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Using PuLP's default solver

        # --- Parse results ---
        if pulp.LpStatus[prob.status] == 'Optimal':
            current_rank = -np.ones(k, dtype=int)
            for i in range(k):
                for d_idx in range(num_viable_docs):
                    if pulp.value(x[i][d_idx]) == 1:
                        current_rank[i] = doc_indices[d_idx]  # Map back to original document ID
                        break
            final_ranking[t] = current_rank
        else:
            # Fallback if no optimal solution found
            num_viable = len(doc_indices)
            # First, use found relevant documents
            final_ranking[t, :min(num_viable, k)] = doc_indices[:min(num_viable, k)]
            
            # If ranking length k is not filled, pad with other documents
            if num_viable < k:
                # Get all document indices except the ones already used
                other_docs = np.setdiff1d(np.arange(D), doc_indices)
                # Fill remaining slots
                final_ranking[t, num_viable:k] = other_docs[:k - num_viable]

    return final_ranking.astype(np.int32)




def _dcg_vec(rels: np.ndarray, disc: np.ndarray, **kwargs) -> float:
    """Linear DCG. rels length <= len(disc)"""
    return float((rels * disc[:len(rels)]).sum())

def _edcg_vec(rels: np.ndarray, disc: np.ndarray, **kwargs) -> float:
    """Exponential DCG. rels length <= len(disc)"""
    exp_gains = np.power(2.0, rels) - 1.0
    return float((exp_gains * disc[:len(rels)]).sum())


def optimal_efficient(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    metric: str = "rel",
    k: int,
    beta: float = 0.1,
    p: float = 0.8,
    tgt_lvl: Optional[float] = None,
    doc_pool_size: int = 12, # Controls the size of the initial candidate pool
    **kwargs
) -> np.ndarray:
    """
    Finds the truly optimal ranking by brute-force search over a limited
    pool of candidate documents, with optimizations for efficiency.

    Optimizations:
    1.  Filters out documents with zero relevance across all intents for a query.
    2.  Uses a smaller candidate pool (`doc_pool_size`) drawn from the most
        relevant documents to make the search computationally feasible.
    """
    from itertools import combinations, permutations
    from .eval import _cvar_one_topic # Import the evaluation helper

    T, C, D = qrels.shape
    ranking = -np.ones((T, k), dtype=np.int32)

    for t in tqdm(range(T), desc=f"Optimal (k={k}, pool={doc_pool_size})"):
        qrels_t = qrels[t]

        # A document is only viable if it has a non-zero relevance for at least one intent.
        viable_doc_mask = qrels_t.sum(axis=0) > 0
        viable_docs = np.where(viable_doc_mask)[0]

        if len(viable_docs) == 0:
            ranking[t, :k] = np.arange(k) # No relevant docs, just use first k
            continue

        # From the viable docs, take the top `doc_pool_size` based on expected relevance.
        exp_rel_t = (cprob[t, :, None] * qrels_t).sum(axis=0)
        
        # Sort only the viable documents to find the best candidates
        sorted_viable_indices = np.argsort(-exp_rel_t[viable_docs])
        candidate_docs = viable_docs[sorted_viable_indices[:doc_pool_size]]

        best_ranking, min_risk = None, np.inf
        
        num_candidates = len(candidate_docs)
        if num_candidates < k:
            # If the pool is smaller than k, pad and move on
            ranking[t, :num_candidates] = candidate_docs
            if k > num_candidates:
                # Get some other docs to fill the ranking
                other_docs = np.setdiff1d(np.arange(D), candidate_docs)
                ranking[t, num_candidates:] = other_docs[:k - num_candidates]
            continue

        # --- Brute-force Search over Combinations and Permutations ---
        for doc_indices_combo in combinations(candidate_docs, k):
            for p in permutations(doc_indices_combo):
                current_ranking = list(p)
                
                risk = _cvar_one_topic(
                    docs_sys=current_ranking, cprob_t=cprob[t], qrels_t=qrels_t,
                    metric=metric, k=k, beta=beta, tgt_lvl=tgt_lvl, p=p
                )

                if risk < min_risk:
                    min_risk = risk
                    best_ranking = current_ranking
        
        if best_ranking:
            ranking[t, :k] = best_ranking
        else: # Fallback if no valid ranking was found
             ranking[t, :k] = candidate_docs[:k]

    return ranking.astype(np.int32)




def greedy_rank_dcg(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, **kwargs) -> np.ndarray:
    """Optimal for IA-DCG (linear gain) by sorting by expected relevance."""
    T, C, D = qrels.shape
    k = k if k is not None else D
    exp_gain = (cprob[:, :, None] * qrels).sum(axis=1)
    ranking = np.argsort(-exp_gain, axis=1)[:, :k]
    return ranking.astype(np.int32)


def greedy_rank_ndcg(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, **kwargs) -> np.ndarray:
    """Heuristic for IA-NDCG, equivalent to greedy_rank_dcg."""
    return greedy_rank_dcg(cprob, qrels, k=k)


def greedy_rank_endcg(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, **kwargs) -> np.ndarray:
    """Greedy construction for eNDCG with exponential gain."""
    T, C, D = qrels.shape
    k = k if k is not None else D
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    
    # Pre-compute exponential gains
    exp_gains = np.power(2.0, qrels) - 1.0
    
    ranking = -np.ones((T, k), dtype=np.int32)
    
    for t in tqdm(range(T), desc="Greedy eNDCG"):
        # Calculate expected exponential gain for each document
        exp_exp_gain = (cprob[t, :, None] * exp_gains[t]).sum(axis=0)
        
        # Sort by expected exponential gain (greedily optimal for eDCG)
        ranking[t] = np.argsort(-exp_exp_gain)[:k]
    
    return ranking.astype(np.int32)


def greedy_rank_rel(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, **kwargs) -> np.ndarray:
    """Optimal for mean expected relevance."""
    return greedy_rank_dcg(cprob, qrels, k=k) # Same logic as DCG


def greedy_rank_sumrel(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, **kwargs) -> np.ndarray:
    """Optimal for sum of expected relevance."""
    return greedy_rank_dcg(cprob, qrels, k=k) # Same logic as DCG


def greedy_rank_rbp(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, rho: float = 0.8, **kwargs) -> np.ndarray:
    """Greedy construction for RBP with Graded Gain."""
    T, C, D = qrels.shape
    k = k if k is not None else D
    max_rel = np.max(qrels)
    gains = qrels / max_rel if max_rel > 0 else np.zeros_like(qrels)
    remaining, ranking = np.ones((T, D), dtype=bool), -np.ones((T, k), dtype=np.int32)
    rbp_score = np.zeros((T, C), dtype=np.float32)

    for pos in range(k):
        best_doc, best_score = np.full(T, -1, dtype=np.int32), np.full(T, -np.inf, dtype=np.float32)
        rbp_weight = (1 - rho) * (rho ** pos)
        for d in range(D):
            mask = remaining[:, d]
            if not mask.any(): continue
            g_d = gains[:, :, d]
            new_rbp = rbp_score + g_d * rbp_weight
            ia_rbp = (cprob * new_rbp).sum(axis=1)
            better = ia_rbp > best_score
            best_score[better] = ia_rbp[better]
            best_doc[better] = d
        ranking[:, pos] = best_doc
        rows_idx = np.arange(T)
        remaining[rows_idx, best_doc] = False
        g_best = gains[rows_idx, :, best_doc]
        rbp_score += g_best * rbp_weight
    return ranking



def greedy_rank_err(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, **kwargs) -> np.ndarray:
    """Greedy construction for ERR."""
    T, C, D = qrels.shape
    k = k if k is not None else D
    max_rel = np.max(qrels)
    stop_prob = (2.0 ** qrels - 1.0) / (2.0 ** max_rel) if max_rel > 0 else np.zeros_like(qrels)
    remaining, ranking = np.ones((T, D), dtype=bool), -np.ones((T, k), dtype=np.int32)
    reach_prob, err_score = np.ones((T, C), dtype=np.float32), np.zeros((T, C), dtype=np.float32)

    for pos in range(k):
        best_doc, best_score = np.full(T, -1, dtype=np.int32), np.full(T, -np.inf, dtype=np.float32)
        rank_pos = pos + 1
        for d in range(D):
            mask = remaining[:, d]
            if not mask.any(): continue
            p_stop_d = stop_prob[:, :, d]
            err_contrib = reach_prob * p_stop_d / rank_pos
            new_err = err_score + err_contrib
            ia_err = (cprob * new_err).sum(axis=1)
            better = ia_err > best_score
            best_score[better] = ia_err[better]
            best_doc[better] = d
        ranking[:, pos] = best_doc
        rows_idx = np.arange(T)
        remaining[rows_idx, best_doc] = False
        p_stop_best = stop_prob[rows_idx, :, best_doc]
        err_score += reach_prob * p_stop_best / rank_pos
        reach_prob *= (1 - p_stop_best)
    return ranking


def greedy_rank_mrr(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, **kwargs) -> np.ndarray:
    """Greedy construction for MRR."""
    T, C, D = qrels.shape
    k = k if k is not None else D
    rels_bin = (qrels >= 0.5).astype(np.float32)
    remaining, ranking = np.ones((T, D), dtype=bool), -np.ones((T, k), dtype=np.int32)
    found_rel, mrr_score = np.zeros((T, C), dtype=bool), np.zeros((T, C), dtype=np.float32)

    for pos in range(k):
        best_doc, best_score = np.full(T, -1, dtype=np.int32), np.full(T, -np.inf, dtype=np.float32)
        rank_pos = pos + 1
        for d in range(D):
            mask = remaining[:, d]
            if not mask.any(): continue
            r_d = rels_bin[:, :, d]
            mrr_contrib = r_d * (~found_rel) / rank_pos
            new_mrr = mrr_score + mrr_contrib
            ia_mrr = (cprob * new_mrr).sum(axis=1)
            better = ia_mrr > best_score
            best_score[better] = ia_mrr[better]
            best_doc[better] = d
        ranking[:, pos] = best_doc
        rows_idx = np.arange(T)
        remaining[rows_idx, best_doc] = False
        r_best = rels_bin[rows_idx, :, best_doc]
        mrr_score += r_best * (~found_rel) / rank_pos
        found_rel |= (r_best > 0)
    return ranking

def greedy_rank_map(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, filter_docs: bool = False, **kwargs) -> np.ndarray:
    """
    Greedy construction for MAP with binary qrels.
    
    Assumes qrels are already binary (0 or 1).
    """
    T, C, D = qrels.shape
    k = k if k is not None else D
    
    # No normalization needed - qrels are already binary
    rels_bin = qrels.astype(np.int8)
    
    # Filter documents if requested
    if filter_docs:
        # Filter out documents with no relevance across all intents
        non_zero_mask = np.any(qrels > 0, axis=1)  # Shape: (T, D)
        # For documents with all-zero relevance, set rels_bin to -1 as a marker
        for t in range(T):
            for d in range(D):
                if not non_zero_mask[t, d]:
                    rels_bin[t, :, d] = -1  # Mark as invalid
    
    num_rel = np.zeros((T, C), dtype=np.int32)
    sum_prec = np.zeros((T, C), dtype=np.float32)
    remaining = np.ones((T, D), dtype=bool)
    ranking = -np.ones((T, k), dtype=np.int32)
    
    # If filtering, update remaining to exclude zero-relevance docs
    if filter_docs:
        for t in range(T):
            for d in range(D):
                if np.all(rels_bin[t, :, d] == -1):
                    remaining[t, d] = False

    for pos in range(k):
        best_doc = np.full(T, -1, dtype=np.int32)
        best_score = np.full(T, -np.inf, dtype=np.float32)
        rank_pos = pos + 1
        
        for d in range(D):
            mask = remaining[:, d]
            if not mask.any(): 
                continue
            
            # Get relevance for this document (excluding marked invalid docs)
            r_d = np.where(rels_bin[:, :, d] >= 0, rels_bin[:, :, d], 0)
            
            # Calculate new MAP if we add this document
            new_num = num_rel + r_d
            prec_d = r_d * new_num / rank_pos
            new_sum = sum_prec + prec_d
            ap_new = np.where(new_num > 0, new_sum / new_num, 0.0)
            ia_new = (cprob * ap_new).sum(axis=1)
            
            # Update best document for queries where this is better
            better = mask & (ia_new > best_score)
            best_score[better] = ia_new[better]
            best_doc[better] = d
        
        # Update ranking with best documents
        ranking[:, pos] = best_doc
        
        # Update state for selected documents
        rows_idx = np.arange(T)
        valid_selections = best_doc >= 0
        
        if valid_selections.any():
            # Mark selected documents as used
            remaining[rows_idx[valid_selections], best_doc[valid_selections]] = False
            
            # Update num_rel and sum_prec for selected documents
            for t in range(T):
                if best_doc[t] >= 0:
                    r_best = np.where(rels_bin[t, :, best_doc[t]] >= 0, rels_bin[t, :, best_doc[t]], 0)
                    num_rel[t] += r_best
                    sum_prec[t] += r_best * num_rel[t] / rank_pos
    
    # Fill remaining positions if needed
    if filter_docs:
        for t in range(T):
            # Find positions that weren't filled
            unfilled = np.where(ranking[t] == -1)[0]
            if len(unfilled) > 0:
                # Get documents with all-zero relevance
                zero_docs = []
                for d in range(D):
                    if np.all(qrels[t, :, d] == 0):
                        zero_docs.append(d)
                
                # Fill unfilled positions with zero-relevance docs
                for i, pos in enumerate(unfilled):
                    if i < len(zero_docs):
                        ranking[t, pos] = zero_docs[i]
    
    return ranking


def greedy_rank_prec(cprob: np.ndarray, qrels: np.ndarray, *, k: Optional[int] = None, **kwargs) -> np.ndarray:
    """
    Optimal for Precision@k, sorting by expected relevance.
    
    Assumes qrels are already binarized {0, 1}.
    """
    T, C, D = qrels.shape
    k = k if k is not None else D
    
    exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)
    
    ranking = np.argsort(-exp_rel, axis=1)[:, :k]
    return ranking.astype(np.int32)


def greedy_rank(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    metric: str = "rel",
    k: int,
    rho: float = 0.8,
    filter_docs: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Unified greedy ranking function. Dispatches to a metric-specific ranker.
    
    Parameters
    ----------
    filter_docs : bool, default True
        If True, filter out documents with zero relevance across all intents.
        If False, consider all documents. This is handled by the specific
        ranking function if it supports the parameter.
    """
    fn_name = f"greedy_rank_{metric}"
    if fn_name not in globals():
        raise ValueError(f"No greedy ranking function found for metric: {metric}")
    
    rank_fn = globals()[fn_name]
    
    # Prepare arguments for the specific ranking function
    call_kwargs = {'k': k}
    
    # Add arguments required by specific metrics
    if metric == "rbp":
        call_kwargs['rho'] = rho
        
    # Check if the target function accepts other optional parameters
    sig = inspect.signature(rank_fn)
    if 'filter_docs' in sig.parameters:
        call_kwargs['filter_docs'] = filter_docs

    return rank_fn(cprob, qrels, **call_kwargs)


def ia_select(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    filter_docs: bool = False,  # NEW: Control document filtering
    **kwargs,
) -> np.ndarray:
    """
    Implements the IA-Select algorithm. (Fully Vectorized & Robust Version)
    
    Parameters
    ----------
    filter_docs : bool, default True
        If True, filter out documents with zero relevance across all intents.
        If False, consider all documents.
    """
    # The transpose fix for when dimensions are swapped
    if qrels.ndim == 3 and cprob.ndim == 2 and qrels.shape[0] == cprob.shape[0] and qrels.shape[2] == cprob.shape[1]:
        qrels = qrels.transpose(0, 2, 1)
    T, C, D = qrels.shape  # T=queries, C=intents, D=documents
    
    # --- Global Relevance Normalization ---
    # Find min and max across the entire dataset
    global_min = np.min(qrels)
    global_max = np.max(qrels)
    global_range = global_max - global_min
    
    # Normalize using global min/max
    if global_range > 0:
        normalized_qrels = (qrels - global_min) / global_range
    else:
        # If all values are the same, set to 0 (or could set to 1)
        normalized_qrels = np.zeros_like(qrels)
    
    ranking = np.empty((T, k), dtype=np.int32)
    
    for t in range(T):
        if filter_docs:
            # Check which documents have any relevance across any intent
            is_relevant_mask = np.any(qrels[t] > 0, axis=0)  # Shape: (D,)
            valid_docs = np.where(is_relevant_mask)[0]
            n_valid = len(valid_docs)
            if n_valid == 0:
                ranking[t, :] = np.arange(min(k, D))
                continue
        else:
            # Consider all documents
            valid_docs = np.arange(D)
            n_valid = D
        
        # Select intents (all) and filter documents (valid_docs)
        norm_qrels_t_filt = normalized_qrels[t][:, valid_docs]  # Shape: (C, n_valid)
        
        available = np.ones(n_valid, dtype=bool)
        unsat_prob = cprob[t].copy()  # Shape: (C,)
        for pos in range(min(k, n_valid)):
            # Compute scores: unsat_prob (C,) * norm_qrels_t_filt (C, n_valid) -> sum over intents
            scores = (unsat_prob[:, np.newaxis] * norm_qrels_t_filt).sum(axis=0)  # Shape: (n_valid,)
            scores[~available] = -np.inf
            
            best_doc_local_idx = np.argmax(scores)
            
            ranking[t, pos] = valid_docs[best_doc_local_idx]
            available[best_doc_local_idx] = False
            
            # Update unsatisfied probabilities
            d_star_rels = norm_qrels_t_filt[:, best_doc_local_idx]  # Shape: (C,)
            unsat_prob *= (1.0 - d_star_rels)
        # Fill remaining positions with non-relevant documents if needed
        if filter_docs and n_valid < k:
            is_relevant_mask = np.any(qrels[t] > 0, axis=0)
            zero_docs = np.where(~is_relevant_mask)[0]
            ranking[t, n_valid:] = zero_docs[:k - n_valid]
    return ranking


def calibrated(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    lamb: float = 0.5,
    alpha: float = 0.01,
    filter_docs: bool = False,  # NEW: Control document filtering
    **kwargs,
) -> np.ndarray:
    """
    Greedy calibrated re-ranking (RecSys'18).
    
    Parameters
    ----------
    filter_docs : bool, default True
        If True, filter out documents with zero relevance across all intents.
        If False, consider all documents.
    """
    # Robustness Fix: Check for and correct qrels dimension order
    if qrels.ndim == 3 and cprob.ndim == 2 and qrels.shape[0] == cprob.shape[0] and qrels.shape[2] == cprob.shape[1]:
        qrels = qrels.transpose(0, 2, 1)
    
    T, C, D = qrels.shape
    ranking = -np.ones((T, k), dtype=np.int32)
    eps = 1e-12

    exp_rel = (cprob[:, :, np.newaxis] * qrels).sum(axis=1)
    genre_bin = (qrels > 0).astype(float)

    for t in tqdm(range(T), desc="Calibrated (Optimized)"):
        qrels_t = qrels[t]
        
        if filter_docs:
            # Filter out documents non-relevant for all intents
            is_relevant_mask = np.any(qrels_t > 0, axis=0)
            valid_docs = np.where(is_relevant_mask)[0]
            n_valid = len(valid_docs)
            
            if n_valid == 0:
                ranking[t, :] = np.arange(min(k, D))
                continue
        else:
            valid_docs = np.arange(D)
            n_valid = D
        
        # Use only the valid subset of documents for ranking
        p_u = cprob[t]
        exp_rel_t_filtered = exp_rel[t, valid_docs]
        genre_bin_t_filtered = genre_bin[t][:, valid_docs]
        
        available = np.ones(n_valid, dtype=bool)
        genre_sum = np.zeros(C, dtype=float)
        log_p_u = np.log(p_u + eps)

        for pos in range(min(k, n_valid)):
            # Vectorized objective calculation for all available, valid documents
            q_num = genre_sum[:, np.newaxis] + genre_bin_t_filtered
            q = q_num / (pos + 1)
            q_tilde = (1 - alpha) * q + alpha * p_u[:, np.newaxis]

            q_tilde_safe = np.clip(q_tilde, eps, 1.0)
            ckl_all = np.sum(
                p_u[:, np.newaxis] * (log_p_u[:, np.newaxis] - np.log(q_tilde_safe)),
                axis=0
            )

            obj_all = (1 - lamb) * exp_rel_t_filtered - lamb * ckl_all
            obj_all[~available] = -np.inf
            
            # Tie-breaking
            max_obj = np.max(obj_all)
            is_max = np.abs(obj_all - max_obj) < 1e-12
            candidate_ckls = np.where(is_max, ckl_all, np.inf)
            best_doc_local_idx = np.argmin(candidate_ckls)

            # Update state for next iteration
            ranking[t, pos] = valid_docs[best_doc_local_idx]
            available[best_doc_local_idx] = False
            genre_sum += genre_bin_t_filtered[:, best_doc_local_idx]

        # Fill remaining slots if k > number of valid documents
        if filter_docs and n_valid < k:
            is_relevant_mask = np.any(qrels_t > 0, axis=0)
            zero_docs = np.where(~is_relevant_mask)[0]
            ranking[t, n_valid:] = zero_docs[:k - n_valid]

    return ranking.astype(np.int32)


def naive_rank(
    cprob: np.ndarray, 
    qrels: np.ndarray, 
    *, 
    k: int,
    filter_docs: bool = False,  # NEW: Control document filtering
    **kwargs
) -> np.ndarray:
    """
    Ranks by E[rel | topic].
    
    Parameters
    ----------
    filter_docs : bool, default True
        If True, filter out documents with zero relevance across all intents.
        If False, consider all documents.
    """
    # Robustness Fix: Check for and correct qrels dimension order
    if qrels.ndim == 3 and cprob.ndim == 2 and qrels.shape[0] == cprob.shape[0] and qrels.shape[2] == cprob.shape[1]:
        qrels = qrels.transpose(0, 2, 1)
    
    T, C, D = qrels.shape
    exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)
    
    if not filter_docs:
        # Simple case: just sort all documents
        return np.argsort(-exp_rel, axis=1)[:, :k].astype(np.int32)
    
    # With filtering
    ranking = np.empty((T, k), dtype=np.int32)
    
    for t in range(T):
        # Check which documents have any relevance
        is_relevant_mask = np.any(qrels[t] > 0, axis=0)
        valid_docs = np.where(is_relevant_mask)[0]
        n_valid = len(valid_docs)
        
        if n_valid == 0:
            # No relevant documents, fill with first k documents
            ranking[t, :] = np.arange(min(k, D))
            continue
        
        # Sort valid documents by expected relevance
        exp_rel_valid = exp_rel[t, valid_docs]
        sorted_indices = np.argsort(-exp_rel_valid)
        
        # Fill ranking with top-k valid documents
        for pos in range(min(k, n_valid)):
            ranking[t, pos] = valid_docs[sorted_indices[pos]]
        
        # Fill remaining positions with non-relevant documents if needed
        if n_valid < k:
            zero_docs = np.where(~is_relevant_mask)[0]
            ranking[t, n_valid:] = zero_docs[:k - n_valid]
    
    return ranking


def xquad(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    lamb: float = 0.5,
    filter_docs: bool = False,  # NEW: Control document filtering
    **kwargs,
) -> np.ndarray:
    """
    xQuAD (binary coverage) with tunable λ. (Fully Vectorized & Robust Version)
    
    Parameters
    ----------
    filter_docs : bool, default True
        If True, filter out documents with zero relevance across all intents.
        If False, consider all documents.
    """
    # Robustness Fix: Check for and correct qrels dimension order
    if qrels.ndim == 3 and cprob.ndim == 2 and qrels.shape[0] == cprob.shape[0] and qrels.shape[2] == cprob.shape[1]:
        qrels = qrels.transpose(0, 2, 1)

    T, C, D = qrels.shape
    exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)  # Shape: (T, D)
    ranking = np.empty((T, k), dtype=np.int32)
    is_covered = qrels >= 0.5  # Shape: (T, C, D)

    for t in tqdm(range(T), desc="xQuAD (Vectorized)"):
        if filter_docs:
            is_relevant_mask = np.any(qrels[t] > 0, axis=0)  # Check along intent axis
            valid_docs = np.where(is_relevant_mask)[0]
            n_valid = len(valid_docs)
            
            if n_valid == 0:
                ranking[t, :] = np.arange(min(k, D))
                continue
        else:
            valid_docs = np.arange(D)
            n_valid = D
            
        exp_rel_t_filt = exp_rel[t, valid_docs]
        # Select all intents, filter documents
        is_covered_t_filt = is_covered[t][:, valid_docs]  # Shape: (C, n_valid)
        
        available = np.ones(n_valid, dtype=bool)
        cover = np.zeros(C, dtype=bool)

        for pos in range(min(k, n_valid)):
            rel_part = exp_rel_t_filt
            new_cov = (~cover[:, np.newaxis]) & is_covered_t_filt  # Shape: (C, n_valid)
            div_part = (new_cov.astype(float) * cprob[t][:, np.newaxis]).sum(axis=0)
            
            scores = (1 - lamb) * rel_part + lamb * div_part
            scores[~available] = -np.inf
            
            best_doc_local_idx = np.argmax(scores)
            
            original_doc_idx = valid_docs[best_doc_local_idx]
            ranking[t, pos] = original_doc_idx
            available[best_doc_local_idx] = False
            cover |= is_covered[t, :, original_doc_idx]
        
        if filter_docs and n_valid < k:
            is_relevant_mask = np.any(qrels[t] > 0, axis=0)
            zero_docs = np.where(~is_relevant_mask)[0]
            ranking[t, n_valid:] = zero_docs[:k - n_valid]

    return ranking


def fair(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    lamb: float = 0.5,
    metric: str = "rel",
    filter_docs: bool = False,  # NEW: Control document filtering
    **kwargs,
) -> np.ndarray:
    """
    Greedy FA*IR-style re-ranker that equalises exposure across intents.
    (Fully Vectorized Version)
    
    Parameters
    ----------
    filter_docs : bool, default True
        If True, filter out documents with zero relevance across all intents.
        If False, consider all documents.
    """
    # Robustness Fix: Check for and correct qrels dimension order
    if qrels.ndim == 3 and cprob.ndim == 2 and qrels.shape[0] == cprob.shape[0] and qrels.shape[2] == cprob.shape[1]:
        qrels = qrels.transpose(0, 2, 1)

    T, C, D = qrels.shape
    ranking = np.empty((T, k), dtype=np.int32)
    exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)      # (T,D)

    for t in tqdm(range(T), desc="FAIR (Vectorized)"):
        if filter_docs:
            # Filter out non-relevant documents
            is_relevant_mask = np.any(qrels[t] > 0, axis=0)  # Check along intent axis
            valid_docs = np.where(is_relevant_mask)[0]
            n_valid = len(valid_docs)
            
            if n_valid == 0:
                # No relevant documents, fill with first k documents
                ranking[t, :] = np.arange(min(k, D))
                continue
        else:
            valid_docs = np.arange(D)
            n_valid = D
        
        # Filter relevance matrices to valid documents only
        qrels_t_filtered = qrels[t][:, valid_docs]  # Shape: (C, n_valid)
        exp_rel_t_filtered = exp_rel[t, valid_docs]  # Shape: (n_valid,)
        
        # Binary coverage matrix for valid documents
        covers = (qrels_t_filtered > 0).astype(float)  # Shape: (C, n_valid)
        
        # Handle case where all intent probabilities are zero
        if cprob[t].sum() == 0:
            target = np.zeros(C)
        else:
            target = cprob[t] / cprob[t].sum()
        
        delivered = np.zeros(C, dtype=float)
        available = np.ones(n_valid, dtype=bool)

        for pos in range(min(k, n_valid)):
            r = pos + 1
            
            # Vectorized computation for all candidates
            new_delivered_all = delivered[:, np.newaxis] + covers
            
            # Compute quota deviation for all candidates
            quota_all = np.abs(target[:, np.newaxis] * r - new_delivered_all).sum(axis=0)
            
            # Compute scores for all candidates
            scores = (1 - lamb) * exp_rel_t_filtered - lamb * quota_all
            
            # Mask out unavailable documents
            scores[~available] = -np.inf
            
            # Select best document
            best_doc_local_idx = np.argmax(scores)
            
            ranking[t, pos] = valid_docs[best_doc_local_idx]
            available[best_doc_local_idx] = False
            delivered += covers[:, best_doc_local_idx]
        
        # Fill remaining positions with non-relevant documents if needed
        if filter_docs and n_valid < k:
            is_relevant_mask = np.any(qrels[t] > 0, axis=0)
            zero_docs = np.where(~is_relevant_mask)[0]
            ranking[t, n_valid:] = zero_docs[:k - n_valid]

    return ranking.astype(np.int32)


def avg_max(cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    metric: str = "rel",
    k: int,
    beta: float = 0.1,
    lamb: float = 0.5,
    baseline_rnk: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    T, C, D = qrels.shape
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    ranking = -np.ones((T, k), dtype=np.int32)

    for t in tqdm(range(T), desc="ia-greedy"):
        gain_c = np.zeros(C)                # marginal gain per intent
        chosen = []
        remaining = list(range(D))

        for pos in range(k):
            best_doc, best_gain = None, -np.inf
            for d in remaining:
                rels = qrels[t, :, d]       # (C,)
                if metric == "dcg":
                    delta = rels * disc[pos]
                elif metric == "endcg":
                    # Use exponential gain for eNDCG
                    exp_gains = np.power(2.0, rels) - 1.0
                    delta = exp_gains * disc[pos]
                else:                       # "rel" fallback
                    delta = rels
                exp_gain = (cprob[t] * (gain_c + delta)).sum()
                if exp_gain > best_gain:
                    best_gain, best_doc, best_delta = exp_gain, d, delta
            chosen.append(best_doc)
            remaining.remove(best_doc)
            gain_c += best_delta
        ranking[t] = np.array(chosen)
    return ranking



def pm2(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    metric: str = "rel",          # accepted but ignored (for dispatcher)
    k: int,
    beta: float = 0.1,           # accepted but ignored
    lamb:  float = 0.5,           # accepted but ignored
    baseline_rnk: Optional[np.ndarray] = None,  # accepted but ignored
    **kwargs,
) -> np.ndarray:
    """
    PM-2 diversification (Xue et al., SIGIR 2008).

    Parameters
    ----------
    cprob : (T, C)  intent probabilities
    qrels : (T, C, D) graded relevance (binary or graded)
    k     : length of the returned ranking

    Returns
    -------
    ranking : (T, k) int indices into axis-2 of `qrels`
    """
    T, C, D = qrels.shape
    ranking   = np.empty((T, k), dtype=np.int32)

    for t in tqdm(range(T), desc="PM-2"):
        # -------- desired quota per intent -------------------------
        tot = cprob[t].sum()
        quota = cprob[t] / tot if tot > 0 else np.full(C, 1.0 / C)

        satisfied = np.zeros(C, dtype=float)
        remaining = list(range(D))

        for pos in range(k):
            best_doc, best_val = None, -np.inf

            for d in remaining:
                need  = np.maximum(0.0, quota - satisfied)          # (C,)
                gain  = (qrels[t, :, d] >= 0.5).astype(float)          # binary cover
                val   = (need * gain).sum()

                # if all val==0, first doc becomes incumbent
                if val > best_val + 1e-12 or best_doc is None:
                    best_val, best_doc = val, d

            # commit choice
            ranking[t, pos] = best_doc
            remaining.remove(best_doc)
            satisfied += (qrels[t, :, best_doc] >= 0.5).astype(float)

            if not remaining:                    # pad if docs exhausted
                ranking[t, pos + 1 :] = best_doc
                break

    return ranking

def nrbp_greedy(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    rho: float = 0.8, 
    **kwargs,
) -> np.ndarray:
    """
    Greedy optimiser for NRBP (Clarke et al. 2009).

    Parameters
    ----------
    cprob : (T, C)     intent probabilities
    qrels : (T, C, D)  graded relevance
    k     :            depth of the returned ranking
    rho   :            user persistence parameter (0.8–0.9 typical)

    Returns
    -------
    ranking : (T, k)   document indices
    """
    T, C, D = qrels.shape
    depth_prob = (1 - rho) * rho ** np.arange(k)   # (k,)
    ranking = np.empty((T, k), dtype=np.int32)

    # expected relevance (not strictly needed by this greedy version,
    # but convenient if you later switch to a graded-gain variant)
    # exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)  # (T,D)

    for t in tqdm(range(T), desc="NRBP"):
        remaining = list(range(D))
        already_seen = np.zeros(C, dtype=bool)        #  ← FIX: boolean mask

        for pos in range(k):
            best_doc, best_val = None, -np.inf
            for d in remaining:
                # coverage that this doc would newly add
                new_cov = (~already_seen) & (qrels[t, :, d] >= 0.5)
                gain = (new_cov.astype(float) * cprob[t]).sum()
                val = depth_prob[pos] * gain
                if val > best_val:
                    best_val, best_doc = val, d

            ranking[t, pos] = best_doc
            remaining.remove(best_doc)
            already_seen |= qrels[t, :, best_doc] >= 0.5  # works because both are bool

    return ranking

def alpha_ndcg_greedy(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    alpha: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """
    Greedy optimiser for α-nDCG. Works with graded relevance.
    """
    T, C, D = qrels.shape
    disc = 1.0 / np.log2(np.arange(2, k + 2))       # (k,)
    ranking = np.empty((T, k), dtype=np.int32)

    for t in tqdm(range(T), desc="α-nDCG"):
        chosen, remaining = [], list(range(D))
        # how many times each intent has been satisfied so far
        satisfied = np.zeros(C, dtype=int)

        for pos in range(k):
            best_doc, best_gain = None, -np.inf
            for d in remaining:
                gain = 0.0
                for c in range(C):
                    if qrels[t, c, d] == 0:
                        continue
                    # diminishing returns on repeated intents
                    gain_c = qrels[t, c, d] * ((1 - alpha) ** satisfied[c])
                    gain += gain_c
                gain *= disc[pos]                     # DCG discount
                if gain > best_gain:
                    best_gain, best_doc = gain, d
            ranking[t, pos] = best_doc
            remaining.remove(best_doc)
            # update intent counts
            satisfied += (qrels[t, :, best_doc] > 0).astype(int)
    return ranking


def random(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: Optional[int] = None,
    seed: Optional[int] = None,
    **kwargs,          # metric, beta, lamb, baseline_rnk … ignored
) -> np.ndarray:
    """
    Produce a length-k ranking by uniform random shuffling.

    Parameters
    ----------
    cprob : (T, C)   – unused (kept for API consistency
    qrels : (T, C, D)
    k     : cut-off (default = D)
    seed  : optional RNG seed to make the result reproducible

    Returns
    -------
    ranking : (T, k) int32 indices into axis-2 of `qrels`
    """
    T, _, D = qrels.shape
    if k is None:
        k = D

    rng = np.random.default_rng(seed)
    
    # Generate random values for all elements at once
    random_vals = rng.random((T, D))
    
    # Get indices that would sort each row (this gives us the permutation)
    ranking = np.argsort(random_vals, axis=1)[:, :k].astype(np.int32)
    
    return ranking


def qabd(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    n_aspects: int = 5,
    lamb: float = 0.3,
    **kwargs,
) -> np.ndarray:
    """
    Query-Aspect Bottleneck Diversification (simplified).

    Idea  (Yu 23): compress intents into a small set of 'bottleneck'
    aspects, then apply an xQuAD-like selection on those aspects.

    Here we approximate the bottleneck with the top-n_aspects principal
    components of the intent-doc relevance matrix.

    Parameters
    ----------
    n_aspects : size of the bottleneck (default 5)
    lamb      : relevance/coverage trade-off (default 0.3)

    Returns
    -------
    ranking : (T, k)  document indices
    """
    T, C, D = qrels.shape
    ranking = np.empty((T, k), dtype=np.int32)

    # Pre-compute doc-wise expected relevance
    exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)      # (T,D)

    for t in tqdm(range(T), desc="QABD"):
        # ---------- 1) learn bottleneck aspects via PCA -------------
        A = qrels[t].T                                     # (D,C)
        # Fast SVD on the centered matrix
        U, _, _ = np.linalg.svd(A - A.mean(0, keepdims=True), full_matrices=False)
        aspects = U[:, :n_aspects]                         # (D, n_aspects)

        # ---------- 2) greedy xQuAD over aspects --------------------
        chosen, remaining = [], list(range(D))
        cover = np.zeros(n_aspects, dtype=float)

        for pos in range(k):
            best_doc, best_val = None, -np.inf
            for d in remaining:
                rel_part = exp_rel[t, d]
                # coverage gain over bottleneck aspects
                cov_vec = aspects[d] > 0                   # binary mask
                new_cov = (~cover.astype(bool)) & cov_vec
                div_part = new_cov.sum() / n_aspects
                val = (1 - lamb) * rel_part + lamb * div_part
                if val > best_val:
                    best_val, best_doc = val, d
            ranking[t, pos] = best_doc
            remaining.remove(best_doc)
            cover |= aspects[best_doc] > 0

    return ranking.astype(np.int32)



def ma4div(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    alpha: float = 0.5,
    **kwargs,
) -> np.ndarray:
    """
    MA4DIV (simplified): cooperative agents = intents; each 'agent'
    proposes its best doc, then a coordinator selects one in
    round-robin fashion with diminishing alpha.

    Parameters
    ----------
    alpha : decay for agent priority (default 0.5)

    Returns
    -------
    ranking : (T, k)  document indices
    """
    T, C, D = qrels.shape
    ranking = np.empty((T, k), dtype=np.int32)

    for t in tqdm(range(T), desc="MA4DIV"):
        remaining = set(range(D))
        weights = cprob[t].copy()
        weights /= weights.sum() + 1e-12                  # normalise

        for pos in range(k):
            # agent scores for still-available docs
            agent_best = []
            for c in range(C):
                rels = qrels[t, c]
                docs_sorted = np.argsort(-rels)           # high-to-low
                # pick first still-available doc
                for d in docs_sorted:
                    if d in remaining:
                        agent_best.append((c, d, rels[d]))
                        break

            # weighted vote across intents
            vote_scores = {}
            for c, d, rel in agent_best:
                vote_scores[d] = vote_scores.get(d, 0.0) + weights[c] * rel

            if not vote_scores:            # no docs left?
                break

            best_doc = max(vote_scores.items(), key=lambda x: x[1])[0]
            ranking[t, pos] = best_doc
            remaining.remove(best_doc)

            # decay the winning agent’s weight, boost others slightly
            winner = max(agent_best, key=lambda x: x[2])[0]
            weights[winner] *= (1 - alpha)
            weights += alpha / C
            weights /= weights.sum()

        # pad if fewer than k docs chosen
        if len(remaining) > 0 and (pos + 1) < k:
            rest = list(remaining)[: (k - pos - 1)]
            ranking[t, pos + 1 :] = rest

    return ranking.astype(np.int32)




import numpy as np
from tqdm import tqdm
from typing import Optional

def mmr(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    lamb: float = 0.5,
    sim: Optional[np.ndarray] = None,
    **kwargs,
) -> np.ndarray:
    """
    Maximal Marginal Relevance (query intent–aware).
    If sim is None we estimate doc–doc similarity from intent overlap.
      sim[d1,d2] in [0,1] (higher => more redundant)
    """
    T, C, D = qrels.shape
    # expected relevance of every doc
    exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)          # (T,D)

    if sim is None:
        # Simple binary-intent Jaccard as fallback
        bin_cov = (qrels > 0).any(axis=1).astype(float)        # (T,D)
        sim = np.empty((T, D, D))
        for t in tqdm(range(T), desc="MMR (similarity)"):
            inter = bin_cov[t] @ bin_cov[t].T                  # |I∩J|
            union = (bin_cov[t][:, None] + bin_cov[t][None, :]) - inter
            sim[t] = np.divide(inter, union, where=union > 0)
    ranking = np.empty((T, k), dtype=np.int32)

    for t in tqdm(range(T), desc="MMR"):
        chosen = []
        remaining = list(range(D))
        for pos in range(k):
            best_doc, best_val = None, -np.inf
            for d in remaining:
                rel_part = exp_rel[t, d]
                nov_part = 0.0 if not chosen else sim[t, d, chosen].max()
                val = (1 - lamb) * rel_part - lamb * nov_part
                if val > best_val:
                    best_val, best_doc = val, d
            ranking[t, pos] = best_doc
            remaining.remove(best_doc)
            chosen.append(best_doc)
    return ranking

def fair(
    cprob: np.ndarray,
    qrels: np.ndarray,
    *,
    k: int,
    lamb: float = 0.5,       # λ = 0  → pure relevance; 1 → pure exposure fairness
    metric: str = "rel",     # modular metrics only ("rel"|"sumrel")
    **kwargs,
) -> np.ndarray:
    """
    Greedy FA*IR-style re-ranker that equalises exposure across intents.
    (Fully Vectorized Version)
    
    At rank r, the algorithm chooses the doc d that maximises
        (1-λ) * E_rel(d)  -  λ * || quota_remaining_after_d ||₁
    where quota_remaining tracks how much exposure each intent still
    *should* receive to match P(c|q).

    Parameters
    ----------
    cprob : (T,C)   intent distribution P(c|q)
    qrels : (T,C,D) graded relevance
    k     : length of ranking
    lamb  : relevance–fairness trade-off (default 0.5)

    Returns
    -------
    ranking : (T,k)  int indices into axis-2 of qrels
    """
    # Robustness Fix: Check for and correct qrels dimension order
    if qrels.ndim == 3 and cprob.ndim == 2 and qrels.shape[0] == cprob.shape[0] and qrels.shape[2] == cprob.shape[1]:
        qrels = qrels.transpose(0, 2, 1)

    T, C, D = qrels.shape
    ranking = np.empty((T, k), dtype=np.int32)
    exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)      # (T,D)

    for t in tqdm(range(T), desc="FAIR (Vectorized)"):
        # --- Filter out non-relevant documents ---
        is_relevant_mask = np.any(qrels[t] > 0, axis=0)  # Check along intent axis
        valid_docs = np.where(is_relevant_mask)[0]
        n_valid = len(valid_docs)
        
        if n_valid == 0:
            # No relevant documents, fill with first k documents
            ranking[t, :] = np.arange(min(k, D))
            continue
        
        # Filter relevance matrices to valid documents only
        qrels_t_filtered = qrels[t][:, valid_docs]  # Shape: (C, n_valid)
        exp_rel_t_filtered = exp_rel[t, valid_docs]  # Shape: (n_valid,)
        
        # Binary coverage matrix for valid documents
        covers = (qrels_t_filtered > 0).astype(float)  # Shape: (C, n_valid)
        
        # Handle case where all intent probabilities are zero
        if cprob[t].sum() == 0:
            target = np.zeros(C)
        else:
            target = cprob[t] / cprob[t].sum()
        
        delivered = np.zeros(C, dtype=float)
        available = np.ones(n_valid, dtype=bool)

        for pos in range(min(k, n_valid)):
            r = pos + 1
            
            # Vectorized computation for all candidates
            candidate_mask = available
            n_candidates = candidate_mask.sum()
            
            if n_candidates == 0:
                break
            
            # Compute projected delivery for all candidates at once
            # Shape: (C, n_valid) but we only care about available candidates
            new_delivered_all = delivered[:, np.newaxis] + covers
            
            # Compute quota deviation for all candidates
            # Shape: (n_valid,)
            quota_all = np.abs(target[:, np.newaxis] * r - new_delivered_all).sum(axis=0)
            
            # Compute scores for all candidates
            scores = (1 - lamb) * exp_rel_t_filtered - lamb * quota_all
            
            # Mask out unavailable documents
            scores[~available] = -np.inf
            
            # Select best document
            best_doc_local_idx = np.argmax(scores)
            
            ranking[t, pos] = valid_docs[best_doc_local_idx]
            available[best_doc_local_idx] = False
            delivered += covers[:, best_doc_local_idx]
        
        # Fill remaining positions with non-relevant documents if needed
        if n_valid < k:
            zero_docs = np.where(~is_relevant_mask)[0]
            ranking[t, n_valid:] = zero_docs[:k - n_valid]

    return ranking.astype(np.int32)
