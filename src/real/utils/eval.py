import numpy as np
from typing import Optional



def _map_binary_vect(rels: np.ndarray) -> np.ndarray:
    k = rels.shape[2]
    csum = np.cumsum(rels, axis=2)
    ranks = np.arange(1, k + 1, dtype=np.float32).reshape(1, 1, k)
    prec_i = csum / ranks
    ap_num = (prec_i * rels).sum(axis=2)
    num_rel = rels.sum(axis=2)
    return np.where(num_rel > 0, ap_num / num_rel, 0.0)

def _map_binary_1d(rels: np.ndarray) -> np.ndarray:
    k = rels.shape[1]
    cumsum = np.cumsum(rels, axis=1)
    ranks = np.arange(1, k + 1, dtype=np.float32)
    prec_i = cumsum / ranks
    ap_num = (prec_i * rels).sum(axis=1)
    num_rel = rels.sum(axis=1)
    return np.where(num_rel > 0, ap_num / num_rel, 0.0)

def _map_binary_single(rels: np.ndarray) -> float:
    if len(rels) == 0 or rels.sum() == 0: return 0.0
    csum = np.cumsum(rels)
    prec = csum / np.arange(1, len(rels) + 1)
    return float((prec * rels).sum() / rels.sum())



def _map_exp_1d(exp_rels: np.ndarray) -> np.ndarray:
    k = exp_rels.shape[1]
    cumsum = np.cumsum(exp_rels, axis=1)
    ranks = np.arange(1, k + 1, dtype=np.float32)
    prec_i = cumsum / ranks
    ap_num = (prec_i * exp_rels).sum(axis=1)
    total_rel = exp_rels.sum(axis=1)
    return np.where(total_rel > 0, ap_num / total_rel, 0.0)

def _map_exp_single(exp_rels: np.ndarray) -> float:
    """Mean Average Precision for expected relevance."""
    if len(exp_rels) == 0: return 0.0
    # For expected relevance, we can treat it as binary by thresholding
    # This is a reasonable approximation for MAP with expected relevance
    bin_rel = exp_rels >= 0.5
    return _map_binary_single(bin_rel.astype(float))

def _rel_single(rels: np.ndarray) -> float:
    """Mean relevance for a single ranking."""
    if len(rels) == 0: return 0.0
    return float(rels.mean())

def _ndcg_single(rels: np.ndarray, **kwargs) -> float:
    """nDCG for a single ranking."""
    if len(rels) == 0: return 0.0
    # For single ranking, we need to compute ideal ranking
    ideal_rels = np.sort(rels)[::-1]  # Sort in descending order
    dcg = _dcg_single(rels)
    idcg = _dcg_single(ideal_rels)
    return dcg / idcg if idcg > 0 else 0.0

def _endcg_single(rels: np.ndarray, **kwargs) -> float:
    """eNDCG for a single ranking."""
    if len(rels) == 0: return 0.0
    # For single ranking, we need to compute ideal ranking
    ideal_rels = np.sort(rels)[::-1]  # Sort in descending order
    edcg = _edcg_single(rels)
    eidcg = _edcg_single(ideal_rels)
    return edcg / eidcg if eidcg > 0 else 0.0

def _err_single(rels: np.ndarray, **kwargs) -> float:
    """Expected Reciprocal Rank for a single ranking."""
    if len(rels) == 0: return 0.0
    max_rel = kwargs.get('max_rel', 4.0)
    if max_rel <= 0: max_rel = 4.0
    
    # Calculate stopping probability for each position
    prob_stop = (np.power(2.0, rels) - 1.0) / (2.0 ** max_rel)
    prob_stop = np.clip(prob_stop, 0, 1)
    
    # Calculate probability of continuing to each position
    prob_continue = np.insert(np.cumprod(1 - prob_stop[:-1]), 0, 1.0)
    
    # Calculate ERR
    ranks = np.arange(1, len(rels) + 1)
    err = (prob_continue * prob_stop / ranks).sum()
    return float(err)

def _rbp_single(rels: np.ndarray, **kwargs) -> float:
    """Rank-Biased Precision for a single ranking."""
    if len(rels) == 0: return 0.0
    max_rel = kwargs.get('max_rel', 4.0)
    p = kwargs.get('p', 0.8)
    if max_rel <= 0: max_rel = 4.0
    
    # Calculate gains
    gains = rels / max_rel if max_rel > 0 else np.zeros_like(rels)
    
    # Calculate weights
    weights = (1 - p) * (p ** np.arange(len(rels)))
    
    # Calculate RBP
    rbp = (gains * weights).sum()
    return float(rbp)



def _prec_binary_single(rels: np.ndarray) -> float:
    """Precision for binarized relevance {0, 1}."""
    if len(rels) == 0: return 0.0
    return float(rels.mean())

def _prec_binary_vect(rels):
    """Precision for binary relevance {0, 1}."""
    return rels.mean(axis=2)

def _err_vect(rels, max_rel):
    """Vectorized ERR."""
    prob_stop = (np.power(2.0, rels) - 1.0) / (2.0 ** max_rel)
    prob_stop = np.clip(prob_stop, 0, 1)
    prob_continue = np.insert(np.cumprod(1 - prob_stop, axis=2), 0, 1.0, axis=2)
    ranks = np.arange(1, rels.shape[2] + 1)
    return (prob_continue * prob_stop / ranks).sum(axis=2)

def _rel_vect(rels, **kwargs):
    """Vectorized mean relevance."""
    return rels.mean(axis=2)

def _dcg_vect(rels):
    k = rels.shape[2]; disc = 1.0 / np.log2(np.arange(2, k + 2)); return (rels * disc).sum(axis=2)

def _edcg_vect(rels):
    k = rels.shape[2]; disc = 1.0 / np.log2(np.arange(2, k + 2)); exp_gains = np.power(2.0, rels) - 1.0; return (exp_gains * disc).sum(axis=2)

def _dcg_1d(rels):
    k = rels.shape[1]; discount = 1.0 / np.log2(np.arange(2, k + 2)); return (rels * discount).sum(axis=1)

def _edcg_1d(rels):
    """Exponential DCG for 1D array."""
    return (np.power(2.0, rels) - 1.0).sum()

def _rel_1d(rels, **kwargs):
    """Mean relevance for 1D array."""
    return float(rels.mean())

def _dcg_single(rels):
    if len(rels) == 0: return 0.0
    k = len(rels); disc = 1.0 / np.log2(np.arange(2, k + 2)); return float((rels * disc).sum())

def _edcg_single(rels):
    if len(rels) == 0: return 0.0
    k = len(rels); disc = 1.0 / np.log2(np.arange(2, k + 2)); exp_gains = np.power(2.0, rels) - 1.0; return float((exp_gains * disc).sum())

def _cvar_one(losses: np.ndarray, weights: np.ndarray, beta: float) -> float:
    order = np.argsort(-losses); losses, weights = losses[order], weights[order]
    tail_weight, tail_sum = 0.0, 0.0
    for l, w in zip(losses, weights):
        if tail_weight + w <= beta + 1e-12:
            tail_sum += l * w; tail_weight += w
        else:
            tail_sum += l * (beta - tail_weight); break
    return tail_sum / beta if beta > 0 else 0.0



def ia_m(rnk: np.ndarray,
         cprob: np.ndarray, 
         qrels: np.ndarray,
         *,
         metric: str = "map",
         k: int | None = None,
         p: float = 0.8) -> float:
    """
    IA-M evaluation.
    For MAP and Precision, assumes qrels are already binary.
    """
    T, K_full = rnk.shape
    k = K_full if k is None else k
    rnk_k = rnk[:, :k]

    idx = np.broadcast_to(rnk_k[:, None, :], (T, qrels.shape[1], k))
    rels = np.take_along_axis(qrels, idx, axis=2)

    kwargs = {}
    if metric == "map":
        # For IA-M with binary qrels, use binary MAP
        metric_fn = _map_binary_vect
    elif metric == "prec":
        # For binary qrels, use mean directly. This is correct as is.
        metric_fn = _prec_binary_vect
    elif metric == "err":
        max_rels = qrels.max(axis=2)
        max_rels[max_rels == 0] = 4.0
        kwargs['max_rel'] = max_rels
        metric_fn = _err_vect
    elif metric == "ndcg":
        ideal_rels = -np.partition(-qrels, k, axis=2)[:, :, :k]
        idcgs = _dcg_vect(ideal_rels)
        kwargs['idcgs'] = idcgs
        metric_fn = _ndcg_vect
    elif metric == "endcg":
        ideal_rels = -np.partition(-qrels, k, axis=2)[:, :, :k]
        eidcgs = _edcg_vect(ideal_rels)
        kwargs['eidcgs'] = eidcgs
        metric_fn = _endcg_vect
    elif metric == "rbp":
        max_rels = qrels.max(axis=2)
        max_rels[max_rels == 0] = 4.0
        kwargs['max_rel'] = max_rels
        kwargs['p'] = p
        metric_fn = _rbp_vect
    else:
        metric_fn = globals()[f"_{metric}_vect"]

    m_ic = metric_fn(rels, **kwargs)
    ia_topic = (cprob * m_ic).sum(axis=1)
    return ia_topic.mean()



def m_raw(rnk: np.ndarray,
          cprob: np.ndarray,
          qrels: np.ndarray,
          *,
          metric: str,
          k: int | None = None,
          p: float = 0.8) -> float:
    """
    M-raw evaluation using expected relevance.
    For MAP, uses continuous expected relevance values directly.
    For Precision, calculates Mean Expected Relevance @k.
    """
    T, K_full = rnk.shape
    k = K_full if k is None else k
    rnk_k = rnk[:, :k]

    # Calculate expected relevance across intents
    exp_rel = (cprob[:, :, None] * qrels).sum(axis=1)
    rels = np.take_along_axis(exp_rel, rnk_k, axis=1)

    kwargs = {}
    if metric == "map":
        # For M-raw MAP: use expected relevance directly (continuous MAP)
        metric_fn = _map_exp_1d
    elif metric == "prec":
        # Calculate Mean Expected Relevance @k by taking the mean of the
        # continuous expected relevance scores, without binarizing.
        m_topic = rels.mean(axis=1)
        return m_topic.mean()
    elif metric == "err" or metric == "rbp":
        max_rel = exp_rel.max()
        kwargs['max_rel'] = max_rel if max_rel > 0 else 4.0
        if metric == "rbp":
            kwargs['p'] = p
        metric_fn = globals()[f"_{metric}_1d"]
    elif metric == "ndcg":
        ideal_rels = -np.partition(-exp_rel, k, axis=1)[:, :k]
        idcgs = _dcg_1d(ideal_rels)
        kwargs['idcgs'] = idcgs
        metric_fn = _ndcg_1d
    elif metric == "endcg":
        ideal_rels = -np.partition(-exp_rel, k, axis=1)[:, :k]
        eidcgs = _edcg_1d(ideal_rels)
        kwargs['idcgs'] = eidcgs
        metric_fn = _endcg_1d
    else:
        metric_fn = globals()[f"_{metric}_1d"]
        
    m_topic = metric_fn(rels, **kwargs)
    return m_topic.mean()



def cvar(rnk: np.ndarray,
         cprob: np.ndarray,
         qrels: np.ndarray,
         *,
         metric: str,
         k: int | None = None,
         p: float = 0.8,
         beta: float = 0.05,
         tgt_lvl: float | None = None,
         baseline_rnk: np.ndarray | None = None) -> float:
    """
    CVaR evaluation.
    For MAP and Precision, assumes qrels are already binary.
    """
    T, K_full = rnk.shape
    k = K_full if k is None else k
    risks = np.empty(T, dtype=np.float32)

    for t in range(T):
        qrels_t = qrels[t]
        C, D = qrels_t.shape
        M_c, M_o = np.empty(C, dtype=np.float32), np.empty(C, dtype=np.float32)

        for c in range(C):
            docs_sys = rnk[t, :k]
            rels_sys_c = qrels_t[c, docs_sys]
            
            if baseline_rnk is not None:
                docs_base = baseline_rnk[t, :k]
                rels_base_c = qrels_t[c, docs_base]
            else:  # Oracle
                top_k_indices = np.argpartition(-qrels_t[c], k-1)[:k]
                rels_base_c = qrels_t[c, top_k_indices]

            if metric == 'map':
                # For CVaR with binary qrels
                M_c[c] = _map_binary_single(rels_sys_c)
                M_o[c] = _map_binary_single(rels_base_c)
            elif metric == 'prec':
                # For CVaR with binary qrels, use binary precision helper
                M_c[c] = _prec_binary_single(rels_sys_c)
                M_o[c] = _prec_binary_single(rels_base_c)
            elif metric == 'ndcg':
                idcg_c = _dcg_single(rels_base_c)
                if idcg_c > 0:
                    M_c[c] = _dcg_single(rels_sys_c) / idcg_c
                    M_o[c] = _dcg_single(rels_base_c) / idcg_c if baseline_rnk is not None else 1.0
                else: 
                    M_c[c], M_o[c] = 0.0, 0.0
            elif metric == 'endcg':
                eidcg_c = _edcg_single(rels_base_c)
                if eidcg_c > 0:
                    M_c[c] = _edcg_single(rels_sys_c) / eidcg_c
                    M_o[c] = _edcg_single(rels_base_c) / eidcg_c if baseline_rnk is not None else 1.0
                else:
                    M_c[c], M_o[c] = 0.0, 0.0
            else:
                kwargs = {}
                if metric == 'err' or metric == 'rbp':
                    max_rel_c = qrels_t[c].max()
                    kwargs['max_rel'] = max_rel_c if max_rel_c > 0 else 4.0
                if metric == 'rbp':
                    kwargs['p'] = p
                
                metric_fn = globals()[f"_{metric}_single"]
                M_c[c] = metric_fn(rels_sys_c, **kwargs)
                M_o[c] = metric_fn(rels_base_c, **kwargs)

        if tgt_lvl is not None and baseline_rnk is None:
            M_o = tgt_lvl * M_o

        losses = np.maximum(0, M_o - M_c)
        total = cprob[t].sum()
        weights = (cprob[t] / total if total > 0 else np.full_like(cprob[t], 1.0 / C))
        risks[t] = _cvar_one(losses, weights, beta)

    return float(risks.mean())




def ia_m_vec(rnk, cprob, qrels, *, metric, k, p=0.8):
    """Per-query IA-M vector."""
    T, K_full = rnk.shape
    k = K_full if k is None else k; rnk_k = rnk[:, :k]
    idx = np.broadcast_to(rnk_k[:, None, :], (T, qrels.shape[1], k))
    rels = np.take_along_axis(qrels, idx, axis=2)
    kwargs = {}
    if metric == "map" or metric == "prec":
        metric_fn = _map_binary_vect if metric == "map" else _prec_binary_vect
    elif metric == "err":
        max_rels = qrels.max(axis=2); max_rels[max_rels == 0] = 4.0; kwargs['max_rel'] = max_rels; metric_fn = _err_vect
    elif metric == "ndcg":
        ideal_rels = -np.partition(-qrels, k, axis=2)[:, :, :k]; idcgs = _dcg_vect(ideal_rels); kwargs['idcgs'] = idcgs
        metric_fn = lambda r, idcgs: np.where(idcgs > 0, _dcg_vect(r) / idcgs, 0.0)
    elif metric == "endcg":
        ideal_rels = -np.partition(-qrels, k, axis=2)[:, :, :k]; eidcgs = _edcg_vect(ideal_rels); kwargs['eidcgs'] = eidcgs
        metric_fn = lambda r, eidcgs: np.where(eidcgs > 0, _edcg_vect(r) / eidcgs, 0.0)
    else:
        # Fallback for other metrics like RBP, etc.
        metric_fn = globals()[f"_{metric}_vect"]
    m_ic = metric_fn(rels, **kwargs)
    return (cprob * m_ic).sum(1).astype(np.float32)

def m_raw_vec(rnk, cprob, qrels, *, metric, k, p=0.8):
    """
    Per-query raw-metric vector.
    """
    T, K_full = rnk.shape
    k = K_full if k is None else k; rnk_k = rnk[:, :k]
    exp_rel = (cprob[:, :, None] * qrels).sum(1)
    rels = np.take_along_axis(exp_rel, rnk_k, axis=1)
    kwargs = {}

    if metric == "map":
        metric_fn = _map_exp_1d
    elif metric == "prec":
        # This is the new logic: calculate mean expected relevance directly.
        # This makes it arithmetically identical to ia_m for 'prec'.
        return rels.mean(axis=1).astype(np.float32)
    elif metric == "ndcg":
        ideal_rels = -np.partition(-exp_rel, k, axis=1)[:, :k]; idcgs = _dcg_1d(ideal_rels); kwargs['idcgs'] = idcgs
        metric_fn = lambda r, idcgs: np.where(idcgs > 0, _dcg_1d(r) / idcgs, 0.0)
    elif metric == "endcg":
        ideal_rels = -np.partition(-exp_rel, k, axis=1)[:, :k]; eidcgs = _edcg_1d(ideal_rels); kwargs['eidcgs'] = eidcgs
        metric_fn = lambda r, eidcgs: np.where(eidcgs > 0, _edcg_1d(r) / eidcgs, 0.0)
    else:
        # Fallback for other metrics
        metric_fn = globals()[f"_{metric}_1d"]
        if metric in ["err", "rbp"]:
            max_rel = exp_rel.max(); kwargs['max_rel'] = max_rel if max_rel > 0 else 4.0
            if metric == "rbp": kwargs['p'] = p
            
    result = metric_fn(rels, **kwargs)
    if isinstance(result, (int, float)):
        result = np.array([result] * T)
    return result.astype(np.float32)

def _cvar_one_topic(docs_sys, cprob_t, qrels_t, metric, k, beta, tgt_lvl, p=0.8):
    """
    Helper for cvar_vec. Includes specific logic for metric='prec' and 'map'.
    """
    C = qrels_t.shape[0]; M_c = np.empty(C); M_o = np.empty(C)
    for c in range(C):
        rels_sys_c = qrels_t[c, docs_sys]
        effective_k = min(k, qrels_t.shape[1] - 1)
        top_k_indices = np.argpartition(-qrels_t[c], effective_k)[:k] if effective_k >= 0 else np.array([])
        rels_oracle_c = qrels_t[c, top_k_indices]
        
        # This block now handles prec and map correctly for binary qrels
        if metric == 'map':
            M_c[c] = _map_binary_single(rels_sys_c); M_o[c] = _map_binary_single(rels_oracle_c)
        elif metric == 'prec':
            M_c[c] = _prec_binary_single(rels_sys_c); M_o[c] = _prec_binary_single(rels_oracle_c)
        elif metric == 'ndcg':
            idcg_c = _dcg_single(rels_oracle_c)
            M_c[c] = _dcg_single(rels_sys_c) / idcg_c if idcg_c > 0 else 0.0; M_o[c] = 1.0 if idcg_c > 0 else 0.0
        elif metric == 'endcg':
            eidcg_c = _edcg_single(rels_oracle_c)
            M_c[c] = _edcg_single(rels_sys_c) / eidcg_c if eidcg_c > 0 else 0.0; M_o[c] = 1.0 if eidcg_c > 0 else 0.0
        else:
            kwargs = {}; metric_fn = globals()[f"_{metric}_single"]
            if metric in ['err', 'rbp']:
                max_rel_c = qrels_t[c].max(); kwargs['max_rel'] = max_rel_c if max_rel_c > 0 else 4.0
                if metric == 'rbp': kwargs['p'] = p
            M_c[c] = metric_fn(rels_sys_c, **kwargs); M_o[c] = metric_fn(rels_oracle_c, **kwargs)

    if tgt_lvl is not None: M_o *= tgt_lvl
    losses = np.maximum(0, M_o - M_c)
    weights = (cprob_t / cprob_t.sum()) if cprob_t.sum() > 0 else np.full_like(cprob_t, 1.0 / C)
    return _cvar_one(losses, weights, beta)

def cvar_vec(rnk, cprob, qrels, *, metric, k, beta, tgt_lvl, p=0.8):
    """Per-query CVaR vector (length = #queries)."""
    T = rnk.shape[0]
    return np.array([
        _cvar_one_topic(rnk[t, :k], cprob[t], qrels[t], metric, k, beta, tgt_lvl, p)
        for t in range(T)
    ], dtype=np.float32)
