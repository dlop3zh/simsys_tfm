"""
Métodos de clustering clásicos para comparar con las comunidades del grafo.

  - K-Means      : sobre numéricas escaladas + one-hot de categóricas (el estándar).
  - K-Prototypes : diseñado para datos mixtos (euclídea en numéricas + discrepancias
                   en categóricas), la contraparte más directa de la métrica hybrid.
  - HDBSCAN      : basado en densidad, decide él solo el número de grupos.

K-Means y K-Prototypes necesitan k de antemano. Se les da el número REAL de perfiles
simulados, así que parten con ventaja (es una referencia optimista); las comunidades
del grafo encuentran su número de grupos sin esa información.
"""
import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder

from .communities import external_scores


def mixed_embedding(processed: pd.DataFrame, num: List[str], cat: List[str]) -> np.ndarray:
    """Numéricas (ya escaladas) + one-hot / √2, para que cada categórica pese como una numérica."""
    parts = [processed[num].to_numpy(dtype=float)] if num else []
    if cat:
        oh = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(processed[cat].astype(str))
        parts.append(oh / np.sqrt(2))
    return np.hstack(parts)


def run_baselines(processed: pd.DataFrame, num: List[str], cat: List[str], k: int,
                  truth: Optional[np.ndarray], hdbscan_pca: int = 5, hdbscan_min_cluster: int = 50,
                  kprototypes: bool = True, seed: int = 0, verbose: bool = True) -> Dict[str, dict]:
    results = {}
    Z = mixed_embedding(processed, num, cat)

    def _store(name, labels, dt, **extra):
        n_groups = len(set(labels) - {-1})
        results[name] = {"labels": labels, "metodo": name, "familia": "Clustering clásico",
                         "comunidades": n_groups, "cobertura": round(float((labels >= 0).mean()), 4),
                         "pct_fuera_del_grafo": np.nan, "modularidad": np.nan, "estabilidad": np.nan,
                         **external_scores(truth, labels), "segundos": round(dt, 2), **extra}
        if verbose:
            r = results[name]
            print(f"  {name:<28} grupos={n_groups:>3}  cobertura={r['cobertura']:.1%}  "
                  f"ARI={r['ARI']:.3f}  NMI={r['NMI']:.3f}  ({dt:.1f}s)")

    t0 = time.time()
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Z)
    _store(f"K-Means (k={k})", lab, time.time() - t0)

    if kprototypes and cat:
        from kmodes.kprototypes import KPrototypes
        t0 = time.time()
        X = processed[num + cat].to_numpy(dtype=object)
        cat_idx = list(range(len(num), len(num) + len(cat)))
        lab = KPrototypes(k, init="Cao", n_init=2, random_state=seed, n_jobs=1).fit_predict(X, categorical=cat_idx)
        _store(f"K-Prototypes (k={k})", np.asarray(lab), time.time() - t0)

    t0 = time.time()
    Zp = PCA(hdbscan_pca, random_state=seed).fit_transform(Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lab = HDBSCAN(min_cluster_size=hdbscan_min_cluster, min_samples=10).fit_predict(Zp)
    _store(f"HDBSCAN (PCA {hdbscan_pca})", lab, time.time() - t0)
    return results
