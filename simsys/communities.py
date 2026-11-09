"""
Análisis de comunidades — Fase 2.

  - Métricas de un grafo y de su partición (cobertura, modularidad, estabilidad…).
  - Barrido de umbral y de k para elegir cómo construir el grafo.
  - Selección de la configuración con un criterio NO supervisado.
  - Réplica de la configuración original (umbral 84 + Louvain) como referencia.
  - Perfilado de comunidades para interpretarlas en términos de negocio.

El segmento simulado (ground truth) solo se usa como VALIDACIÓN EXTERNA (ARI, NMI):
nunca para elegir parámetros, porque con datos reales no existiría.
"""
import itertools
import time
from typing import Dict, List, Optional

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .graph_manager import (build_graph, detect_communities, knn_graph, modularity,
                            pairs_to_positions, threshold_graph)

RESTO = "Resto"


# ============================================================================
# MÉTRICAS
# ============================================================================
def external_scores(truth: Optional[np.ndarray], labels: np.ndarray) -> Dict[str, float]:
    if truth is None:
        return {"ARI": np.nan, "NMI": np.nan}
    return {"ARI": round(adjusted_rand_score(truth, labels), 4),
            "NMI": round(normalized_mutual_info_score(truth, labels), 4)}


def partition_summary(g: ig.Graph, membership: np.ndarray, min_size: int) -> Dict[str, float]:
    sizes = np.bincount(membership)
    covered = sizes[membership] >= min_size
    deg = np.asarray(g.degree())
    comps = g.connected_components()
    return {
        "aristas": g.ecount(),
        "grado_medio": round(2 * g.ecount() / g.vcount(), 2),
        "pct_aislados": round(float((deg == 0).mean()), 4),
        "pct_componente_principal": round(max(comps.sizes()) / g.vcount(), 4),
        "comunidades": int((sizes >= min_size).sum()),
        "cobertura": round(float(covered.mean()), 4),
        "pct_mayor_comunidad": round(sizes.max() / g.vcount(), 4),
        "modularidad": round(modularity(g, membership), 4),
    }


def stability(g: ig.Graph, method: str, seeds: List[int], resolution: float = 1.0):
    """ARI medio entre las particiones obtenidas con distintas semillas (1 = siempre igual)."""
    parts = [detect_communities(g, method, s, resolution) for s in seeds]
    if len(parts) < 2:
        return 1.0, parts
    aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(parts, 2)]
    return round(float(np.mean(aris)), 4), parts


def evaluate_graph(g: ig.Graph, tipo: str, parametro, truth: Optional[np.ndarray], seeds: List[int],
                   min_size: int, method: str = "leiden", resolution: float = 1.0) -> Dict:
    t0 = time.time()
    stab, parts = stability(g, method, seeds, resolution)
    membership = parts[0]
    row = {"grafo": tipo, "parametro": parametro, **partition_summary(g, membership, min_size),
           "estabilidad": stab, **external_scores(truth, membership),
           "segundos": round((time.time() - t0) / max(len(seeds), 1), 2)}
    return row


# ============================================================================
# BARRIDOS
# ============================================================================
def threshold_sweep(pairs: pd.DataFrame, ids_sorted: np.ndarray, thresholds: List[int],
                    truth: Optional[np.ndarray], seeds: List[int], min_size: int,
                    resolution: float = 1.0, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for u in thresholds:
        g = threshold_graph(pairs, ids_sorted, u)
        rows.append(evaluate_graph(g, "umbral", u, truth, seeds, min_size, resolution=resolution))
        if verbose:
            r = rows[-1]
            print(f"  umbral {u:3d}: aristas={r['aristas']:>9,}  aislados={r['pct_aislados']:.1%}  "
                  f"comunidades={r['comunidades']:>3}  cobertura={r['cobertura']:.1%}  "
                  f"Q={r['modularidad']:.3f}  estab={r['estabilidad']:.3f}")
    return pd.DataFrame(rows)


def knn_sweep(knn_sim: np.ndarray, knn_idx: np.ndarray, ids_sorted: np.ndarray, ks: List[int],
              truth: Optional[np.ndarray], seeds: List[int], min_size: int,
              resolution: float = 1.0, verbose: bool = True) -> pd.DataFrame:
    rows = []
    for k in ks:
        g = knn_graph(ids_sorted, knn_sim, knn_idx, k)
        rows.append(evaluate_graph(g, "knn", k, truth, seeds, min_size, resolution=resolution))
        if verbose:
            r = rows[-1]
            print(f"  k = {k:3d}:     aristas={r['aristas']:>9,}  aislados={r['pct_aislados']:.1%}  "
                  f"comunidades={r['comunidades']:>3}  cobertura={r['cobertura']:.1%}  "
                  f"Q={r['modularidad']:.3f}  estab={r['estabilidad']:.3f}")
    return pd.DataFrame(rows)


def select_configuration(sweep: pd.DataFrame, min_coverage: float = 0.95,
                         min_stability: float = 0.90) -> pd.Series:
    """
    Criterio NO supervisado (no usa el segmento simulado):
      1. cobertura >= min_coverage: casi todas las entidades caen en una comunidad útil;
      2. estabilidad >= min_stability: la partición no depende de la semilla;
      3. entre las que cumplen, máxima modularidad (desempate: grafo más ligero).
    Por qué no solo estabilidad: crece al densificar el grafo porque las comunidades se
    fusionan, y acabaría eligiendo siempre el umbral más bajo del barrido.
    """
    cand = sweep[(sweep["cobertura"] >= min_coverage) & (sweep["estabilidad"] >= min_stability)]
    if cand.empty:
        cand = sweep[sweep["cobertura"] >= min_coverage]
    if cand.empty:
        cand = sweep
    cand = cand.sort_values(["modularidad", "aristas"], ascending=[False, True])
    return cand.iloc[0]


# ============================================================================
# CONFIGURACIÓN ORIGINAL (referencia)
# ============================================================================
def production_reference(pairs: pd.DataFrame, ids_sorted: np.ndarray, truth: Optional[np.ndarray],
                         threshold: int = 84, min_size: int = 20, seed: int = 0) -> Dict:
    """
    Reproduce la versión en producción: umbral 84, Louvain y SOLO los nodos que aparecen
    en algún par. Las entidades que quedan fuera se cuentan como no cubiertas y, para
    ARI/NMI, cada una como su propio grupo.
    """
    sub = pairs[pairs["SIMILARITY"] >= threshold]
    present = np.unique(np.concatenate([sub["ENTITY_A"].to_numpy(), sub["ENTITY_B"].to_numpy()]))
    a, b, s = pairs_to_positions(sub, present)
    g = build_graph(present, a, b, s)
    t0 = time.time()
    memb_present = detect_communities(g, "louvain", seed)
    dt = time.time() - t0

    labels = np.arange(len(ids_sorted)) + 10**6              # fuera del grafo: grupo propio
    pos = np.searchsorted(ids_sorted, present)
    labels[pos] = memb_present
    sizes = np.bincount(memb_present)
    covered = np.zeros(len(ids_sorted), bool)
    covered[pos] = sizes[memb_present] >= min_size
    return {"metodo": f"Original: umbral {threshold} + Louvain", "familia": "Grafo",
            "comunidades": int((sizes >= min_size).sum()),
            "cobertura": round(float(covered.mean()), 4),
            "pct_fuera_del_grafo": round(1 - len(present) / len(ids_sorted), 4),
            "modularidad": round(modularity(g, memb_present), 4),
            **external_scores(truth, labels), "segundos": round(dt, 2)}


def compare_graph_methods(g: ig.Graph, truth: Optional[np.ndarray], seeds: List[int], min_size: int,
                          resolution: float = 1.0) -> pd.DataFrame:
    rows = []
    for method in ("leiden", "louvain"):
        t0 = time.time()
        stab, parts = stability(g, method, seeds, resolution)
        dt = (time.time() - t0) / len(seeds)
        m = parts[0]
        s = partition_summary(g, m, min_size)
        rows.append({"metodo": method.capitalize(), "familia": "Grafo", "comunidades": s["comunidades"],
                     "cobertura": s["cobertura"], "pct_fuera_del_grafo": 0.0,
                     "modularidad": s["modularidad"], "estabilidad": stab,
                     **external_scores(truth, m), "segundos": round(dt, 2)})
    return pd.DataFrame(rows)


# ============================================================================
# ETIQUETAS Y PERFILES
# ============================================================================
def relabel_by_size(membership: np.ndarray, min_size: int) -> np.ndarray:
    """C01 = comunidad más grande… Las de menos de min_size entidades pasan a 'Resto'."""
    sizes = pd.Series(membership).value_counts()
    big = [c for c in sizes.index if sizes[c] >= min_size]
    names = {c: f"C{i + 1:02d}" for i, c in enumerate(big)}
    return np.array([names.get(c, RESTO) for c in membership], dtype=object)


def community_segment_crosstab(labels: pd.Series, segments: pd.Series) -> pd.DataFrame:
    """% de cada segmento simulado dentro de cada comunidad (filas suman 100)."""
    ct = pd.crosstab(labels, segments, normalize="index") * 100
    return ct.round(1)


def profile_communities(data: pd.DataFrame, variable_types: Dict[str, str], labels: pd.Series,
                        segments: Optional[pd.Series] = None, key_columns: Optional[List[str]] = None,
                        top_n: int = 4) -> pd.DataFrame:
    """
    data: datos imputados SIN escalar (processed_data_before_scaling), índice = ID.
    Rasgos característicos de cada comunidad frente al total:
      - numéricas: percentil medio de la comunidad (↑ si > 50, ↓ si < 50);
      - categóricas/booleanas: categoría sobre-representada (% en la comunidad vs % global).
    """
    labels = labels.reindex(data.index)
    num = [c for c, t in variable_types.items() if t == "numeric" and c in data]
    cat = [c for c, t in variable_types.items() if t in ("categorical", "boolean") and c in data]
    ranks = data[num].rank(pct=True)
    global_share = {c: data[c].astype(str).value_counts(normalize=True) for c in cat}

    rows = []
    order = [c for c in labels.value_counts().index if c != RESTO] + ([RESTO] if (labels == RESTO).any() else [])
    for comm in order:
        mask = (labels == comm).to_numpy()
        n = int(mask.sum())
        traits = []
        if comm != RESTO:
            mr = ranks[mask].mean()
            for c in num:
                traits.append((abs(mr[c] - 0.5) * 2, f"{'↑' if mr[c] > 0.5 else '↓'} {c} (p{round(mr[c] * 100)})"))
            for c in cat:
                share_c = data.loc[mask, c].astype(str).value_counts(normalize=True)
                for val, sc in share_c.items():
                    if val in ("__NA__", "nan"):
                        continue
                    sg = global_share[c].get(val, 0)
                    if sc >= 0.4 and sg > 0 and sc / sg >= 1.5:
                        traits.append((sc - sg, f"{c} = {val} ({sc:.0%} vs {sg:.0%})"))
            traits.sort(key=lambda t: -t[0])
        row = {"COMUNIDAD": comm, "N": n, "PCT": round(100 * n / len(labels), 1),
               "RASGOS": " · ".join(t[1] for t in traits[:top_n])}
        if segments is not None:
            seg = segments.reindex(data.index)[mask].value_counts(normalize=True)
            row["SEGMENTO_DOMINANTE"] = seg.index[0]
            row["PUREZA"] = round(float(seg.iloc[0]), 3)
        for c in (key_columns or []):
            if c in num:
                row[f"MEDIANA_{c}"] = round(float(data.loc[mask, c].median()), 2)
        rows.append(row)
    return pd.DataFrame(rows)
