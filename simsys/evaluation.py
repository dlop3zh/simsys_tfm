"""
Evaluación de la Fase 1: ¿las similitudes calculadas tienen sentido?

1. Distribución de similitudes (a partir del histograma acumulado en el cálculo).
2. Validación contra el segmento simulado (ground truth sintético):
     - similitud media intra-segmento vs inter-segmento
     - Precision@k: de los k vecinos más similares de una entidad, % que comparte segmento,
       comparado con lo que se obtendría eligiendo vecinos al azar (baseline).
3. Top-k similares de entidades concretas (ejemplos cualitativos para el informe).

Todo se calcula recorriendo los parquets por partes, sin cargar los ~50M pares a la vez.
"""
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .storage import LocalStage

UMBRALES = (50, 70, 80, 84, 90, 95)


# ============================================================================
# 1. DISTRIBUCIÓN
# ============================================================================
def distribution_summary(hist: np.ndarray) -> pd.DataFrame:
    values = np.arange(-100, 101)
    total = hist.sum()
    mean = (values * hist).sum() / total
    std = np.sqrt(((values - mean) ** 2 * hist).sum() / total)
    cdf = np.cumsum(hist) / total

    def pct(q):
        return int(values[np.searchsorted(cdf, q)])

    rows = [("Pares totales", int(total)), ("Media", round(mean, 2)), ("Desviación típica", round(std, 2)),
            ("Mínimo", int(values[hist > 0].min())), ("Máximo", int(values[hist > 0].max()))]
    rows += [(f"Percentil {int(q * 100)}", pct(q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)]
    for u in UMBRALES:
        n_u = int(hist[values >= u].sum())
        rows.append((f"Pares con similitud >= {u}", n_u))
        rows.append((f"% pares >= {u}", round(100 * n_u / total, 4)))
    return pd.DataFrame(rows, columns=["METRICA", "VALOR"])


def histogram_table(hist: np.ndarray) -> pd.DataFrame:
    values = np.arange(-100, 101)
    df = pd.DataFrame({"SIMILARITY": values, "PARES": hist})
    df = df[df["PARES"] > 0].reset_index(drop=True)
    df["PCT"] = (100 * df["PARES"] / df["PARES"].sum()).round(4)
    df["PCT_ACUMULADO_DESDE_ARRIBA"] = df["PCT"][::-1].cumsum()[::-1].round(4)
    return df


# ============================================================================
# 2. VALIDACIÓN CONTRA SEGMENTOS
# ============================================================================
def validate_against_segments(stage: LocalStage, project: str, metric: str, axis: str,
                              segments: pd.Series, sample_size: int = 300, k: int = 10,
                              extra_ids: Optional[Iterable] = None, seed: int = 42) -> dict:
    """
    segments: Serie con índice = ID de entidad y valor = segmento (ground truth).
    Devuelve dict con tablas: 'global', 'por_segmento', 'matriz_segmentos' y
    'pares_muestra' (para top-k).
    """
    rng = np.random.default_rng(seed)
    seg_sorted = segments.sort_index()
    ids_sorted = seg_sorted.index.to_numpy()
    codes, labels = pd.factorize(seg_sorted.to_numpy())
    n_seg = len(labels)

    # Muestra de entidades "consulta" para Precision@k (+ ejemplos pedidos)
    query_ids = rng.choice(ids_sorted, size=min(sample_size, len(ids_sorted)), replace=False)
    sample_ids = query_ids
    if extra_ids is not None:
        sample_ids = np.union1d(query_ids, np.asarray(list(extra_ids)))

    sum_same = cnt_same = sum_diff = cnt_diff = 0.0
    intra_sum = np.zeros(n_seg); intra_cnt = np.zeros(n_seg)
    inter_sum = np.zeros(n_seg); inter_cnt = np.zeros(n_seg)
    mat_sum = np.zeros(n_seg * n_seg); mat_cnt = np.zeros(n_seg * n_seg)
    collected = []

    for part in stage.iter_parts(project, metric, axis):
        a = part["ENTITY_A"].to_numpy()
        b = part["ENTITY_B"].to_numpy()
        s = part["SIMILARITY"].to_numpy().astype(np.float64)
        ca = codes[np.searchsorted(ids_sorted, a)]
        cb = codes[np.searchsorted(ids_sorted, b)]
        same = ca == cb

        sum_same += s[same].sum(); cnt_same += same.sum()
        sum_diff += s[~same].sum(); cnt_diff += (~same).sum()
        intra_sum += np.bincount(ca[same], weights=s[same], minlength=n_seg)
        intra_cnt += np.bincount(ca[same], minlength=n_seg)
        cell = ca * n_seg + cb
        mat_sum += np.bincount(cell, weights=s, minlength=n_seg * n_seg)
        mat_cnt += np.bincount(cell, minlength=n_seg * n_seg)
        for c in (ca, cb):
            inter_sum += np.bincount(c[~same], weights=s[~same], minlength=n_seg)
            inter_cnt += np.bincount(c[~same], minlength=n_seg)

        # Pares donde participa alguna entidad de la muestra (en ambas direcciones)
        ma, mb = np.isin(a, sample_ids), np.isin(b, sample_ids)
        if ma.any():
            collected.append(pd.DataFrame({"QUERY": a[ma], "VECINO": b[ma], "SIMILARITY": s[ma]}))
        if mb.any():
            collected.append(pd.DataFrame({"QUERY": b[mb], "VECINO": a[mb], "SIMILARITY": s[mb]}))

    pairs = pd.concat(collected, ignore_index=True)
    pairs["SEG_QUERY"] = pairs["QUERY"].map(segments)
    pairs["SEG_VECINO"] = pairs["VECINO"].map(segments)

    # Precision@k (solo sobre la muestra aleatoria, no sobre los ejemplos añadidos)
    top = (pairs[pairs["QUERY"].isin(query_ids)]
           .sort_values(["QUERY", "SIMILARITY"], ascending=[True, False], kind="stable")
           .groupby("QUERY").head(k))
    top["ACIERTO"] = top["SEG_QUERY"] == top["SEG_VECINO"]
    prec_by_query = top.groupby("QUERY").agg(SEGMENTO=("SEG_QUERY", "first"), PRECISION=("ACIERTO", "mean"))

    sizes = segments.value_counts()
    n = len(segments)
    prec_by_query["BASELINE"] = prec_by_query["SEGMENTO"].map((sizes - 1) / (n - 1))

    glob = pd.DataFrame([
        ("Similitud media intra-segmento", round(sum_same / cnt_same, 2)),
        ("Similitud media inter-segmento", round(sum_diff / cnt_diff, 2)),
        ("Diferencia (intra - inter)", round(sum_same / cnt_same - sum_diff / cnt_diff, 2)),
        (f"Precision@{k} (muestra de {len(prec_by_query)})", round(prec_by_query["PRECISION"].mean(), 4)),
        (f"Precision@{k} esperada al azar", round(prec_by_query["BASELINE"].mean(), 4)),
        ("Mejora sobre azar (x veces)",
         round(prec_by_query["PRECISION"].mean() / prec_by_query["BASELINE"].mean(), 2)),
    ], columns=["METRICA", "VALOR"])

    por_seg = pd.DataFrame({
        "SEGMENTO": labels,
        "N_ENTIDADES": [int(sizes[l]) for l in labels],
        "SIM_MEDIA_INTRA": np.round(intra_sum / np.maximum(intra_cnt, 1), 2),
        "SIM_MEDIA_HACIA_OTROS": np.round(inter_sum / np.maximum(inter_cnt, 1), 2),
    })
    por_seg["SEPARACION"] = por_seg["SIM_MEDIA_INTRA"] - por_seg["SIM_MEDIA_HACIA_OTROS"]
    prec_seg = prec_by_query.groupby("SEGMENTO")[["PRECISION", "BASELINE"]].mean().round(4)
    por_seg = por_seg.merge(prec_seg.add_suffix(f"@{k}"), left_on="SEGMENTO", right_index=True, how="left")
    por_seg = por_seg.sort_values("SEPARACION", ascending=False).reset_index(drop=True)

    # Matriz segmento x segmento de similitud media (simétrica)
    ms, mc = mat_sum.reshape(n_seg, n_seg), mat_cnt.reshape(n_seg, n_seg)
    matriz = pd.DataFrame(np.round((ms + ms.T) / np.maximum(mc + mc.T, 1), 2), index=labels, columns=labels)

    return {"global": glob, "por_segmento": por_seg, "matriz_segmentos": matriz, "pares_muestra": pairs}


# ============================================================================
# 3. TOP-K SIMILARES (ejemplos)
# ============================================================================
def top_k_from_pairs(pairs: pd.DataFrame, entity_id, k: int = 5) -> pd.DataFrame:
    sub = pairs[pairs["QUERY"] == entity_id]
    return sub.sort_values("SIMILARITY", ascending=False, kind="stable").head(k)


def top_k_similares(stage: LocalStage, project: str, metric: str, axis: str,
                    entity_id, k: int = 10) -> pd.DataFrame:
    """Consulta directa: los k más similares a una entidad (recorre todas las partes)."""
    found = []
    for part in stage.iter_parts(project, metric, axis):
        m_a = part["ENTITY_A"] == entity_id
        m_b = part["ENTITY_B"] == entity_id
        found.append(pd.DataFrame({"VECINO": part.loc[m_a, "ENTITY_B"], "SIMILARITY": part.loc[m_a, "SIMILARITY"]}))
        found.append(pd.DataFrame({"VECINO": part.loc[m_b, "ENTITY_A"], "SIMILARITY": part.loc[m_b, "SIMILARITY"]}))
    res = pd.concat(found, ignore_index=True)
    return res.sort_values("SIMILARITY", ascending=False, kind="stable").head(k).reset_index(drop=True)
