"""
AxisSimilarity — versión local.

Basado en ClassAxisSimilarity.py (versión original). Calcula la similitud entre
TODOS los pares de entidades para un eje (conjunto de variables) y la guarda en
parquet por partes, igual que en producción.

Métricas:
  - hybrid : S = 1 - [ α·D_coseno(numéricas) + (1-α)·D_gower(categóricas/booleanas) ]
             α = nº numéricas / nº total de variables del eje
  - volume : S = min(‖a‖, ‖b‖) / max(‖a‖, ‖b‖)   (solo numéricas)

Salida: parquet con columnas ENTITY_A, ENTITY_B, SIMILARITY (entero, escala x100),
solo el triángulo superior (A < B en orden de fila), como en la plataforma original.

Cambios respecto al original:
  - Guarda en disco local (LocalStage) en vez de en un almacenamiento en la nube.
  - Corrige un bug: la ruta de guardado usaba siempre distance_metrics[0], así que
    con ['hybrid', 'volume'] los ficheros de volume acababan en la carpeta de hybrid.
  - Gower categórico calculado columna a columna: el original creaba un array
    (batch × n × p) que multiplica la memoria por el nº de variables categóricas.
  - Tamaño de batch en función de la RAM configurada (antes fijo a 50 GB) y
    limitado por nº máximo de pares por fichero.
  - Coseno con vectores nulos -> distancia neutra (1.0) en vez de NaN
    (el NaN convertido a int8 daba valores basura).
  - Extracción del triángulo superior sin generar la máscara completa.
  - Acumula el histograma de similitudes al vuelo (para el informe, sin releer).
"""
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from .storage import LocalStage

# Bytes aproximados por celda (fila_batch × n) durante el cálculo: matrices float32
# intermedias + comparaciones + extracción del triángulo. Conservador a propósito.
_BYTES_PER_CELL = 40


class AxisSimilarity:
    def __init__(self,
                 name: str,
                 variables: List[str],
                 project_name: str,
                 stage: LocalStage,
                 distance_metrics: List[str] = ("hybrid", "volume"),
                 ram_gb: float = 2.0,
                 max_pairs_per_file: int = 10_000_000,
                 verbose: bool = False):
        self._name = name
        self._variables = list(variables)
        self._project_name = project_name
        self._stage = stage
        self._distance_metrics = list(distance_metrics)
        self._ram_gb = ram_gb
        self._max_pairs_per_file = max_pairs_per_file
        self._verbose = verbose
        self._results: Dict[str, dict] = {}   # resumen por métrica

    def _log(self, *args):
        if self._verbose:
            print(*args)

    # ==================================================================
    # PREPARACIÓN
    # ==================================================================
    def _split_matrices(self, data: pd.DataFrame, variable_types: Dict[str, str]):
        num_cols = [v for v in self._variables if variable_types.get(v) == "numeric"]
        cat_cols = [v for v in self._variables if variable_types.get(v) in ("categorical", "boolean")]
        ignored = [v for v in self._variables if v not in num_cols + cat_cols]
        if ignored:
            self._log(f"  ⚠️  Variables ignoradas (tipo no soportado): {ignored}")

        n = len(data)
        X_num = data[num_cols].to_numpy(dtype=np.float32) if num_cols else np.zeros((n, 0), np.float32)

        # Categóricas: texto -> códigos enteros (comparar enteros es mucho más barato)
        X_cat = np.zeros((n, len(cat_cols)), dtype=np.int32)
        for j, c in enumerate(cat_cols):
            X_cat[:, j], _ = pd.factorize(data[c].astype(str))
        return X_num, X_cat, num_cols, cat_cols

    def _batch_rows(self, n: int) -> int:
        usable = self._ram_gb * 0.8 * 1e9                 # 20 % de margen
        return max(1, int(usable / (_BYTES_PER_CELL * n)))

    def _row_blocks(self, n: int):
        """Bloques de filas [start, end) limitados por RAM y por nº de pares por fichero."""
        max_rows = self._batch_rows(n)
        start = 0
        while start < n - 1:
            end, pairs = start, 0
            while end < n - 1 and (end - start) < max_rows:
                row_pairs = n - end - 1
                if pairs and pairs + row_pairs > self._max_pairs_per_file:
                    break
                pairs += row_pairs
                end += 1
            yield start, end
            start = end

    @staticmethod
    def _upper_triangle(S_block: np.ndarray, start: int, end: int, n: int):
        """Devuelve (idx_a, idx_b, valores) de los pares j > i del bloque."""
        rows = np.arange(start, end)
        lens = n - rows - 1
        idx_a = np.repeat(rows, lens).astype(np.int32)
        idx_b = np.concatenate([np.arange(i + 1, n, dtype=np.int32) for i in rows])
        vals = np.concatenate([S_block[i - start, i + 1:] for i in rows])
        return idx_a, idx_b, vals

    # ==================================================================
    # SIMILITUDES POR BLOQUE
    # ==================================================================
    @staticmethod
    def _cosine_distance_block(X_block, X_all, norms_block, norms_all):
        if X_all.shape[1] == 0:
            return np.zeros((X_block.shape[0], X_all.shape[0]), np.float32)
        denom = np.outer(norms_block, norms_all)
        D = X_block @ X_all.T                                  # producto escalar
        np.divide(D, denom, out=D, where=denom != 0)
        D[denom == 0] = 0.0                                    # sin información -> coseno 0
        return (1.0 - D).astype(np.float32, copy=False)        # distancia en [0, 2]

    @staticmethod
    def _gower_cat_distance_block(C_block, C_all):
        p = C_all.shape[1]
        if p == 0:
            return np.zeros((C_block.shape[0], C_all.shape[0]), np.float32)
        acc = np.zeros((C_block.shape[0], C_all.shape[0]), np.float32)
        for j in range(p):                                     # una columna cada vez
            acc += C_block[:, j, None] != C_all[None, :, j]
        acc /= p
        return acc                                             # proporción de discrepancias [0, 1]

    # ==================================================================
    # CÁLCULO PRINCIPAL
    # ==================================================================
    def calculate(self, data: pd.DataFrame, variable_types: Dict[str, str]) -> Dict[str, dict]:
        """
        data: DataFrame preprocesado con índice = ID de entidad y columnas = variables del eje.
        """
        missing = [v for v in self._variables if v not in data.columns]
        if missing:
            raise ValueError(f"Eje '{self._name}': variables no encontradas en los datos: {missing}")

        n = len(data)
        X_num, X_cat, num_cols, cat_cols = self._split_matrices(data, variable_types)
        ids = data.index.to_numpy()
        if np.issubdtype(ids.dtype, np.integer) and ids.max() < 2**31:
            ids = ids.astype(np.int32)

        p_num, p_cat = len(num_cols), len(cat_cols)
        alpha = p_num / (p_num + p_cat) if (p_num + p_cat) else 0.5
        blocks = list(self._row_blocks(n))

        self._log(f"  Entidades: {n:,} | Pares a calcular: {n * (n - 1) // 2:,}")
        self._log(f"  Numéricas: {p_num} | Categóricas/booleanas: {p_cat} | α (peso numéricas) = {alpha:.3f}")
        self._log(f"  RAM configurada: {self._ram_gb} GB -> {len(blocks)} bloques/ficheros")

        for metric in self._distance_metrics:
            if metric not in ("hybrid", "volume"):
                raise ValueError(f"Métrica no soportada: {metric}")
            if metric == "volume" and p_num == 0:
                self._log("  ⚠️  'volume' necesita variables numéricas; se omite este eje")
                continue

            t0 = time.time()
            self._stage.clear_axis_day(self._project_name, metric, self._name)
            hist = np.zeros(201, dtype=np.int64)               # similitudes de -100 a 100
            total_pairs = 0

            norms = np.linalg.norm(X_num, axis=1).astype(np.float32)

            for k, (start, end) in enumerate(blocks, 1):
                if metric == "hybrid":
                    D_num = self._cosine_distance_block(X_num[start:end], X_num, norms[start:end], norms)
                    D_cat = self._gower_cat_distance_block(X_cat[start:end], X_cat)
                    S = 1.0 - (alpha * D_num + (1 - alpha) * D_cat)
                    del D_num, D_cat
                else:  # volume
                    n_i, n_j = norms[start:end, None], norms[None, :]
                    mx = np.maximum(n_i, n_j)
                    S = np.divide(np.minimum(n_i, n_j), mx,
                                  out=np.ones((end - start, n), np.float32), where=mx != 0)

                S_int = np.clip(np.rint(S * 100), -100, 100).astype(np.int8)
                del S
                idx_a, idx_b, vals = self._upper_triangle(S_int, start, end, n)
                del S_int

                hist += np.bincount(vals.astype(np.int16) + 100, minlength=201)
                total_pairs += len(vals)

                file_name = f"{self._name}_similitud_{metric}_part_{k}.parquet"
                self._stage.put(pd.DataFrame({"ENTITY_A": ids[idx_a],
                                              "ENTITY_B": ids[idx_b],
                                              "SIMILARITY": vals}),
                                self._project_name, metric, self._name, file_name)
                self._log(f"    [{metric}] bloque {k}/{len(blocks)} filas {start:,}-{end:,} "
                          f"-> {len(vals):,} pares ✓")
                del idx_a, idx_b, vals

            elapsed = time.time() - t0
            self._results[metric] = dict(
                eje=self._name, metrica=metric, entidades=n, pares=total_pairs,
                ficheros=len(blocks), segundos=round(elapsed, 1), alpha=round(alpha, 4),
                n_numericas=p_num, n_categoricas=p_cat, histograma=hist,
                carpeta=str(self._stage.axis_dir(self._project_name, metric, self._name)))
            self._log(f"  ✅ {metric}: {total_pairs:,} pares en {elapsed:.1f}s")
        return self._results

    # ==================================================================
    # PROPIEDADES
    # ==================================================================
    @property
    def name(self): return self._name

    @property
    def variables(self): return self._variables

    @property
    def project_name(self): return self._project_name

    @property
    def distance_metrics(self): return self._distance_metrics

    @property
    def results(self): return self._results


# ============================================================================
# MATRIZ COMPLETA PARA MUESTRAS PEQUEÑAS (visualización, análisis)
# ============================================================================
def dense_similarity(data: pd.DataFrame, variable_types: Dict[str, str], metric: str = "hybrid") -> np.ndarray:
    """
    Matriz n×n de similitud (valores en [-1, 1] para hybrid, [0, 1] para volume) con la
    misma fórmula que AxisSimilarity. Pensada para muestras de pocos miles de entidades.
    """
    n = len(data)
    if n > 6000:
        raise ValueError("dense_similarity es para muestras pequeñas (<= 6.000 entidades)")
    axis = AxisSimilarity("tmp", list(data.columns), "tmp", stage=None, distance_metrics=[metric])
    X_num, X_cat, num_cols, cat_cols = axis._split_matrices(data, variable_types)
    norms = np.linalg.norm(X_num, axis=1).astype(np.float32)
    if metric == "hybrid":
        p = len(num_cols) + len(cat_cols)
        alpha = len(num_cols) / p if p else 0.5
        D = (alpha * AxisSimilarity._cosine_distance_block(X_num, X_num, norms, norms)
             + (1 - alpha) * AxisSimilarity._gower_cat_distance_block(X_cat, X_cat))
        return 1.0 - D
    mx = np.maximum(norms[:, None], norms[None, :])
    return np.divide(np.minimum(norms[:, None], norms[None, :]), mx,
                     out=np.ones((n, n), np.float32), where=mx != 0)
