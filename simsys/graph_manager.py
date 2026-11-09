"""
GraphManager — versión local del gestor de grafos del sistema original.

Construye el grafo de similitud a partir de los pares guardados en la fase 1, detecta
comunidades y guarda / carga el resultado en disco (caché versionada), igual que el
original hacía con el almacenamiento en la nube.

Dos formas de construir el grafo:
  - UMBRAL : arista entre A y B si similitud >= umbral (lo que hace producción, umbral 84).
  - KNN    : cada entidad se une a sus k entidades más similares (grafo de k vecinos).

Cambios respecto al original:
  - Todas las entidades son vértices, también las que no tienen ninguna arista.
    En el original solo existían los nodos que aparecían en algún par, así que las
    entidades aisladas desaparecían sin aviso y no podían recibir recomendación.
  - Leiden además de Louvain, con semilla fija (resultados reproducibles).
  - Grafo de k vecinos calculado en streaming sobre los parquet de la fase 1.
"""
import gzip
import pickle
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import igraph as ig
import numpy as np
import pandas as pd

from .storage import LocalStage


# ============================================================================
# CONSTRUCCIÓN DEL GRAFO
# ============================================================================
def pairs_to_positions(pairs: pd.DataFrame, ids_sorted: np.ndarray):
    """Convierte ENTITY_A / ENTITY_B en posiciones 0..n-1 dentro de ids_sorted."""
    a = np.searchsorted(ids_sorted, pairs["ENTITY_A"].to_numpy()).astype(np.int32)
    b = np.searchsorted(ids_sorted, pairs["ENTITY_B"].to_numpy()).astype(np.int32)
    return a, b, pairs["SIMILARITY"].to_numpy()


def build_graph(ids_sorted: np.ndarray, src: np.ndarray, dst: np.ndarray, weight: np.ndarray) -> ig.Graph:
    """Grafo no dirigido con TODAS las entidades como vértices (atributo entity_id)."""
    g = ig.Graph(n=len(ids_sorted), edges=np.column_stack([src, dst]).tolist(), directed=False)
    g.vs["entity_id"] = ids_sorted.tolist()
    g.es["weight"] = np.asarray(weight, dtype=float).tolist()
    return g


def threshold_graph(pairs: pd.DataFrame, ids_sorted: np.ndarray, threshold: int) -> ig.Graph:
    """pairs puede venir ya filtrado con un umbral menor (se reutiliza en el barrido)."""
    sub = pairs[pairs["SIMILARITY"] >= threshold]
    a, b, s = pairs_to_positions(sub, ids_sorted)
    return build_graph(ids_sorted, a, b, s)


def top_k_neighbors(stage: LocalStage, project: str, metric: str, axis: str,
                    ids_sorted: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    k vecinos más similares de cada entidad, recorriendo los parquet por partes.
    Devuelve (similitudes n×k, posiciones n×k), ordenados de mayor a menor similitud.
    """
    n = len(ids_sorted)
    best_s = np.full((n, k), -32768, dtype=np.int32)
    best_j = np.full((n, k), -1, dtype=np.int32)
    own = np.repeat(np.arange(n, dtype=np.int32), k)

    for part in stage.iter_parts(project, metric, axis):
        a, b, s = pairs_to_positions(part, ids_sorted)
        s = s.astype(np.int32)
        # Cada par cuenta para las dos entidades; se mezcla con el top-k acumulado
        node = np.concatenate([a, b, own])
        nbr = np.concatenate([b, a, best_j.ravel()])
        sim = np.concatenate([s, s, best_s.ravel()])
        order = np.lexsort((-sim, node))
        node, nbr, sim = node[order], nbr[order], sim[order]
        first = np.searchsorted(node, np.arange(n))
        take = first[:, None] + np.arange(k)[None, :]
        best_s, best_j = sim[take], nbr[take]
        del node, nbr, sim, order
    return best_s, best_j


def knn_graph(ids_sorted: np.ndarray, knn_sim: np.ndarray, knn_idx: np.ndarray, k: int,
              mutual: bool = False) -> ig.Graph:
    """
    Grafo de k vecinos. Por defecto, arista si B está entre los k de A O A entre los k de B.
    mutual=True exige las dos cosas (grafo más limpio pero con posibles aislados).
    """
    n = len(ids_sorted)
    src = np.repeat(np.arange(n), k)
    dst = knn_idx[:, :k].ravel()
    w = knn_sim[:, :k].ravel()
    e = pd.DataFrame({"u": np.minimum(src, dst), "v": np.maximum(src, dst), "w": w})
    if mutual:
        e = e[e.duplicated(["u", "v"], keep=False)]
    e = e.drop_duplicates(["u", "v"])
    return build_graph(ids_sorted, e["u"].to_numpy(), e["v"].to_numpy(), e["w"].to_numpy())


# ============================================================================
# DETECCIÓN DE COMUNIDADES
# ============================================================================
def detect_communities(g: ig.Graph, method: str = "leiden", seed: int = 0,
                       resolution: float = 1.0) -> np.ndarray:
    """Devuelve la comunidad de cada vértice (array de enteros)."""
    random.seed(seed)
    ig.set_random_number_generator(random)
    w = "weight" if g.ecount() else None
    if method == "leiden":
        cl = g.community_leiden(objective_function="modularity", weights=w,
                                resolution=resolution, n_iterations=-1)
    elif method == "louvain":
        cl = g.community_multilevel(weights=w, resolution=resolution)
    else:
        raise ValueError(f"Método no soportado: {method}")
    return np.asarray(cl.membership)


def modularity(g: ig.Graph, membership: np.ndarray) -> float:
    if g.ecount() == 0:
        return 0.0
    return float(g.modularity(membership.tolist(), weights="weight"))


# ============================================================================
# CACHÉ VERSIONADA (equivalente al stage del original)
# ============================================================================
class GraphManager:
    """
    Guarda y carga grafo + comunidades en:
        output/graphs/<PROYECTO>/<metrica>/<nombre>/<version>/graph.pkl
                                                            /metadata.pkl.gz
    """

    def __init__(self, root: str | Path, project_name: str, metric: str = "hybrid",
                 name: str = "GRAPH_COMMUNITIES", verbose: bool = True):
        self.root = Path(root)
        self._project_name = project_name
        self._metric = metric
        self._name = name
        self._verbose = verbose

    def _log(self, *a):
        if self._verbose:
            print(*a)

    def version_dir(self, version: str) -> Path:
        return self.root / self._project_name / self._metric / self._name / version

    # ------------------------------------------------------------------
    def save_graph_data(self, g: ig.Graph, membership: np.ndarray, version: str = "v1",
                        extra: Optional[Dict] = None) -> Path:
        folder = self.version_dir(version)
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / "graph.pkl", "wb") as f:
            pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)
        ids = g.vs["entity_id"]
        metadata = {
            "membership": np.asarray(membership).tolist(),
            "mapping": {e: i for i, e in enumerate(ids)},
            "reverse_mapping": dict(enumerate(ids)),
            "n_communities": int(np.max(membership)) + 1,
            "n_nodes": g.vcount(),
            "n_edges": g.ecount(),
            "version": version,
            "created_at": datetime.now().isoformat(),
            **(extra or {}),
        }
        with gzip.open(folder / "metadata.pkl.gz", "wb") as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        self._log(f"💾 Grafo guardado en {folder.relative_to(self.root.parent)} "
                  f"({metadata['n_nodes']:,} nodos, {metadata['n_edges']:,} aristas, "
                  f"{metadata['n_communities']:,} comunidades)")
        return folder

    def load_graph_data(self, version: str = "v1") -> Optional[Dict]:
        folder = self.version_dir(version)
        if not (folder / "graph.pkl").exists():
            self._log(f"   No existe caché para la versión '{version}'")
            return None
        t0 = time.time()
        with open(folder / "graph.pkl", "rb") as f:
            g = pickle.load(f)
        with gzip.open(folder / "metadata.pkl.gz", "rb") as f:
            metadata = pickle.load(f)
        communities = ig.VertexClustering(g, membership=metadata["membership"])
        self._log(f"📂 Grafo '{version}' cargado en {time.time() - t0:.2f}s "
                  f"({metadata['n_nodes']:,} nodos, {metadata['n_communities']:,} comunidades)")
        return {"graph": g, "communities": communities, "membership": np.asarray(metadata["membership"]),
                "mapping": metadata["mapping"], "reverse_mapping": metadata["reverse_mapping"],
                "metadata": metadata}

    def compute_or_load(self, graph: Optional[ig.Graph] = None, method: str = "leiden", seed: int = 0,
                        resolution: float = 1.0, version: str = "v1", force_recompute: bool = False,
                        extra: Optional[Dict] = None):
        """
        Misma firma de salida que el original:
            (graph, communities, membership, mapping, reverse_mapping)
        """
        if not force_recompute:
            cached = self.load_graph_data(version)
            if cached:
                return (cached["graph"], cached["communities"], cached["membership"],
                        cached["mapping"], cached["reverse_mapping"])
        if graph is None:
            raise ValueError("No hay caché: pasa el grafo construido para calcular las comunidades")

        t0 = time.time()
        membership = detect_communities(graph, method, seed, resolution)
        self._log(f"🔄 Comunidades ({method}) calculadas en {time.time() - t0:.2f}s")
        info = {"method": method, "seed": seed, "resolution": resolution, **(extra or {})}
        self.save_graph_data(graph, membership, version, info)
        loaded = self.load_graph_data(version)
        return (loaded["graph"], loaded["communities"], loaded["membership"],
                loaded["mapping"], loaded["reverse_mapping"])
