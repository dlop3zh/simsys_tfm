"""Cuenta comunidades desconectadas por dentro (Louvain frente a Leiden) en el grafo kNN elegido
y en el grafo de umbral 84. Uso: python experimentos/conectividad_comunidades.py"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config
from simsys import LocalStage
from simsys.graph_manager import knn_graph, threshold_graph, top_k_neighbors, detect_communities

def desconectadas(g, memb):
    malas = tot = afectados = 0
    for c in np.unique(memb):
        nodos = np.where(memb == c)[0]
        if len(nodos) < 2:
            continue
        tot += 1
        if not g.induced_subgraph(nodos.tolist()).is_connected():
            malas += 1
            comps = g.induced_subgraph(nodos.tolist()).connected_components().sizes()
            afectados += len(nodos) - max(comps)
    return malas, tot, afectados

st = LocalStage(config.STAGE_DIR)
filas = []
for proy in ("CLIENTES", "LOCALES"):
    c = config.PROYECTOS[proy]
    ids = np.sort(pd.read_excel(c["fichero"])[c["id"]].to_numpy())
    ks, kj = top_k_neighbors(st, proy, "hybrid", "General", ids, 15)
    pares = st.load_latest(proy, "hybrid", "General", min_similarity=84)
    grafos = {"kNN k=15": knn_graph(ids, ks, kj, 15), "Umbral 84": threshold_graph(pares, ids, 84)}
    for gname, g in grafos.items():
        for metodo in ("louvain", "leiden"):
            res = [desconectadas(g, detect_communities(g, metodo, s)) for s in range(5)]
            filas.append({"PROYECTO": proy, "GRAFO": gname, "METODO": metodo,
                          "COMUNIDADES_DESCONECTADAS_MEDIA": np.mean([r[0] for r in res]),
                          "COMUNIDADES_MEDIA": np.mean([r[1] for r in res]),
                          "ENTIDADES_FUERA_DEL_BLOQUE_PRINCIPAL": np.mean([r[2] for r in res])})
            print(filas[-1])
pd.DataFrame(filas).to_csv(config.OUTPUT_DIR / "fase2" / "conectividad.csv", index=False)
