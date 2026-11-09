"""
SimSys Local — FASE 2: grafo de similitud y detección de comunidades.

Parte de los pares calculados en la fase 1 (output/stage/) y:
  1. Barre umbrales globales y valores de k para construir el grafo.
  2. Elige la configuración con un criterio NO supervisado (cobertura, estabilidad, modularidad).
  3. Detecta comunidades (Leiden) y guarda grafo + comunidades en caché versionada.
  4. Compara con Louvain, con la configuración original (umbral 84 + Louvain) y con
     clustering clásico (K-Means, K-Prototypes, HDBSCAN).
  5. Perfila cada comunidad y genera un informe Excel.

Equivale al proceso de detección de comunidades del sistema original.

Uso:
    python main_fase2.py                     # todos los proyectos
    python main_fase2.py --proyecto LOCALES
    python main_fase2.py --sin-kprototypes   # más rápido
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
from simsys import DataPreprocessor, LocalStage
from simsys.baselines import run_baselines
from simsys.communities import (RESTO, community_segment_crosstab, compare_graph_methods,
                                knn_sweep, production_reference, profile_communities,
                                relabel_by_size, select_configuration, threshold_sweep)
from simsys.graph_manager import GraphManager, knn_graph, threshold_graph, top_k_neighbors


def ejecutar_proyecto(nombre: str, cfg: dict, p2: dict, kprototypes: bool, verbose: bool):
    print(f"\n{'#' * 78}\n#  FASE 2 · PROYECTO {nombre}\n{'#' * 78}")
    t_total = time.time()
    stage = LocalStage(config.STAGE_DIR)
    metrica, eje, min_size, seeds = p2["metrica"], p2["eje"], p2["tam_min_comunidad"], p2["semillas"]

    # ------------------------------------------------------------------ datos
    df = pd.read_excel(cfg["fichero"], sheet_name=cfg["hoja"])
    ids = np.sort(df[cfg["id"]].to_numpy())
    segments = None
    truth = None
    if cfg.get("columna_validacion"):
        segments = df.set_index(cfg["id"])[cfg["columna_validacion"]]
        truth = segments.loc[ids].to_numpy()
    try:
        stage.latest_dir(nombre, metrica, eje)
    except FileNotFoundError:
        raise SystemExit(f"❌ No hay similitudes de la fase 1 para {nombre}. "
                         f"Ejecuta antes: python main_fase1.py --proyecto {nombre} --metricas {metrica}")

    # ------------------------------------------------------------------ 1. barridos
    print(f"\n[1/5] Barrido de umbral (Leiden, {len(seeds)} semillas por configuración)")
    t0 = time.time()
    pairs = stage.load_latest(nombre, metrica, eje, min_similarity=min(p2["umbrales"]))
    print(f"  {len(pairs):,} pares con similitud >= {min(p2['umbrales'])} ({time.time() - t0:.1f}s)")
    sweep_u = threshold_sweep(pairs, ids, p2["umbrales"], truth, seeds, min_size, p2["resolucion"], verbose)

    print(f"\n[1/5] Barrido de k vecinos (k máx. = {max(p2['ks'])})")
    t0 = time.time()
    knn_sim, knn_idx = top_k_neighbors(stage, nombre, metrica, eje, ids, max(p2["ks"]))
    print(f"  Vecinos calculados en {time.time() - t0:.1f}s")
    sweep_k = knn_sweep(knn_sim, knn_idx, ids, p2["ks"], truth, seeds, min_size, p2["resolucion"], verbose)

    sweep = pd.concat([sweep_u, sweep_k], ignore_index=True)
    elegido = select_configuration(sweep, p2["cobertura_minima"], p2["estabilidad_minima"])
    print(f"\n  ➜ Configuración elegida: {elegido['grafo']} = {elegido['parametro']} "
          f"(cobertura {elegido['cobertura']:.1%}, estabilidad {elegido['estabilidad']:.3f}, "
          f"Q {elegido['modularidad']:.3f})")

    # ------------------------------------------------------------------ 2. grafo final
    print("\n[2/5] Grafo final y comunidades (Leiden)")
    if elegido["grafo"] == "knn":
        g = knn_graph(ids, knn_sim, knn_idx, int(elegido["parametro"]))
    else:
        g = threshold_graph(pairs, ids, int(elegido["parametro"]))
    manager = GraphManager(config.GRAPHS_DIR, nombre, metrica, verbose=verbose)
    g, communities, membership, mapping, reverse_mapping = manager.compute_or_load(
        graph=g, method="leiden", seed=seeds[0], resolution=p2["resolucion"],
        version=p2["version_grafo"], force_recompute=True,
        extra={"grafo": elegido["grafo"], "parametro": int(elegido["parametro"])})
    etiquetas = pd.Series(relabel_by_size(membership, min_size), index=ids, name="COMUNIDAD")

    # ------------------------------------------------------------------ 3. comparación
    print("\n[3/5] Comparación de métodos")
    comp_graph = compare_graph_methods(g, truth, seeds, min_size, p2["resolucion"])
    ref = production_reference(pairs, ids, truth, p2["umbral_original"], min_size, seeds[0])
    print(f"  {ref['metodo']:<28} comunidades={ref['comunidades']:>3}  cobertura={ref['cobertura']:.1%}  "
          f"fuera del grafo={ref['pct_fuera_del_grafo']:.1%}  ARI={ref['ARI']:.3f}  NMI={ref['NMI']:.3f}")
    for _, r in comp_graph.iterrows():
        print(f"  {r['metodo'] + ' (grafo elegido)':<28} comunidades={r['comunidades']:>3}  "
              f"cobertura={r['cobertura']:.1%}  estab={r['estabilidad']:.3f}  ARI={r['ARI']:.3f}  NMI={r['NMI']:.3f}")
    comp_graph["metodo"] = comp_graph["metodo"] + f" · {elegido['grafo']} {elegido['parametro']}"

    prep = DataPreprocessor("StandardScaler", [metrica], cfg["id"], cfg["excluir"], verbose=False)
    prep.fit_transform(df)
    X = prep.processed_data.loc[ids]
    num, cat = prep.columns_of_type("numeric"), prep.columns_of_type("categorical", "boolean")
    k_ref = int(pd.Series(truth).nunique()) if truth is not None else int(comp_graph["comunidades"].iloc[0])
    bl = run_baselines(X, num, cat, k_ref, truth, p2["baselines"]["hdbscan_pca"],
                       p2["baselines"]["hdbscan_min_cluster"],
                       kprototypes and p2["baselines"]["kprototypes"], seeds[0], verbose)
    comparativa = pd.concat([pd.DataFrame([ref]), comp_graph,
                             pd.DataFrame([{k: v for k, v in r.items() if k != "labels"} for r in bl.values()])],
                            ignore_index=True)

    # ------------------------------------------------------------------ 4. perfiles
    print("\n[4/5] Perfilado de comunidades")
    perfiles = profile_communities(prep.processed_data_before_scaling, prep.variable_types, etiquetas,
                                   segments, cfg.get("columnas_ejemplo"))
    cols_show = ["COMUNIDAD", "N", "PCT"] + (["SEGMENTO_DOMINANTE", "PUREZA"] if segments is not None else [])
    print(perfiles[cols_show].to_string(index=False))
    cruce = community_segment_crosstab(etiquetas, segments.loc[ids]) if segments is not None else None

    # ------------------------------------------------------------------ 5. salidas
    print("\n[5/5] Guardando resultados")
    out_dir = config.OUTPUT_DIR / "fase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    asign = pd.DataFrame({cfg["id"]: ids, "COMUNIDAD": etiquetas.to_numpy(), "COMUNIDAD_ID": membership})
    asign.to_parquet(out_dir / f"{nombre}_comunidades.parquet", index=False)

    resumen = pd.DataFrame([
        ("Proyecto", nombre), ("Métrica / eje", f"{metrica} / {eje}"), ("Entidades", len(ids)),
        ("Grafo elegido", f"{elegido['grafo']} = {elegido['parametro']}"),
        ("Criterio de selección", f"cobertura >= {p2['cobertura_minima']:.0%}, estabilidad >= "
                                  f"{p2['estabilidad_minima']:.2f} y máxima modularidad"),
        ("Aristas", g.ecount()), ("Comunidades (>= tamaño mínimo)", int(elegido["comunidades"])),
        ("Tamaño mínimo de comunidad", min_size),
        ("Cobertura", float(comp_graph["cobertura"].iloc[0])),
        ("Modularidad (Leiden)", float(comp_graph["modularidad"].iloc[0])),
        ("Estabilidad (ARI medio entre semillas)", float(comp_graph["estabilidad"].iloc[0])),
        ("ARI vs segmento simulado", float(comp_graph["ARI"].iloc[0])),
        ("NMI vs segmento simulado", float(comp_graph["NMI"].iloc[0])),
        ("Cobertura configuración original (umbral 84 + Louvain)", ref["cobertura"]),
        ("Carpeta del grafo", str(manager.version_dir(p2["version_grafo"]).relative_to(config.BASE_DIR))),
        ("Tiempo total (s)", round(time.time() - t_total, 1)),
    ], columns=["CONCEPTO", "VALOR"])

    hojas = {"Resumen": resumen, "Barrido_grafo": sweep, "Comparativa_metodos": comparativa,
             "Perfiles": perfiles}
    if cruce is not None:
        hojas["Comunidad_x_Segmento"] = cruce.reset_index()
    hojas["Asignaciones"] = asign

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.REPORTS_DIR / f"fase2_{nombre.lower()}.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        for hoja, tabla in hojas.items():
            tabla.to_excel(w, sheet_name=hoja[:31], index=False)
            ws = w.sheets[hoja[:31]]
            for col in ws.columns:
                ancho = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, ancho + 2), 80)
    print(f"📄 Informe: {ruta}")
    print(f"⏱  Proyecto {nombre} completado en {time.time() - t_total:.1f}s")
    return {"sweep": sweep, "elegido": elegido, "comparativa": comparativa, "perfiles": perfiles}


def main():
    parser = argparse.ArgumentParser(description="SimSys Local — Fase 2")
    parser.add_argument("--proyecto", choices=list(config.PROYECTOS) + ["TODOS"], default="TODOS",
                        type=str.upper)
    parser.add_argument("--sin-kprototypes", action="store_true", help="Omitir K-Prototypes (el más lento)")
    parser.add_argument("--silencioso", action="store_true")
    args = parser.parse_args()

    proyectos = config.PROYECTOS if args.proyecto == "TODOS" else {args.proyecto: config.PROYECTOS[args.proyecto]}
    for nombre, cfg in proyectos.items():
        ejecutar_proyecto(nombre, cfg, config.FASE2, kprototypes=not args.sin_kprototypes,
                          verbose=not args.silencioso)


if __name__ == "__main__":
    main()
