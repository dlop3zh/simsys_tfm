"""
SimSys Local — FASE 1: preprocesado + cálculo de similitudes + validación.

Equivale en producción a:
    el procedimiento de cálculo de similitudes (preprocesado + ejes -> stage)
    el procedimiento de carga (lectura del último cálculo)
más un informe de validación que en producción no existía.

Uso:
    python main_fase1.py                                  # todos los proyectos y métricas
    python main_fase1.py --proyecto CLIENTES
    python main_fase1.py --proyecto LOCALES --metricas hybrid
    python main_fase1.py --muestra 2000                   # prueba rápida con 2.000 entidades
    python main_fase1.py --ram-gb 4                       # más RAM -> menos bloques
"""
import argparse
import time

import numpy as np
import pandas as pd

import config
from simsys import GenericSimilaritySystem, LocalStage
from simsys.evaluation import (distribution_summary, histogram_table, top_k_from_pairs,
                               validate_against_segments)


def ejecutar_proyecto(nombre: str, cfg: dict, metricas: list, muestra: int, ram_gb: float,
                      validar: bool, verbose: bool):
    print(f"\n{'#' * 78}\n#  PROYECTO {nombre}\n{'#' * 78}")
    t0 = time.time()
    stage = LocalStage(config.STAGE_DIR)

    df = GenericSimilaritySystem.read_excel(cfg["fichero"], cfg["hoja"], sample=muestra)
    print(f"Leído {cfg['fichero'].name}: {df.shape[0]:,} filas x {df.shape[1]} columnas")

    informe = {}
    resumen_rows = []
    rng = np.random.default_rng(7)
    ejemplos_ids = rng.choice(df[cfg["id"]].to_numpy(), size=3, replace=False)

    for metrica in metricas:
        norm = config.METRICAS[metrica]["normalizacion"]
        print(f"\n>>> Métrica: {metrica.upper()}  (normalización: {norm})")

        system = GenericSimilaritySystem(project_name=nombre, stage=stage, distance_metrics=[metrica],
                                         ram_gb=ram_gb, max_pairs_per_file=config.MAX_PARES_POR_FICHERO,
                                         verbose=verbose)
        system.preprocess_data(df, exclude_variables=cfg["excluir"], normalization=norm,
                               entity_id_column=cfg["id"])
        system.save_processed_data(config.PROCESSED_DIR, suffix=metrica.upper())

        if "Variables" not in informe:
            informe["Variables"] = system.preprocessor.variable_types_table()

        for eje, variables in cfg["ejes"].items():
            res = system.add_axis(eje, variables).get(metrica)
            if res is None:
                continue
            hoja = f"{eje}_{metrica}"[:25]
            informe[f"Dist_{hoja}"] = distribution_summary(res["histograma"])
            informe[f"Hist_{hoja}"] = histogram_table(res["histograma"])
            fila = {k: v for k, v in res.items() if k != "histograma"}
            fila["normalizacion"] = norm
            fila["carpeta"] = str(config.STAGE_DIR.relative_to(config.BASE_DIR) / nombre / metrica / eje)

            if validar and cfg.get("columna_validacion"):
                print(f"  Validando {eje}/{metrica} contra '{cfg['columna_validacion']}'...")
                segs = df.set_index(cfg["id"])[cfg["columna_validacion"]]
                val = validate_against_segments(stage, nombre, metrica, eje, segs,
                                                sample_size=300, k=10, extra_ids=ejemplos_ids)
                informe[f"Val_{hoja}"] = val["global"]
                informe[f"ValSeg_{hoja}"] = val["por_segmento"]
                informe[f"MatSeg_{hoja}"] = val["matriz_segmentos"].reset_index(names="SEGMENTO")
                g = dict(zip(val["global"]["METRICA"], val["global"]["VALOR"]))
                fila.update({k: v for k, v in g.items()})
                print(val["global"].to_string(index=False))

                # Ejemplos cualitativos: top-5 similares de 3 entidades
                cols = cfg["columnas_ejemplo"]
                ficha = df.set_index(cfg["id"])[cols]
                bloques = []
                for eid in ejemplos_ids:
                    ref = ficha.loc[[eid]].reset_index().assign(ROL="REFERENCIA", SIMILARITY=np.nan)
                    top = top_k_from_pairs(val["pares_muestra"], eid, k=5)
                    vec = (ficha.loc[top["VECINO"]].reset_index()
                           .assign(ROL="SIMILAR", SIMILARITY=top["SIMILARITY"].to_numpy()))
                    bloques += [ref, vec]
                ej = pd.concat(bloques, ignore_index=True)
                informe[f"Ejemplos_{hoja}"] = ej[["ROL", cfg["id"], "SIMILARITY"] + cols]
            resumen_rows.append(fila)

    informe = {"Resumen": pd.DataFrame(resumen_rows), **informe}
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.REPORTS_DIR / f"fase1_{nombre.lower()}.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        for hoja, tabla in informe.items():
            tabla.to_excel(w, sheet_name=hoja[:31], index=False)
            ws = w.sheets[hoja[:31]]
            for col in ws.columns:
                ancho = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(10, ancho + 2), 60)
    print(f"\n📄 Informe: {ruta}")
    print(f"⏱  Proyecto {nombre} completado en {time.time() - t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="SimSys Local — Fase 1")
    parser.add_argument("--proyecto", choices=list(config.PROYECTOS) + ["TODOS"], default="TODOS",
                        type=str.upper)
    parser.add_argument("--metricas", nargs="+", choices=list(config.METRICAS),
                        default=list(config.METRICAS))
    parser.add_argument("--muestra", type=int, default=None,
                        help="Usar solo N entidades (pruebas rápidas)")
    parser.add_argument("--ram-gb", type=float, default=config.RAM_GB)
    parser.add_argument("--sin-validacion", action="store_true")
    parser.add_argument("--silencioso", action="store_true")
    args = parser.parse_args()

    proyectos = config.PROYECTOS if args.proyecto == "TODOS" else {args.proyecto: config.PROYECTOS[args.proyecto]}
    for nombre, cfg in proyectos.items():
        if not cfg["fichero"].exists():
            raise FileNotFoundError(f"No existe {cfg['fichero']}. Ejecuta antes: python generar_datos.py")
        ejecutar_proyecto(nombre, cfg, args.metricas, args.muestra, args.ram_gb,
                          validar=not args.sin_validacion, verbose=not args.silencioso)


if __name__ == "__main__":
    main()
