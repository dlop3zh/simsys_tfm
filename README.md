# SimSys Local — Sistema de similitud entre entidades

Código del Trabajo Fin de Estudios *Similitud híbrida y detección de comunidades en grafos para la
ampliación de audiencias en campañas CRM* (Máster Universitario en Inteligencia Artificial, UNIR).
Autor: David López Hernández.

Todos los datos del repositorio son **sintéticos**. El trabajo se inspira en un sistema real de una
empresa del sector, pero no incluye su código original ni datos reales de clientes.

Versión local (sin plataforma en la nube ni bases de datos) de un sistema que calcula la similitud entre
todas las entidades de un negocio (clientes, locales…) para, en fases posteriores, detectar
comunidades y recomendar entidades parecidas a un grupo de referencia (p. ej., los clientes
que respondieron bien a una campaña).

El código es **genérico**: todo lo específico de cada negocio vive en `config.py`.

---

## Fase 1: datos + preprocesado + similitudes + validación

```
Excel ─► DataPreprocessor ─► AxisSimilarity (hybrid / volume) ─► parquet por partes ─► informe de validación
```

### Estructura

```
simsys_local/
├── config.py               # proyectos, ejes, métricas, RAM
├── generar_datos.py        # crea data/clientes.xlsx y data/locales.xlsx (sintéticos)
├── main_fase1.py           # ejecuta la fase 1 de principio a fin
├── main_fase2.py           # ejecuta la fase 2 (grafo + comunidades)
├── experimentos/
│   ├── comparar_medidas.py           # comparación de medidas de similitud y prueba de dificultad
│   └── conectividad_comunidades.py   # comunidades desconectadas: Louvain frente a Leiden
├── simsys/
│   ├── data_preprocessor.py   # tipos de variable, nulos, escalado
│   ├── axis_similarity.py     # similitud por pares, por bloques, a parquet
│   ├── similarity_system.py   # orquestador
│   ├── storage.py             # LocalStage: sustituye al almacenamiento en la nube
│   ├── evaluation.py          # distribución, validación y top-k
│   ├── graph_manager.py       # fase 2: construcción del grafo + caché (versión local del gestor de grafos original)
│   ├── communities.py         # fase 2: barridos, selección, comparación y perfiles
│   └── baselines.py           # fase 2: K-Means, K-Prototypes, HDBSCAN
├── notebooks/
│   ├── 01_generacion_y_exploracion_datos.ipynb   # generación + EDA de los datos sintéticos
│   ├── 02_similitudes_y_validacion.ipynb         # fase 1 paso a paso, con gráficos
│   └── 03_grafo_y_comunidades.ipynb              # fase 2 paso a paso, con gráficos
├── data/                   # Excels de entrada
└── output/                 # se genera al ejecutar
    ├── stage/<PROY>/<metrica>/<eje>/yyyy/mm/dd/*.parquet
    ├── processed/          # datos preprocesados (antes y después de escalar)
    ├── graphs/<PROY>/<metrica>/GRAPH_COMMUNITIES/<version>/   # grafo + comunidades (fase 2)
    ├── fase2/<PROY>_comunidades.parquet                      # comunidad de cada entidad
    └── informes/fase1_<proyecto>.xlsx, fase2_<proyecto>.xlsx
```

### Ejecución

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generar_datos.py            # ~10 s, crea los dos Excel
python main_fase1.py --muestra 1500   # prueba rápida (~10 s)
python main_fase1.py               # ejecución completa
```

O con notebooks (mismo código, paso a paso y con gráficos):

```bash
jupyter lab            # abrir notebooks/01_... y luego notebooks/02_...
```

En el notebook 02 cambia `PROYECTO = "CLIENTES"` / `"LOCALES"` en la celda de parámetros.
Los notebooks se entregan ya ejecutados (con `CLIENTES`) para poder leerlos sin lanzarlos.

Opciones de `main_fase1.py`: `--proyecto CLIENTES|LOCALES|TODOS`, `--metricas hybrid volume`,
`--muestra N`, `--ram-gb X`, `--sin-validacion`, `--silencioso`.

Referencia medida (1 CPU, 3 GB de RAM): los dos proyectos completos con las dos métricas
tardan ~90 s, con un pico de ~1,3 GB de RAM y ~500 MB de parquet en disco
(≈50 M de pares por proyecto y métrica).

### Datos sintéticos

| Dataset | Registros | Columnas | ID | Qué representa |
|---|---|---|---|---|
| `clientes.xlsx` | 10.000 | 40 | `PLAYER_ID` | Clientes de una plataforma online de apuestas/casino: demografía, canal, actividad, depósitos, turnover, NGR, CRM |
| `locales.xlsx` | 10.000 | 32 | `LOCAL_ID` | Locales con máquinas de juego (hostelería, salones, bingos): ubicación, tamaño, máquinas, recaudación, servicios, operación |

Cada registro nace de un **perfil latente** (8 por dataset: *Apostador deportivo casual*,
*Jugador de slots*, *Salón premium*, *Bingo moderno*…) con sus propias distribuciones, más ruido,
variables derivadas y nulos (1–20 % según la columna). El perfil se guarda en
`SEGMENTO_SIMULADO`, que **se excluye del cálculo** y solo se usa para validar.

### Métricas de similitud

El notebook 02 incluye una comprobación manual de ambas fórmulas sobre un par de entidades
contra el valor guardado por el sistema.

- **hybrid**: `S = 1 − [α · D_coseno(numéricas) + (1 − α) · D_Gower(categóricas y booleanas)]`,
  con `α = nº numéricas / nº variables`. Rango −100…100 (el coseno puede ser negativo con
  datos estandarizados).
- **volume**: `S = min(‖a‖, ‖b‖) / max(‖a‖, ‖b‖)` sobre las numéricas. Compara la *magnitud*
  global de dos entidades, no su perfil. Rango 0…100.

Se guardan como enteros ×100 y solo el triángulo superior (A < B), igual que en producción.

### Validación (informe Excel)

Para cada proyecto y métrica: distribución de similitudes, % de pares por encima de varios
umbrales, similitud media intra vs. inter segmento, **Precision@10** (de los 10 vecinos más
similares, cuántos comparten segmento) frente a la esperada al azar, y 3 ejemplos cualitativos
con sus 5 más similares.

Resultados de referencia (10.000 entidades, semilla por defecto):

| Proyecto | Métrica | Intra | Inter | Precision@10 | Azar | Mejora |
|---|---|---|---|---|---|---|
| CLIENTES | hybrid | ~49 | ~10 | ~0,85 | ~0,16 | ~5,5× |
| CLIENTES | volume | ~85 | ~72 | ~0,37 | ~0,16 | ~2,4× |
| LOCALES | hybrid | ~48 | ~10 | ~0,84 | ~0,18 | ~4,6× |
| LOCALES | volume | ~85 | ~77 | ~0,28 | ~0,18 | ~1,6× |

Lectura: *hybrid* recupera bien los perfiles; *volume* por sí sola discrimina poco, lo cual es
esperable porque solo mide "tamaño" y no forma.

### Cambios respecto a la versión original

| Área | Original | Local |
|---|---|---|
| Entrada | Vista en la plataforma en la nube | Excel (`pd.read_excel`) |
| Salida | `put_stream` a stage + el procedimiento de carga del sistema original | `LocalStage` con la misma estructura de carpetas + `load_latest()` |
| Ruta por métrica | Usaba siempre `distance_metrics[0]` (con dos métricas, `volume` caía en la carpeta de `hybrid`) | Corregido |
| Gower | Array `batch × n × p` | Columna a columna (memoria ÷ p) |
| Batch | Calculado sobre 50 GB fijos | Según `RAM_GB` y nº máximo de pares por fichero |
| Coseno con vector nulo | NaN → valor basura al pasar a int8 | Distancia neutra (coseno 0) |
| NaN categóricos | Relleno con `0` (mezcla int/str) | `"__NA__"` |
| pandas 3 | Textos como `StringDtype` → tipo `other` | Detectados como categóricos |
| Normalización | En producción se llama con `None` | Configurable por métrica (`config.METRICAS`) |

---

## Fase 2: grafo de similitud y detección de comunidades

```
pares de la fase 1 ─► grafo (umbral o k vecinos) ─► Leiden ─► comunidades ─► perfiles + informe
```

### Ejecución

```bash
python main_fase1.py --metricas hybrid     # si aún no están las similitudes de la fase 1
python main_fase2.py                       # ~3-4 min los dos proyectos (--sin-kprototypes: más rápido)
```

O paso a paso en `notebooks/03_grafo_y_comunidades.ipynb`. Parámetros en `config.FASE2`.

### Qué hace

1. **Barrido de construcción del grafo.** Umbral global (70–86) y k vecinos más cercanos (5–30). Para cada
   configuración, Leiden con 5 semillas y: cobertura, % de aislados, nº de comunidades, modularidad y estabilidad
   (ARI medio entre semillas). ARI y NMI frente al segmento simulado se calculan solo como validación externa.
2. **Selección no supervisada.** Cobertura ≥ 95 %, estabilidad ≥ 0,90 y, entre esas, máxima modularidad.
   El segmento simulado no interviene. En ambos datasets sale un grafo de **k = 15 vecinos**.
3. **Comunidades** con Leiden, guardadas con `GraphManager` (versión local del gestor de grafos original):
   grafo + metadatos versionados, con `compute_or_load()` como en producción.
4. **Comparación** con la configuración original (umbral 84 + Louvain, solo nodos con aristas), con Louvain
   y con clustering clásico: K-Means y K-Prototypes (con el número real de perfiles, referencia optimista)
   y HDBSCAN.
5. **Perfilado** de cada comunidad: tamaño, rasgos característicos (percentil medio en numéricas, categoría
   sobre-representada en categóricas) y cruce con los segmentos simulados.

### Resultados de referencia

| Proyecto | Configuración | Comunidades | Cobertura | Modularidad | Estabilidad | ARI | NMI |
|---|---|---|---|---|---|---|---|
| CLIENTES | Original: umbral 84 + Louvain | 14 | 66,4 % | – | – | 0,35 | 0,51 |
| CLIENTES | **k = 15 + Leiden** | 11 | 100 % | 0,78 | 0,94 | 0,59 | 0,67 |
| CLIENTES | K-Means (k = 8 real) | 8 | 100 % | – | – | 0,55 | 0,68 |
| LOCALES | Original: umbral 84 + Louvain | 13 | 59,2 % | – | – | 0,27 | 0,46 |
| LOCALES | **k = 15 + Leiden** | 9 | 100 % | 0,74 | 0,94 | 0,50 | 0,70 |
| LOCALES | K-Means (k = 8 real) | 8 | 100 % | – | – | 0,51 | 0,66 |

Lectura:

- Con el umbral 84 de producción, el **28,5 % de los clientes y el 34,6 % de los locales quedan fuera del grafo**:
  no tienen comunidad y no pueden recibir recomendaciones. El grafo de k vecinos cubre el 100 %.
- Sin conocer el número de perfiles, las comunidades logran un acuerdo con los segmentos similar al de K-Means
  cuando K-Means recibe el número real. HDBSCAN no separa bien perfiles que se solapan.
- Los perfiles grandes se dividen en subgrupos con sentido de negocio: por ejemplo, los *Bares de barrio* se separan
  por horario reducido y por antigüedad del último recanje.

### Cambios respecto al gestor de grafos original

| Área | Original | Local |
|---|---|---|
| Vértices | Solo entidades con alguna arista (las aisladas desaparecen) | Todas las entidades |
| Construcción | Umbral fijo (84, filtrado previo ≥ 80) | Umbral o k vecinos, elegido por barrido |
| Algoritmo | Louvain sin semilla | Leiden (y Louvain) con semilla fija |
| Almacenamiento | Stage de la plataforma original + caché en /tmp | `output/graphs/…/<version>/` |

---

## Reproducir los resultados de la memoria

```bash
pip install -r requirements.txt
python main_fase1.py --metricas hybrid      # similitudes de los dos proyectos (~1 min)
python main_fase2.py                        # barrido del grafo, comunidades y comparación de métodos (~3 min)
python experimentos/comparar_medidas.py     # Tabla 5 y Figuras 5 y 6 de la memoria
python experimentos/conectividad_comunidades.py
```

Los resultados se guardan en `output/` (informes Excel en `output/informes/` y tablas de los experimentos
en `output/fase2/`). Todas las semillas son fijas, así que los números coinciden con los de la memoria.

---

## Planificación

| Entrega | Fecha | Contenido |
|---|---|---|
| 1 | 5 de octubre | Datos sintéticos, preprocesado, similitudes por pares y primera validación |
| **2 (esta)** | 9 de noviembre | Comparación de medidas, prueba de dificultad, grafo de similitud (umbral y k vecinos), comunidades con Louvain y Leiden, comparación con agrupamiento clásico y perfilado |
| 3 | 7 de diciembre | Recomendador de audiencias y su evaluación, discusión y conclusiones |
| Predepósito / depósito | 3 y 10 de febrero | Correcciones y versión final |
