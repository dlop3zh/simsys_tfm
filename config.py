"""
Configuración de SimSys Local — Fase 1.

Aquí se define TODO lo que cambia entre proyectos: fichero de entrada, columna ID,
variables excluidas y ejes. El código de simsys/ es genérico y no sabe nada de
clientes ni de locales.
"""
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
STAGE_DIR = OUTPUT_DIR / "stage"          # equivale al almacenamiento del sistema original
PROCESSED_DIR = OUTPUT_DIR / "processed"  # equivale a TB_SIMSYS_*_PROCESSED_DATA
REPORTS_DIR = OUTPUT_DIR / "informes"

# ----------------------------------------------------------------------------
# Recursos. Ajusta RAM_GB a lo que quieras dedicar en tu portátil.
# ----------------------------------------------------------------------------
RAM_GB = 2.0
MAX_PARES_POR_FICHERO = 10_000_000

# ----------------------------------------------------------------------------
# Métricas y normalización.
# En el sistema original se llama a preproces_data con
# normalization=None, es decir, SIN escalar. Aquí se deja configurable para poder
# compararlo en la memoria del máster.
# ----------------------------------------------------------------------------
METRICAS = {
    "hybrid": {"normalizacion": "StandardScaler"},
    "volume": {"normalizacion": "MinMaxScaler"},   # MinMax evita negativos en la norma L2
}

# ----------------------------------------------------------------------------
# Proyectos (un "negocio" = un Excel)
#   ejes: {"NombreEje": "*"}  -> todas las variables no excluidas
#         {"NombreEje": [...]} -> lista explícita de variables
# ----------------------------------------------------------------------------
PROYECTOS = {
    "CLIENTES": {
        "fichero": DATA_DIR / "clientes.xlsx",
        "hoja": 0,
        "id": "PLAYER_ID",
        "excluir": ["SEGMENTO_SIMULADO"],       # ground truth: NO entra en el cálculo
        "columna_validacion": "SEGMENTO_SIMULADO",
        "columnas_ejemplo": ["SEGMENTO_SIMULADO", "EDAD", "PRODUCTO_FAVORITO",
                             "NUM_ACCESOS_30D", "TOTAL_DEPOSITADO_30D", "DIAS_DESDE_ULTIMO_ACCESO"],
        "ejes": {"General": "*"},
    },
    "LOCALES": {
        "fichero": DATA_DIR / "locales.xlsx",
        "hoja": 0,
        "id": "LOCAL_ID",
        "excluir": ["SEGMENTO_SIMULADO"],
        "columna_validacion": "SEGMENTO_SIMULADO",
        "columnas_ejemplo": ["SEGMENTO_SIMULADO", "TIPO_LOCAL", "TIPO_ZONA",
                             "NUM_MAQUINAS_B", "SUPERFICIE_M2", "RECAUDACION_MEDIA_MENSUAL"],
        "ejes": {"General": "*"},
    },
}

# ----------------------------------------------------------------------------
# FASE 2: grafo y comunidades
# ----------------------------------------------------------------------------
GRAPHS_DIR = OUTPUT_DIR / "graphs"        # equivale a la carpeta del grafo en el stage

FASE2 = {
    "metrica": "hybrid",
    "eje": "General",
    "umbrales": [70, 72, 74, 76, 78, 80, 82, 84, 86],   # grafo por umbral global
    "ks": [5, 10, 15, 20, 30],                          # grafo de k vecinos
    "umbral_original": 84,                              # el de producción, como referencia
    "tam_min_comunidad": 20,         # comunidades más pequeñas -> "Resto" (no útiles para campañas)
    "cobertura_minima": 0.95,        # criterio de selección (no supervisado) ...
    "estabilidad_minima": 0.90,      # ... y entre las que cumplen, máxima modularidad
    "semillas": [0, 1, 2, 3, 4],     # para medir la estabilidad
    "resolucion": 1.0,               # resolución de Leiden / Louvain
    "version_grafo": "v1",
    "baselines": {"hdbscan_pca": 5, "hdbscan_min_cluster": 50, "kprototypes": True},
}
