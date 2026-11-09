"""
DataPreprocessor — versión local.

Basado en ClassDataPreprocessor.py (versión original). Cambios respecto al original:
  - Compatible con pandas 2.x y 3.x (en pandas 3 los textos son StringDtype, no 'object').
  - Los NaN de variables categóricas se rellenan con "__NA__" en vez de 0
    (evita mezclar enteros y textos en la misma columna).
  - Las variables 'binary_numeric' ya no se pierden al escalar.
  - Sin prints de depuración; todo el log va controlado por `verbose`.
  - Se guarda la mediana usada en cada columna (útil para documentar y para
    transformar datos nuevos en fases posteriores).
"""
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler, StandardScaler

_SCALERS = {
    "standardscaler": StandardScaler, "standard": StandardScaler,
    "minmaxscaler": MinMaxScaler, "minmax": MinMaxScaler,
    "robustscaler": RobustScaler, "robust": RobustScaler,
    "maxabsscaler": MaxAbsScaler, "maxabs": MaxAbsScaler,
}

CATEGORICAL_NA = "__NA__"


class DataPreprocessor:
    def __init__(self,
                 normalization: Optional[str] = "StandardScaler",
                 distance_metrics: List[str] = ("hybrid", "volume"),
                 entity_id_column: str = "entity_id",
                 exclude_variables: Optional[List[str]] = None,
                 pairwise_valid_only: bool = False,
                 verbose: bool = False):

        self._normalization = normalization
        self._distance_metrics = list(distance_metrics)
        self._exclude_variables = list(exclude_variables or [])
        self._entity_id_column = entity_id_column
        self._pairwise_valid_only = pairwise_valid_only
        self._verbose = verbose

        self._variable_types: Dict[str, str] = {}
        self._fill_values: Dict[str, object] = {}
        self._processed_data: Optional[pd.DataFrame] = None
        self._processed_data_before_scaling: Optional[pd.DataFrame] = None
        self._original_data: Optional[pd.DataFrame] = None
        self._feature_names: List[str] = []

        if normalization is None or str(normalization).lower() == "none":
            self._scaler = None
        elif str(normalization).lower() in _SCALERS:
            self._scaler = _SCALERS[str(normalization).lower()]()
        else:
            raise ValueError("normalization debe ser 'StandardScaler', 'MinMaxScaler', "
                             "'RobustScaler', 'MaxAbsScaler' o None")

    # ------------------------------------------------------------------
    def _log(self, *args):
        if self._verbose:
            print(*args)

    # ==================================================================
    # PIPELINE PRINCIPAL
    # ==================================================================
    def fit_transform(self, df: pd.DataFrame) -> "DataPreprocessor":
        if self._entity_id_column not in df.columns:
            raise ValueError(f"❌ Columna '{self._entity_id_column}' no encontrada en el DataFrame")
        if len(df) < 2:
            raise ValueError(f"❌ Se necesitan al menos 2 entidades (tienes {len(df)})")
        if df[self._entity_id_column].duplicated().any():
            dups = df.loc[df[self._entity_id_column].duplicated(), self._entity_id_column].unique()[:10]
            raise ValueError(f"❌ IDs duplicados encontrados (primeros): {list(dups)}")

        self._log("\n🚀 INICIANDO PREPROCESADO DE DATOS 🚀")
        self._log(f"  Normalización: {self._normalization}")
        self._log(f"  Métricas: {', '.join(self._distance_metrics)}")
        self._log(f"  Columna ID: {self._entity_id_column}")
        self._log(f"  Variables excluidas: {self._exclude_variables or 'Ninguna'}")
        self._log(f"  Dimensiones entrada: {df.shape[0]:,} entidades x {df.shape[1]} columnas")

        self._original_data = df.copy()
        self._variable_types = self.detect_variable_types(df)
        self._processed_data = self.preprocess_data(df)
        self._processed_data_before_scaling = self._processed_data.copy()

        num_cols = self.columns_of_type("numeric")
        other_cols = [c for c in self._processed_data.columns if c not in num_cols]

        self._log("\n" + "=" * 70 + "\nFASE 3: NORMALIZACIÓN DE DATOS\n" + "=" * 70)
        if self._scaler is not None and num_cols:
            scaled = pd.DataFrame(
                self._scaler.fit_transform(self._processed_data[num_cols]).astype(np.float32),
                index=self._processed_data.index, columns=num_cols)
            self._processed_data = pd.concat([scaled, self._processed_data[other_cols]], axis=1)
            self._log(f"  {self._normalization} aplicado a {len(num_cols)} columnas numéricas")
            if "volume" in self._distance_metrics:
                self._log(f"  ⚠️  Con 'volume', {self._normalization} cambia la interpretación de las "
                          f"magnitudes originales (ver README)")
        else:
            self._log("  Sin normalización (normalization=None)")

        self._feature_names = list(self._processed_data.columns)
        return self

    # ==================================================================
    # FASE 1: TIPOS DE VARIABLE
    # ==================================================================
    def detect_variable_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Tipos detectados:
          - 'boolean'     : True/False o 0/1
          - 'numeric'     : numéricas con más de dos valores
          - 'categorical' : texto / categóricas
          - 'other'       : el resto (fechas, etc.) -> se ignoran
        """
        self._log("\n" + "=" * 70 + "\nFASE 1: DETECCIÓN DE TIPOS DE VARIABLES\n" + "=" * 70)
        types = {}
        for column in df.columns:
            if column == self._entity_id_column or column in self._exclude_variables:
                continue
            s = df[column]
            uniques = set(s.dropna().unique())
            if s.dtype == bool or (uniques and uniques.issubset({0, 1, True, False})):
                types[column] = "boolean"
            elif pd.api.types.is_numeric_dtype(s):
                types[column] = "numeric"
            elif (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)
                  or isinstance(s.dtype, pd.CategoricalDtype)):
                types[column] = "categorical"
            else:
                types[column] = "other"

        if self._verbose:
            counts = pd.Series(types).value_counts()
            print(f"  Total variables analizadas: {len(types)}")
            for t, c in counts.items():
                print(f"  - {t}: {c}")
        return types

    # ==================================================================
    # FASE 2: LIMPIEZA
    # ==================================================================
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log("\n" + "=" * 70 + "\nFASE 2: PREPROCESAMIENTO DE DATOS\n" + "=" * 70)

        out = df.set_index(self._entity_id_column)
        keep = [c for c, t in self._variable_types.items() if t != "other"]
        dropped = [c for c in out.columns if c not in keep]
        if dropped:
            self._log(f"  Columnas no usadas (excluidas u 'other'): {dropped}")
        out = out[keep].copy()

        nan_total = 0
        for col in keep:
            t = self._variable_types[col]
            n_nan = int(out[col].isna().sum())
            if t == "categorical":
                s = out[col].astype(object)
                if n_nan and not self._pairwise_valid_only:
                    s = s.where(s.notna(), CATEGORICAL_NA)
                    self._fill_values[col] = CATEGORICAL_NA
                out[col] = s.astype(str) if not self._pairwise_valid_only else s
            elif t == "boolean":
                s = out[col].astype(float)
                if n_nan and not self._pairwise_valid_only:
                    s = s.fillna(0)
                    self._fill_values[col] = 0
                out[col] = s.astype(np.int8) if not s.isna().any() else s
            elif t == "numeric":
                s = pd.to_numeric(out[col], errors="coerce").astype(float)
                if n_nan and not self._pairwise_valid_only:
                    med = float(s.median())
                    s = s.fillna(med)
                    self._fill_values[col] = med
                    self._log(f"  [NaN] {col}: {n_nan} valores -> mediana ({med:,.2f})")
                out[col] = s.astype(np.float32)
            nan_total += n_nan

        if self._pairwise_valid_only:
            self._log("  MODO PAIRWISE: NaN mantenidos (⚠️ aún no soportado en el cálculo de ejes)")
        else:
            self._log(f"  Total NaN rellenados: {nan_total:,}")

        if "volume" in self._distance_metrics:
            neg = [c for c in self.columns_of_type("numeric") if (out[c] < 0).any()]
            if neg:
                self._log(f"  ⚠️  'volume' usa la norma L2 (-1000 y +1000 pesan igual). Variables con "
                          f"negativos antes de escalar: {neg}")

        self._log(f"  Dimensiones finales: {out.shape[0]:,} entidades x {out.shape[1]} variables")
        mb = out.memory_usage(deep=True).sum() / 1024 ** 2
        self._log(f"  Memoria: {mb:.2f} MB")
        return out

    # ------------------------------------------------------------------
    def columns_of_type(self, *types: str) -> List[str]:
        return [c for c, t in self._variable_types.items() if t in types]

    def variable_types_table(self) -> pd.DataFrame:
        """Resumen de variables para el informe: tipo, nulos y valor de imputación."""
        rows = []
        for col, t in self._variable_types.items():
            orig = self._original_data[col]
            rows.append({
                "VARIABLE": col,
                "TIPO": t,
                "NULOS": int(orig.isna().sum()),
                "PCT_NULOS": round(float(orig.isna().mean()), 4),
                "VALORES_UNICOS": int(orig.nunique()),
                "IMPUTACION": self._fill_values.get(col, ""),
            })
        return pd.DataFrame(rows)

    # ==================================================================
    # PROPIEDADES
    # ==================================================================
    @property
    def normalization(self): return self._normalization

    @property
    def distance_metrics(self): return self._distance_metrics

    @property
    def exclude_variables(self): return self._exclude_variables

    @property
    def entity_id_column(self): return self._entity_id_column

    @property
    def pairwise_valid_only(self): return self._pairwise_valid_only

    @property
    def verbose(self): return self._verbose

    @property
    def variable_types(self): return self._variable_types

    @property
    def scaler(self): return self._scaler

    @property
    def processed_data(self): return self._processed_data

    @property
    def processed_data_before_scaling(self): return self._processed_data_before_scaling

    @property
    def original_data(self): return self._original_data

    @property
    def feature_names(self): return self._feature_names
