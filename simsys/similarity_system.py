"""
GenericSimilaritySystem — versión local.

Basado en ClassGenericSimilaritySystem.py. Orquesta:
  1. Lectura del Excel de entrada
  2. Preprocesado (DataPreprocessor)
  3. Cálculo de similitudes por eje (AxisSimilarity) -> parquet en LocalStage
"""
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .axis_similarity import AxisSimilarity
from .data_preprocessor import DataPreprocessor
from .storage import LocalStage


class GenericSimilaritySystem:
    def __init__(self,
                 project_name: str,
                 stage: LocalStage,
                 distance_metrics: List[str] = ("hybrid", "volume"),
                 pairwise_valid_only: bool = False,
                 ram_gb: float = 2.0,
                 max_pairs_per_file: int = 10_000_000,
                 verbose: bool = False):
        self._project_name = project_name
        self._stage = stage
        self._distance_metrics = list(distance_metrics)
        self._pairwise_valid_only = pairwise_valid_only
        self._ram_gb = ram_gb
        self._max_pairs_per_file = max_pairs_per_file
        self._verbose = verbose

        self._preprocessor: Optional[DataPreprocessor] = None
        self._axes: Dict[str, AxisSimilarity] = {}
        self._axis_results: Dict[str, Dict[str, dict]] = {}

    # ==================================================================
    # ENTRADA
    # ==================================================================
    @staticmethod
    def read_excel(path: str | Path, sheet_name=0, sample: Optional[int] = None,
                   seed: int = 42) -> pd.DataFrame:
        df = pd.read_excel(path, sheet_name=sheet_name)
        if sample and sample < len(df):
            df = df.sample(sample, random_state=seed).reset_index(drop=True)
        return df

    # ==================================================================
    # PREPROCESADO
    # ==================================================================
    def preprocess_data(self, df: pd.DataFrame,
                        exclude_variables: Optional[List[str]] = None,
                        normalization: Optional[str] = "StandardScaler",
                        entity_id_column: str = "entity_id") -> "GenericSimilaritySystem":
        self._preprocessor = DataPreprocessor(normalization, self._distance_metrics, entity_id_column,
                                              exclude_variables, self._pairwise_valid_only, self._verbose)
        self._preprocessor.fit_transform(df)
        return self

    # Alias con el nombre del código original
    preproces_data = preprocess_data

    def save_processed_data(self, out_dir: str | Path, suffix: str = "") -> List[Path]:
        """Equivalente a los write_pandas de TB_SIMSYS_<PROY>_<METRICA>_PROCESSED_DATA[_BEFORE_SCALING]."""
        self._check_preprocessed()
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = f"{self._project_name}{('_' + suffix) if suffix else ''}"
        p1 = out_dir / f"{base}_PROCESSED_DATA.parquet"
        p2 = out_dir / f"{base}_PROCESSED_DATA_BEFORE_SCALING.parquet"
        self._preprocessor.processed_data.reset_index().to_parquet(p1, index=False)
        self._preprocessor.processed_data_before_scaling.reset_index().to_parquet(p2, index=False)
        return [p1, p2]

    # ==================================================================
    # EJES
    # ==================================================================
    def add_axis(self, name: str, variables: List[str]) -> Dict[str, dict]:
        """Crea un eje y calcula sus similitudes. variables='*' usa todas las variables."""
        self._check_preprocessed()
        if name in self._axes:
            raise ValueError(f"Eje '{name}' ya existe")

        data = self._preprocessor.processed_data
        if variables == "*" or variables == ["*"]:
            variables = list(data.columns)

        axis = AxisSimilarity(name=name, variables=variables, project_name=self._project_name,
                              stage=self._stage, distance_metrics=self._distance_metrics,
                              ram_gb=self._ram_gb, max_pairs_per_file=self._max_pairs_per_file,
                              verbose=self._verbose)
        self._axes[name] = axis

        if self._verbose:
            print(f"\n{'=' * 70}\nCalculando eje {name.upper()} ({len(variables)} variables)\n{'=' * 70}")
        missing = [v for v in variables if v not in data.columns]
        if missing:
            raise ValueError(f"Eje '{name}': variables no encontradas tras el preprocesado: {missing}")

        self._axis_results[name] = axis.calculate(data[variables], self._preprocessor.variable_types)
        return self._axis_results[name]

    def create_axes_from_definitions(self, axis_definitions: Dict[str, Dict]):
        for name, cfg in axis_definitions.items():
            self.add_axis(name=name, variables=cfg["variables"])

    # ==================================================================
    def _check_preprocessed(self):
        if self._preprocessor is None or self._preprocessor.processed_data is None:
            raise RuntimeError("Primero ejecuta preprocess_data()")

    # ==================================================================
    # PROPIEDADES
    # ==================================================================
    @property
    def project_name(self): return self._project_name

    @property
    def stage(self): return self._stage

    @property
    def distance_metrics(self): return self._distance_metrics

    @property
    def preprocessor(self): return self._preprocessor

    @property
    def axes(self): return self._axes

    @property
    def axis_results(self): return self._axis_results
