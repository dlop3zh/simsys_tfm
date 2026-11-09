"""
LocalStage — sustituto local del almacenamiento en la nube del sistema original.

Mantiene la misma estructura de carpetas que en producción:

    output/stage/<PROYECTO>/<metrica>/<eje>/<yyyy>/<mm>/<dd>/<eje>_similitud_<metrica>_part_N.parquet

y ofrece `load_latest()`, que es el equivalente al procedimiento de carga del sistema original:
busca la fecha más reciente de un eje y carga todos sus parquets.
"""
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import pandas as pd
import pyarrow.parquet as pq


class LocalStage:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # ESCRITURA
    # ------------------------------------------------------------------
    def axis_dir(self, project: str, metric: str, axis: str, date: Optional[datetime] = None) -> Path:
        d = date or datetime.now()
        return self.root / project / metric / axis / d.strftime("%Y") / d.strftime("%m") / d.strftime("%d")

    def clear_axis_day(self, project: str, metric: str, axis: str) -> Path:
        """Borra los parquets de hoy de un eje (evita mezclar partes de dos ejecuciones)."""
        path = self.axis_dir(project, metric, axis)
        if path.exists():
            for f in path.glob("*.parquet"):
                f.unlink()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def put(self, df: pd.DataFrame, project: str, metric: str, axis: str, file_name: str) -> Path:
        path = self.axis_dir(project, metric, axis)
        path.mkdir(parents=True, exist_ok=True)
        dest = path / file_name
        df.to_parquet(dest, index=False, compression="snappy")
        return dest

    # ------------------------------------------------------------------
    # LECTURA  (equivalente al procedimiento de carga del sistema original)
    # ------------------------------------------------------------------
    def latest_dir(self, project: str, metric: str, axis: str) -> Path:
        base = self.root / project / metric / axis
        days = sorted(p for p in base.glob("*/*/*") if p.is_dir() and any(p.glob("*.parquet")))
        if not days:
            raise FileNotFoundError(f"No hay datos en el stage para {project}/{metric}/{axis}")
        return days[-1]

    def list_parts(self, project: str, metric: str, axis: str) -> List[Path]:
        folder = self.latest_dir(project, metric, axis)
        return sorted(folder.glob("*.parquet"), key=lambda p: int(p.stem.rsplit("_", 1)[-1]))

    def iter_parts(self, project: str, metric: str, axis: str,
                   min_similarity: Optional[int] = None) -> Iterator[pd.DataFrame]:
        """Recorre las partes una a una (no carga todo en memoria)."""
        filters = [("SIMILARITY", ">=", min_similarity)] if min_similarity is not None else None
        for part in self.list_parts(project, metric, axis):
            yield pq.read_table(part, filters=filters).to_pandas()

    def load_latest(self, project: str, metric: str, axis: str,
                    min_similarity: Optional[int] = None) -> pd.DataFrame:
        """
        Carga todos los pares de la última ejecución de un eje.
        Con 10.000 entidades son ~50M de pares (~450 MB en memoria): usa
        `min_similarity` para quedarte solo con los pares relevantes.
        """
        parts = list(self.iter_parts(project, metric, axis, min_similarity))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["ENTITY_A", "ENTITY_B", "SIMILARITY"])
