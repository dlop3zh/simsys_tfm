"""SimSys Local — sistema de similitud entre entidades (versión local, sin plataforma en la nube)."""
from .data_preprocessor import DataPreprocessor
from .axis_similarity import AxisSimilarity
from .similarity_system import GenericSimilaritySystem
from .storage import LocalStage
from .graph_manager import GraphManager

__all__ = ["DataPreprocessor", "AxisSimilarity", "GenericSimilaritySystem", "LocalStage", "GraphManager"]
__version__ = "0.2.0-fase2"
