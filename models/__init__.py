"""Model architecture modules for protein function prediction."""
from models.sequence_encoder import SequenceEncoder
from models.graph_encoder import GraphSAGEEncoder
from models.fusion import (
    ConcatenationFusion,
    WeightedSumFusion,
    AttentionFusion,
    GatedFusion,
    CrossModalAttentionFusion,
    get_fusion_module,
)
from models.classifier import ClassificationHead
from models.sequence_only import SequenceOnlyModel
from models.network_only import NetworkOnlyModel
from models.multimodal import MultimodalModel
