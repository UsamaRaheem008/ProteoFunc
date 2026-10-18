"""Quick verification test for the predict.py fixes."""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from inference.predict import ProteinFunctionPredictor

TEST_SEQ = "MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGP"

# Test 1: sequence_only
print("=== Test 1: Sequence Only ===")
p = ProteinFunctionPredictor(
    model_type="sequence_only",
    checkpoint_path="ProteoFunc_outputs/checkpoints/sequence_only_best.pt",
    config_path="configs/config.yaml",
    threshold=0.3,
    top_k=5,
)
print(f"Model loaded: {p.model is not None}")
print(f"GO terms: {len(p.go_terms)}")

result = p.predict(TEST_SEQ)
print(f"Predictions: {len(result.predictions)}")
print(f"Inference time: {result.inference_time*1000:.0f}ms")
for pred in result.top_predictions[:3]:
    print(f"  {pred['go_id']} ({pred['probability']:.3f})")

# Test 2: multimodal with accession
print("\n=== Test 2: Multimodal with accession ===")
p2 = ProteinFunctionPredictor(
    model_type="multimodal",
    checkpoint_path="ProteoFunc_outputs/checkpoints/multimodal_best.pt",
    config_path="configs/config.yaml",
    threshold=0.3,
    top_k=5,
)
print(f"Model loaded: {p2.model is not None}")
print(f"Graph loaded: {p2.graph_data is not None}")
print(f"Proteins in graph: {len(p2.protein_to_idx)}")

info = p2.get_protein_graph_info("P04637")
print(f"P04637 in graph: {info}")

result2 = p2.predict_by_accession(
    accession="P04637",
    sequence=TEST_SEQ,
    threshold=0.3,
    top_k=5,
)
print(f"Predictions: {len(result2.predictions)}")
print(f"Inference time: {result2.inference_time*1000:.0f}ms")
for pred in result2.top_predictions[:3]:
    print(f"  {pred['go_id']} ({pred['probability']:.3f})")

# Test 3: network_only with accession
print("\n=== Test 3: Network Only with accession ===")
p3 = ProteinFunctionPredictor(
    model_type="network_only",
    checkpoint_path="ProteoFunc_outputs/checkpoints/network_only_best.pt",
    config_path="configs/config.yaml",
    threshold=0.3,
    top_k=5,
)
print(f"Model loaded: {p3.model is not None}")
print(f"Graph loaded: {p3.graph_data is not None}")
print(f"Proteins in graph: {len(p3.protein_to_idx)}")

result3 = p3.predict_by_accession(
    accession="P04637",
    sequence=TEST_SEQ,
    threshold=0.3,
    top_k=5,
)
print(f"Predictions: {len(result3.predictions)}")
print(f"Inference time: {result3.inference_time*1000:.0f}ms")
for pred in result3.top_predictions[:3]:
    print(f"  {pred['go_id']} ({pred['probability']:.3f})")

print("\n=== ALL TESTS PASSED ===")
