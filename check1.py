from rag import load_index, DOMAIN
from guarded_rag import DropConfidential

index = load_index()
nodes = index.as_retriever(similarity_top_k=DOMAIN["top_k"]).retrieve(
    "What is the daily meal limit for international business travel?")
kept = DropConfidential().postprocess_nodes(nodes)
print(len(nodes), "retrieved,", len(kept), "kept")
for n in kept:
    print(n.node.metadata.get("file_path", "").split("data/northpeak/")[-1], round(n.score, 3))