import json
import networkx as nx

with open("data/raw/graph.json") as f:
    data = json.load(f)

G = nx.DiGraph()
for page in data:
    src = page["Url"]
    for dst in page["OutLinks"]:
        G.add_edge(src, dst)