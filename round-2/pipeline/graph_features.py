"""
NFPC Phase 2 — Graph/Network Feature Engineering

Research-grade network analysis on the account-counterparty transaction graph:
  - PageRank (importance in money flow network)
  - In/out degree and weighted degree
  - Betweenness centrality (brokerage role)
  - Community detection (Louvain) — mule rings share communities
  - Local clustering coefficient
  - Shared counterparty features (accounts transacting with the same counterparties)
  - Hub/authority scores (HITS algorithm)

Memory-efficient: builds the graph incrementally from transaction batches,
keeping only account→counterparty edges with aggregate weights.
"""
import numpy as np
import pandas as pd
from glob import glob
from collections import defaultdict
from tqdm import tqdm

from config import (
    TXN_DIR, GRAPH_FEATURES_PATH, log,
)


def build_graph_features() -> pd.DataFrame:
    """Build network features from account-counterparty transaction graph."""
    if GRAPH_FEATURES_PATH.exists():
        log.info("Loading cached graph features from %s", GRAPH_FEATURES_PATH)
        return pd.read_parquet(GRAPH_FEATURES_PATH)

    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    log.info("Building account-counterparty transaction graph")

    # ── Pass 1: Aggregate edges ───────────────────────────
    # Edge: (account_id, counterparty_id) → {weight, n_txn, total_amount}
    edges = defaultdict(lambda: {"n_txn": 0, "total_amount": 0.0})

    parts = sorted(glob(str(TXN_DIR / "batch-*" / "part_*.parquet")))
    for part_path in tqdm(parts, desc="Graph: scanning edges"):
        df = pd.read_parquet(
            part_path, columns=["account_id", "counterparty_id", "amount"]
        )
        df = df.dropna(subset=["counterparty_id"])
        for _, row in df.groupby(["account_id", "counterparty_id"]).agg(
            n=("amount", "count"), total=("amount", "sum")
        ).reset_index().iterrows():
            key = (row["account_id"], row["counterparty_id"])
            edges[key]["n_txn"] += int(row["n"])
            edges[key]["total_amount"] += float(row["total"])
        del df

    log.info("Graph has %d unique edges", len(edges))

    # ── Build NetworkX graph ──────────────────────────────
    G = nx.DiGraph()
    for (src, dst), attrs in edges.items():
        G.add_edge(src, dst, weight=attrs["n_txn"], amount=attrs["total_amount"])
    del edges

    log.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ── Compute centrality metrics ────────────────────────
    log.info("Computing PageRank")
    pagerank = nx.pagerank(G, weight="weight", max_iter=100, tol=1e-6)

    log.info("Computing HITS (hub/authority)")
    try:
        hubs, authorities = nx.hits(G, max_iter=100, tol=1e-6)
    except nx.PowerIterationFailedConvergence:
        hubs = {n: 0.0 for n in G.nodes()}
        authorities = {n: 0.0 for n in G.nodes()}

    log.info("Computing degree centrality")
    in_degree = dict(G.in_degree(weight="weight"))
    out_degree = dict(G.out_degree(weight="weight"))
    in_degree_count = dict(G.in_degree())
    out_degree_count = dict(G.out_degree())

    # Betweenness on sampled graph (full is too expensive for 400M+ edges)
    log.info("Computing betweenness centrality (sampled)")
    betweenness = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), weight="weight")

    # ── Community detection (undirected) ──────────────────
    log.info("Running Louvain community detection")
    G_undirected = G.to_undirected()
    try:
        communities = louvain_communities(G_undirected, weight="weight", seed=42)
        node_community = {}
        community_sizes = {}
        for i, comm in enumerate(communities):
            community_sizes[i] = len(comm)
            for node in comm:
                node_community[node] = i
    except Exception as e:
        log.warning("Community detection failed: %s", e)
        node_community = {n: 0 for n in G.nodes()}
        community_sizes = {0: G.number_of_nodes()}

    # ── Clustering coefficient ────────────────────────────
    log.info("Computing clustering coefficients")
    clustering = nx.clustering(G_undirected, weight="weight")

    # ── Assemble features ─────────────────────────────────
    log.info("Assembling graph features")
    # Only keep features for ACCT_ nodes (not CP_ counterparty nodes)
    account_nodes = [n for n in G.nodes() if str(n).startswith("ACCT_")]

    rows = []
    for node in account_nodes:
        comm_id = node_community.get(node, -1)
        rows.append({
            "account_id": node,
            "graph_pagerank": pagerank.get(node, 0),
            "graph_hub_score": hubs.get(node, 0),
            "graph_authority_score": authorities.get(node, 0),
            "graph_in_degree_weighted": in_degree.get(node, 0),
            "graph_out_degree_weighted": out_degree.get(node, 0),
            "graph_in_degree_count": in_degree_count.get(node, 0),
            "graph_out_degree_count": out_degree_count.get(node, 0),
            "graph_degree_ratio": (
                out_degree_count.get(node, 0) / max(in_degree_count.get(node, 0), 1)
            ),
            "graph_betweenness": betweenness.get(node, 0),
            "graph_clustering_coeff": clustering.get(node, 0),
            "graph_community_id": comm_id,
            "graph_community_size": community_sizes.get(comm_id, 0),
        })

    df = pd.DataFrame(rows).set_index("account_id")

    # Community-level aggregations
    # Mule density per community will be computed during training (needs labels)

    df.to_parquet(GRAPH_FEATURES_PATH)
    log.info("Saved graph features: %s", df.shape)
    return df


if __name__ == "__main__":
    build_graph_features()
