"""
Cloud Task Placement: Streamlit Web Dashboard
Algorithms: MST (Kruskal), SSSP (Dijkstra), DP Knapsack, Greedy
Run with: streamlit run app.py
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import random
import time
import pandas as pd
from typing import List, Tuple, Dict

# ==========================================
# 1. CORE GRAPH & ALGORITHM ENGINE
# ==========================================
class CloudAlgorithms:
    @staticmethod
    def kruskal_mst(num_nodes: int, edges: List[Tuple[int, int, int]]) -> Tuple[List, int]:
        parent = list(range(num_nodes))
        def find(i):
            if parent[i] == i: return i
            parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            root_i, root_j = find(i), find(j)
            if root_i != root_j:
                parent[root_i] = root_j
                return True
            return False

        mst_edges, mst_cost = [], 0
        edges.sort() 
        for latency, u, v in edges:
            if union(u, v):
                mst_cost += latency
                mst_edges.append((u, v, latency))
        return mst_edges, mst_cost

    @staticmethod
    def dijkstra(adj_list: Dict, src: int) -> Dict[int, int]:
        distances = {node: float('inf') for node in adj_list}
        distances[src] = 0
        pq = [(0, src)]

        while pq:
            curr_dist, u = heapq.heappop(pq)
            if curr_dist > distances[u]: continue
                
            for v, weight in adj_list[u]:
                if curr_dist + weight < distances[v]:
                    distances[v] = curr_dist + weight
                    heapq.heappush(pq, (distances[v], v))
        return distances

    @staticmethod
    def dp_knapsack(capacity: int, tasks: List[Tuple[str, int, int]]) -> Tuple[int, List[str]]:
        n = len(tasks)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]

        for i in range(1, n + 1):
            t_id, weight, val = tasks[i-1]
            for w in range(1, capacity + 1):
                if weight <= w:
                    dp[i][w] = max(val + dp[i-1][w-weight], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]

        selected, w = [], capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                t_id, weight, val = tasks[i-1]
                selected.append(t_id)
                w -= weight
        return dp[n][capacity], selected

    @staticmethod
    def greedy_scheduler(capacity: int, tasks: List[Tuple[str, int, int]]) -> Tuple[int, List[str]]:
        sorted_tasks = sorted(tasks, key=lambda x: x[2]/x[1], reverse=True)
        total_val, curr_weight, selected = 0, 0, []
        
        for t_id, weight, val in sorted_tasks:
            if curr_weight + weight <= capacity:
                selected.append(t_id)
                curr_weight += weight
                total_val += val
        return total_val, selected

    @staticmethod
    def floyd_warshall(V: int, adj_list: Dict) -> List[List[float]]:
        """O(V^3) - APSP: Computes latency between EVERY pair of servers."""
        dist = [[float('inf')] * V for _ in range(V)]
        for i in range(V):
            dist[i][i] = 0
            for v, weight in adj_list[i]:
                dist[i][v] = weight

        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
        return dist

# ==========================================
# 2. STREAMLIT UI & SIMULATION
# ==========================================
st.set_page_config(page_title="Cloud Task Placement", layout="wide")
st.title("☁️️ Cloud Task Placement & Routing Dashboard")
st.markdown("DAA Project Prototype | **Algorithms:** MST (Kruskal), SSSP (Dijkstra), DP (Knapsack), Greedy")

# --- Sidebar Inputs ---
st.sidebar.header("Experimentation Parameters")
V = st.sidebar.slider("Number of Servers (Graph Size)", 5, 50, 15)
# Ensure max edges don't exceed fully connected graph limit
max_edges = int(V * (V - 1) / 2)
E = st.sidebar.slider("Number of Links (Density)", V - 1, max_edges, min(25, max_edges))
TASKS = st.sidebar.slider("Number of Incoming VMs/Tasks", 5, 100, 30)

if st.sidebar.button("Run Simulation & Generate Policy Outputs"):
    
    # 1. Generate Dataset
    G = nx.gnm_random_graph(V, E)
    while not nx.is_connected(G):
        G = nx.gnm_random_graph(V, E)

    adj_list = {i: [] for i in range(V)}
    edge_list = []
    capacities = {i: random.choice([32, 64, 128]) for i in range(V)}
    tasks = [(f"VM-{i}", random.randint(4, 32), random.randint(10, 100)) for i in range(TASKS)]

    for u, v in G.edges():
        latency = random.randint(1, 20)
        adj_list[u].append((v, latency))
        adj_list[v].append((u, latency))
        edge_list.append((latency, u, v))
        G[u][v]['weight'] = latency

    # 2. Run Benchmarks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Datacenter Topology & MST Backbone")
        start_time = time.perf_counter()
        mst_edges, mst_cost = CloudAlgorithms.kruskal_mst(V, edge_list)
        mst_time = (time.perf_counter() - start_time) * 1000
        
        # Draw Graph
        fig, ax = plt.subplots(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='#a0c4ff', ax=ax)
        nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=9, ax=ax)
        
        mst_ebunch = [(u, v) for u, v, w in mst_edges]
        nx.draw_networkx_edges(G, pos, edgelist=mst_ebunch, width=2.5, edge_color='red', ax=ax)
        
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax)
        ax.axis('off')
        st.pyplot(fig)  # Streamlit rendering instead of plt.show()
        
        st.success(f"**Recommended Backbone Cost:** {mst_cost} ms latency (Computed in {mst_time:.4f} ms)")

    with col2:
        st.subheader("2. Cost/Benefit Table (Server Placement)")
        target_server = 1
        capacity = capacities[target_server]
        st.info(f"Targeting Server {target_server} | CPU Capacity limit: **{capacity} Cores**")
        
        # Run DP
        start_time = time.perf_counter()
        dp_yield, dp_placed = CloudAlgorithms.dp_knapsack(capacity, tasks)
        dp_time = (time.perf_counter() - start_time) * 1000
        
        # Run Greedy
        start_time = time.perf_counter()
        gr_yield, gr_placed = CloudAlgorithms.greedy_scheduler(capacity, tasks)
        gr_time = (time.perf_counter() - start_time) * 1000
        
        # Display Cost/Benefit Table
        metrics_data = {
            "Algorithm": ["DP Knapsack (Optimal)", "Greedy Scheduler (Heuristic)"],
            "Runtime (ms)": [f"{dp_time:.4f}", f"{gr_time:.4f}"],
            "Priority Yield": [dp_yield, gr_yield],
            "Tasks Placed": [len(dp_placed), len(gr_placed)]
        }
        st.table(pd.DataFrame(metrics_data))
        
        # Placement Validation summary
        st.write("### Server Placement Validation")
        st.write(f"**Optimal DP Schedule:** `{', '.join(dp_placed)}`")
        st.write(f"**Greedy Schedule:** `{', '.join(gr_placed)}`")
        
    st.divider()
    
    st.subheader("3. Recommended Routing Policy (Dijkstra SSSP)")
    gateway = 0
    start_time = time.perf_counter()
    routes = CloudAlgorithms.dijkstra(adj_list, gateway)
    route_time = (time.perf_counter() - start_time) * 1000
    st.write(f"Shortest path latencies from Gateway Node {gateway} to all other server nodes (Computed in {route_time:.4f} ms):")
    st.json(routes)

    st.divider()
    st.subheader("4. Global Latency Matrix (Floyd-Warshall APSP)")
    st.write("Calculates the All-Pairs Shortest Path matrix for instant routing between any two servers.")
    
    start_time = time.perf_counter()
    apsp_matrix = CloudAlgorithms.floyd_warshall(V, adj_list)
    apsp_time = (time.perf_counter() - start_time) * 1000
    
    st.write(f"*(Computed in {apsp_time:.4f} ms)*")
    # Display the matrix for the first 5 nodes to keep the UI clean
    st.dataframe(pd.DataFrame(apsp_matrix).head(5).iloc[:, :5])