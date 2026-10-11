"""
Cloud Task Placement: Streamlit Web Dashboard
Algorithms: MST (Kruskal), SSSP (Dijkstra), DP Knapsack, Greedy, Floyd-Warshall
Run with: streamlit run app.py
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import heapq
import random
import time
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict

# ==========================================
# PAGE CONFIG (must be first Streamlit call)
# ==========================================
st.set_page_config(
    page_title="Cloud Task Placement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# CUSTOM CSS - Premium Dark Theme
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29 0%, #121b3a 50%, #0f1e2e 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1a1a3e 0%, #0d1b2a 100%); border-right: 1px solid rgba(99,102,241,0.3); }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #818cf8 !important; font-weight: 700; }
h1 { background: linear-gradient(90deg, #818cf8, #38bdf8, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700 !important; font-size: 2.4rem !important; }
h2, h3 { color: #93c5fd !important; font-weight: 600 !important; }
[data-testid="metric-container"] { background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3); border-radius: 12px; padding: 12px !important; }
.stButton > button { background: linear-gradient(135deg, #6366f1, #38bdf8) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.6rem 1.6rem !important; font-weight: 600 !important; font-size: 1rem !important; width: 100%; }
[data-testid="stTabs"] [role="tab"] { background: rgba(255,255,255,0.04) !important; border-radius: 8px !important; color: #94a3b8 !important; font-size: 0.8rem !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: rgba(99,102,241,0.3) !important; color: #e2e8f0 !important; }
hr { border-color: rgba(99,102,241,0.25) !important; }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 1. CORE ALGORITHM ENGINE
# ==========================================
class CloudAlgorithms:

    @staticmethod
    def kruskal_mst(num_nodes: int, edges: List[Tuple[int, int, int]]) -> Tuple[List, int]:
        parent = list(range(num_nodes))

        def find(i):
            if parent[i] != i:
                parent[i] = find(parent[i])
            return parent[i]

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
                return True
            return False

        mst_edges, mst_cost = [], 0
        for latency, u, v in sorted(edges):
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
            if curr_dist > distances[u]:
                continue
            for v, weight in adj_list[u]:
                nd = curr_dist + weight
                if nd < distances[v]:
                    distances[v] = nd
                    heapq.heappush(pq, (nd, v))
        return distances

    @staticmethod
    def dp_knapsack(capacity: int, tasks: List[Tuple[str, int, int]]) -> Tuple[int, List[str]]:
        n = len(tasks)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]

        for i in range(1, n + 1):
            t_id, weight, val = tasks[i - 1]
            for w in range(capacity + 1):
                dp[i][w] = dp[i - 1][w]
                if weight <= w:
                    dp[i][w] = max(dp[i][w], val + dp[i - 1][w - weight])

        selected, w = [], capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i - 1][w]:
                t_id, weight, val = tasks[i - 1]
                selected.append(t_id)
                w -= weight
        return dp[n][capacity], selected

    @staticmethod
    def greedy_scheduler(capacity: int, tasks: List[Tuple[str, int, int]]) -> Tuple[int, List[str]]:
        sorted_tasks = sorted(tasks, key=lambda x: x[2] / max(x[1], 1), reverse=True)
        total_val, curr_weight, selected = 0, 0, []
        for t_id, weight, val in sorted_tasks:
            if curr_weight + weight <= capacity:
                selected.append(t_id)
                curr_weight += weight
                total_val += val
        return total_val, selected

    @staticmethod
    def floyd_warshall(V: int, adj_list: Dict) -> Tuple[List, List]:
        import copy
        INF = float('inf')
        dist = [[INF] * V for _ in range(V)]
        for i in range(V):
            dist[i][i] = 0
        for u in adj_list:
            for v, w in adj_list[u]:
                if 0 <= u < V and 0 <= v < V:
                    dist[u][v] = min(dist[u][v], w)
                    dist[v][u] = min(dist[v][u], w)

        matrices = [copy.deepcopy(dist)]
        for k in range(V):
            for i in range(V):
                for j in range(V):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
            matrices.append(copy.deepcopy(dist))
        return dist, matrices


# ==========================================
# 2. GRAPH DRAWING HELPERS
# ==========================================
def draw_graph_mst(G, pos, mst_edges, ax):
    mst_set = {(u, v) for u, v, _ in mst_edges} | {(v, u) for u, v, _ in mst_edges}
    normal_edges = [(u, v) for u, v in G.edges() if (u, v) not in mst_set]
    mst_ebunch = [(u, v) for u, v in G.edges() if (u, v) in mst_set]

    ax.set_facecolor("#0d1117")

    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, width=1.0,
                           edge_color="#334155", alpha=0.6, ax=ax, style="dashed")
    nx.draw_networkx_edges(G, pos, edgelist=mst_ebunch, width=2.8,
                           edge_color="#f472b6", alpha=0.95, ax=ax)

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_colors = [plt.cm.plasma(degrees[n] / max_deg) for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=420, node_color=node_colors,
                           linewidths=1.5, edgecolors="#818cf8", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="white",
                            font_weight="bold", ax=ax)

    mst_labels = {(u, v): G[u][v]['weight'] for u, v in mst_ebunch if G.has_edge(u, v)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=mst_labels, font_size=7,
                                 font_color="#fbbf24",
                                 bbox=dict(boxstyle="round,pad=0.2", fc="#1e293b",
                                           ec="none", alpha=0.8), ax=ax)

    patches = [
        mpatches.Patch(color="#f472b6", label="MST backbone"),
        mpatches.Patch(color="#334155", label="Other links"),
    ]
    ax.legend(handles=patches, loc="upper left", framealpha=0.3,
              labelcolor="white", facecolor="#1e293b", edgecolor="#334155", fontsize=8)
    ax.axis("off")


def draw_dijkstra_graph(G, pos, routes, gateway, ax):
    ax.set_facecolor("#0d1117")
    finite_vals = [v for v in routes.values() if v != float('inf')]
    max_dist = max(finite_vals) if finite_vals else 1
    node_colors = []
    for n in G.nodes():
        d = routes.get(n, float('inf'))
        if n == gateway:
            node_colors.append("#f59e0b")
        elif d == float('inf'):
            node_colors.append("#475569")
        else:
            node_colors.append(plt.cm.cool(1 - d / max_dist))

    nx.draw_networkx_edges(G, pos, width=1.0, edge_color="#334155", alpha=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=460, node_color=node_colors,
                           linewidths=1.5, edgecolors="#38bdf8", ax=ax)
    dist_labels = {n: f"{n}\n({routes.get(n, chr(8734))})" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=dist_labels, font_size=6.5,
                            font_color="white", ax=ax)
    ax.set_title(f"Dijkstra SSSP from Gateway {gateway}",
                 color="#93c5fd", fontsize=10, pad=8)
    ax.axis("off")


def render_matrix_heatmap(matrix, step_label, V):
    arr = np.array([[v if v != float('inf') else np.nan for v in row]
                    for row in matrix], dtype=float)
    size = max(4, min(V * 0.75 + 1.5, 10))
    fig, ax = plt.subplots(figsize=(size, size))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    vmin = float(np.nanmin(arr[arr != 0])) if np.any(arr != 0) else 0
    vmax = float(np.nanmax(arr)) if not np.all(np.isnan(arr)) else 1
    im = ax.imshow(arr, cmap="plasma", aspect="auto", vmin=vmin, vmax=vmax)

    for i in range(V):
        for j in range(V):
            val = matrix[i][j]
            txt = "\u221e" if val == float('inf') else str(int(val))
            cell_val = arr[i][j]
            color = "white" if np.isnan(cell_val) or (vmax > 0 and cell_val > vmax * 0.5) else "#1e293b"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color=color, fontweight="bold")

    ax.set_xticks(range(V))
    ax.set_yticks(range(V))
    ax.set_xticklabels([str(i) for i in range(V)], color="#94a3b8", fontsize=8)
    ax.set_yticklabels([str(i) for i in range(V)], color="#94a3b8", fontsize=8)
    ax.set_title(step_label, color="#93c5fd", fontsize=10, pad=10)

    for x in range(V + 1):
        ax.axhline(x - 0.5, color="#1e293b", linewidth=0.8)
        ax.axvline(x - 0.5, color="#1e293b", linewidth=0.8)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


# ==========================================
# 3. STREAMLIT UI
# ==========================================
st.markdown(
    "<h1 style='text-align:center; padding-bottom:0.2rem;'>Cloud Task Placement & Routing</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; color:#64748b; margin-top:0;'>"
    "DAA Project Prototype &nbsp;|&nbsp; "
    "<b>Algorithms:</b> MST (Kruskal) &nbsp;&middot;&nbsp; SSSP (Dijkstra) &nbsp;&middot;&nbsp; "
    "DP Knapsack &nbsp;&middot;&nbsp; Greedy &nbsp;&middot;&nbsp; Floyd-Warshall"
    "</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

with st.sidebar:
    st.markdown("## Simulation Parameters")
    st.markdown("---")
    V = st.slider("Servers (Nodes)", 3, 30, 8)
    max_edges = V * (V - 1) // 2
    E = st.slider("Links (Edges)", V - 1, max_edges, min(15, max_edges))
    TASKS = st.slider("Incoming VMs / Tasks", 3, 60, 20)
    source_node = st.slider("Gateway Node (Dijkstra src)", 0, V - 1, 0)
    st.markdown("---")
    run = st.button("Run Simulation", use_container_width=True)
    st.markdown("---")
    st.markdown("<small style='color:#475569;'>DAA Cloud Project 2025</small>", unsafe_allow_html=True)

if run:
    with st.spinner("Generating network topology..."):
        attempts = 0
        while True:
            G = nx.gnm_random_graph(V, E, seed=random.randint(0, 9999))
            attempts += 1
            if nx.is_connected(G) or attempts > 300:
                break
        if not nx.is_connected(G):
            for u, v in nx.minimum_spanning_edges(nx.complete_graph(V), data=False):
                G.add_edge(u, v)

    adj_list = {i: [] for i in range(V)}
    edge_list = []
    capacities = {i: random.choice([32, 64, 128]) for i in range(V)}
    tasks = [(f"VM-{i}", random.randint(2, 24), random.randint(5, 100)) for i in range(TASKS)]

    for u, v in G.edges():
        latency = random.randint(1, 20)
        adj_list[u].append((v, latency))
        adj_list[v].append((u, latency))
        edge_list.append((latency, u, v))
        G[u][v]['weight'] = latency

    pos = nx.spring_layout(G, seed=42, k=1.8 / (V ** 0.5))

    # KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Servers", V)
    kpi2.metric("Network Links", E)
    kpi3.metric("Tasks Queued", TASKS)
    kpi4.metric("Gateway Node", source_node)
    st.markdown("---")

    # =========================================================
    # SECTION 1 - MST + Scheduling
    # =========================================================
    st.markdown("## 1 . Network Topology & MST Backbone")
    col_graph, col_sched = st.columns([3, 2], gap="large")

    with col_graph:
        start = time.perf_counter()
        mst_edges, mst_cost = CloudAlgorithms.kruskal_mst(V, edge_list)
        mst_time = (time.perf_counter() - start) * 1000

        fig, ax = plt.subplots(figsize=(7, 5))
        draw_graph_mst(G, pos, mst_edges, ax)
        fig.patch.set_facecolor("#0d1117")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.success(
            f"Kruskal MST - Backbone cost: {mst_cost} ms | "
            f"Computed in {mst_time:.3f} ms | MST edges: {len(mst_edges)}"
        )

    with col_sched:
        st.markdown("### Server Scheduling")
        target_server = st.selectbox("Select target server", options=list(range(V)),
                                     index=min(1, V - 1), key="target_srv")
        capacity = capacities[target_server]
        st.info(f"Server {target_server} — CPU capacity: {capacity} cores")

        start = time.perf_counter()
        dp_yield, dp_placed = CloudAlgorithms.dp_knapsack(capacity, tasks)
        dp_time = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        gr_yield, gr_placed = CloudAlgorithms.greedy_scheduler(capacity, tasks)
        gr_time = (time.perf_counter() - start) * 1000

        metrics_df = pd.DataFrame({
            "Algorithm": ["DP Knapsack (Optimal)", "Greedy (Heuristic)"],
            "Runtime (ms)": [f"{dp_time:.4f}", f"{gr_time:.4f}"],
            "Priority Yield": [dp_yield, gr_yield],
            "Tasks Placed": [len(dp_placed), len(gr_placed)],
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        fig2, ax2 = plt.subplots(figsize=(5, 2.8))
        fig2.patch.set_facecolor("#0d1117")
        ax2.set_facecolor("#0d1117")
        bars = ax2.bar(["DP Knapsack\n(Optimal)", "Greedy\n(Heuristic)"],
                       [dp_yield, gr_yield], color=["#818cf8", "#34d399"],
                       width=0.45, edgecolor="none")
        ymax = max(dp_yield, gr_yield, 1)
        for bar, val in zip(bars, [dp_yield, gr_yield]):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + ymax * 0.02,
                     str(val), ha="center", va="bottom",
                     color="white", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Priority Yield", color="#94a3b8", fontsize=9)
        ax2.set_title("Algorithm Comparison", color="#93c5fd", fontsize=10)
        ax2.tick_params(colors="#94a3b8")
        for spine in ax2.spines.values():
            spine.set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

        with st.expander("Placed Task Lists"):
            c1, c2 = st.columns(2)
            c1.markdown("**DP (Optimal)**")
            c1.write(", ".join(dp_placed) if dp_placed else "_None_")
            c2.markdown("**Greedy**")
            c2.write(", ".join(gr_placed) if gr_placed else "_None_")

    st.markdown("---")

    # =========================================================
    # SECTION 2 - Dijkstra SSSP
    # =========================================================
    st.markdown("## 2 . Shortest-Path Routing (Dijkstra SSSP)")
    col_dijk, col_routes = st.columns([3, 2], gap="large")

    with col_dijk:
        start = time.perf_counter()
        routes = CloudAlgorithms.dijkstra(adj_list, source_node)
        dijk_time = (time.perf_counter() - start) * 1000

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        draw_dijkstra_graph(G, pos, routes, source_node, ax3)
        fig3.patch.set_facecolor("#0d1117")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        st.success(f"Dijkstra from Node {source_node} — Computed in {dijk_time:.3f} ms")

    with col_routes:
        st.markdown(f"### Distance Table from Node {source_node}")
        route_df = pd.DataFrame(
            [(node, dist if dist != float('inf') else "inf")
             for node, dist in sorted(routes.items())],
            columns=["Server Node", "Shortest Latency (ms)"],
        )
        st.dataframe(route_df, use_container_width=True, hide_index=True)

        finite_dists = [d for d in routes.values() if d != float('inf') and d > 0]
        if finite_dists:
            fig4, ax4 = plt.subplots(figsize=(5, 2.8))
            fig4.patch.set_facecolor("#0d1117")
            ax4.set_facecolor("#0d1117")
            ax4.hist(finite_dists, bins=min(10, len(finite_dists)),
                     color="#38bdf8", alpha=0.8, edgecolor="#0d1117")
            ax4.set_xlabel("Latency (ms)", color="#94a3b8", fontsize=9)
            ax4.set_ylabel("Nodes", color="#94a3b8", fontsize=9)
            ax4.set_title("Latency Distribution", color="#93c5fd", fontsize=10)
            ax4.tick_params(colors="#94a3b8")
            for spine in ax4.spines.values():
                spine.set_color("#334155")
            plt.tight_layout()
            st.pyplot(fig4, use_container_width=True)
            plt.close(fig4)

    st.markdown("---")

    # =========================================================
    # SECTION 3 - Floyd-Warshall
    # =========================================================
    st.markdown("## 3 . All-Pairs Shortest Paths (Floyd-Warshall)")
    MAX_V_FOR_MATRIX = 12

    if V <= MAX_V_FOR_MATRIX:
        with st.spinner("Computing Floyd-Warshall matrices..."):
            start = time.perf_counter()
            final_dist, fw_matrices = CloudAlgorithms.floyd_warshall(V, adj_list)
            fw_time = (time.perf_counter() - start) * 1000

        st.success(
            f"Floyd-Warshall — Computed {V + 1} matrices (D(0) to D({V})) in {fw_time:.3f} ms"
        )
        st.markdown(
            f"Step-by-step distance matrices **D(0)** (initial) through **D({V})** (optimal). "
            f"Each step k allows paths through intermediate nodes 0 to k-1."
        )

        tab_labels = [f"D({k})" for k in range(V + 1)]
        tabs = st.tabs(tab_labels)

        for k, tab in enumerate(tabs):
            with tab:
                if k == 0:
                    caption = "D(0) — Initial adjacency matrix (direct edges only, inf = no direct link)"
                else:
                    caption = f"D({k}) — Shortest paths allowing intermediate nodes 0 to {k - 1}"
                st.caption(caption)

                fig_m = render_matrix_heatmap(fw_matrices[k], f"D({k})", V)
                st.pyplot(fig_m, use_container_width=True)
                plt.close(fig_m)

                fmt = [['inf' if v == float('inf') else int(v) for v in row]
                       for row in fw_matrices[k]]
                df_mat = pd.DataFrame(fmt,
                                      index=[f"Node {i}" for i in range(V)],
                                      columns=[f"Node {j}" for j in range(V)])
                with st.expander("View raw matrix table"):
                    st.dataframe(df_mat, use_container_width=True)
    else:
        st.warning(
            f"Graph has V={V} nodes. Reduce servers to <= {MAX_V_FOR_MATRIX} "
            "to view step-by-step matrices. Showing final matrix only."
        )
        with st.spinner("Computing Floyd-Warshall final matrix..."):
            start = time.perf_counter()
            final_dist, fw_matrices = CloudAlgorithms.floyd_warshall(V, adj_list)
            fw_time = (time.perf_counter() - start) * 1000
        st.success(f"Computed in {fw_time:.3f} ms")
        fig_fin = render_matrix_heatmap(fw_matrices[-1], f"D({V}) - Final APSP Matrix", V)
        st.pyplot(fig_fin, use_container_width=True)
        plt.close(fig_fin)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
        "Built for DAA Project Based Learning | "
        "Algorithms: Kruskal | Dijkstra | DP Knapsack | Greedy | Floyd-Warshall"
        "</p>",
        unsafe_allow_html=True,
    )

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem;">
        <div style="font-size:5rem; margin-bottom:1rem;">&#9729;&#65039;</div>
        <h2 style="color:#93c5fd; margin-bottom:0.5rem;">Welcome to the Cloud Placement Dashboard</h2>
        <p style="color:#64748b; max-width:600px; margin:0 auto 2rem;">
            Configure your simulation parameters in the sidebar and click
            <b style="color:#818cf8;">Run Simulation</b> to visualize all algorithms.
        </p>
        <div style="display:flex; justify-content:center; gap:2rem; flex-wrap:wrap; margin-top:2rem;">
            <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); border-radius:12px; padding:1.2rem 1.8rem; min-width:160px;">
                <div style="font-size:2rem;">&#127760;</div>
                <div style="color:#93c5fd; font-weight:600; margin-top:0.5rem;">MST (Kruskal)</div>
                <div style="color:#64748b; font-size:0.8rem;">Network Backbone</div>
            </div>
            <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:1.2rem 1.8rem; min-width:160px;">
                <div style="font-size:2rem;">&#128506;</div>
                <div style="color:#38bdf8; font-weight:600; margin-top:0.5rem;">Dijkstra SSSP</div>
                <div style="color:#64748b; font-size:0.8rem;">Shortest Routing</div>
            </div>
            <div style="background:rgba(52,211,153,0.1); border:1px solid rgba(52,211,153,0.3); border-radius:12px; padding:1.2rem 1.8rem; min-width:160px;">
                <div style="font-size:2rem;">&#127890;</div>
                <div style="color:#34d399; font-weight:600; margin-top:0.5rem;">DP Knapsack</div>
                <div style="color:#64748b; font-size:0.8rem;">Optimal Scheduling</div>
            </div>
            <div style="background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3); border-radius:12px; padding:1.2rem 1.8rem; min-width:160px;">
                <div style="font-size:2rem;">&#9889;</div>
                <div style="color:#fbbf24; font-weight:600; margin-top:0.5rem;">Floyd-Warshall</div>
                <div style="color:#64748b; font-size:0.8rem;">All-Pairs SP</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
