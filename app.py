"""
Cloud Task Placement: Streamlit Web Dashboard
Algorithms: MST (Kruskal), SSSP (Dijkstra), DP Knapsack, Greedy, Floyd-Warshall
Run with: streamlit run app.py
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #0f172a 100%); color: #e2e8f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #111827 0%, #0b0f19 100%); border-right: 1px solid rgba(99,102,241,0.25); }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #818cf8 !important; font-weight: 700; }
h1 { background: linear-gradient(90deg, #818cf8, #38bdf8, #34d399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: 700 !important; font-size: 2.3rem !important; }
h2, h3 { color: #93c5fd !important; font-weight: 600 !important; }
[data-testid="metric-container"] { background: rgba(30,41,59,0.7); border: 1px solid rgba(99,102,241,0.3); border-radius: 12px; padding: 12px !important; }
.stButton > button { background: linear-gradient(135deg, #4f46e5, #0284c7) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 0.6rem 1.6rem !important; font-weight: 600 !important; font-size: 0.95rem !important; width: 100%; transition: all 0.2s ease; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(79,70,229,0.4); }
[data-testid="stTabs"] [role="tab"] { background: rgba(255,255,255,0.03) !important; border-radius: 8px !important; color: #94a3b8 !important; font-size: 0.85rem !important; font-weight: 500 !important; padding: 0.4rem 0.9rem !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { background: rgba(99,102,241,0.25) !important; color: #e2e8f0 !important; border: 1px solid rgba(99,102,241,0.4) !important; }
hr { border-color: rgba(99,102,241,0.2) !important; margin: 1.5rem 0 !important; }
.stat-box { background: rgba(30,41,59,0.6); border: 1px solid rgba(99,102,241,0.25); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; }
.stat-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
.stat-val { font-size: 1.3rem; font-weight: 700; color: #38bdf8; margin-top: 2px; }
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
def draw_graph_mst(G, pos, mst_edges, ax, selected_server=None):
    """Draws network topology highlighting MST backbone and selected server."""
    mst_set = {(u, v) for u, v, _ in mst_edges} | {(v, u) for u, v, _ in mst_edges}
    normal_edges = [(u, v) for u, v in G.edges() if (u, v) not in mst_set]
    mst_ebunch = [(u, v) for u, v in G.edges() if (u, v) in mst_set]

    ax.set_facecolor("#0b0f19")

    # Regular non-backbone links
    nx.draw_networkx_edges(
        G, pos, edgelist=normal_edges, width=1.1,
        edge_color="#334155", alpha=0.55, ax=ax, style="dashed"
    )
    # MST Backbone links (vibrant pink/magenta)
    nx.draw_networkx_edges(
        G, pos, edgelist=mst_ebunch, width=2.8,
        edge_color="#f472b6", alpha=0.95, ax=ax
    )

    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1
    node_cmap = LinearSegmentedColormap.from_list("mst_nodes", ["#1e293b", "#2563eb", "#6366f1"])

    # Draw nodes and crisp high-contrast labels
    for n in G.nodes():
        if n == selected_server:
            # Highlight selected server (Gold / Amber)
            nx.draw_networkx_nodes(
                G, pos, nodelist=[n], node_size=750,
                node_color="#f59e0b", linewidths=2.8,
                edgecolors="#fde047", ax=ax
            )
            nx.draw_networkx_labels(
                G, pos, labels={n: str(n)}, font_size=11,
                font_color="#0f172a", font_weight="bold", ax=ax
            )
        else:
            # Regular nodes with degree-based dark/royal blue tint
            c = node_cmap(degrees[n] / max_deg)
            nx.draw_networkx_nodes(
                G, pos, nodelist=[n], node_size=600,
                node_color=[c], linewidths=1.8,
                edgecolors="#818cf8", ax=ax
            )
            # High-contrast pure white text
            nx.draw_networkx_labels(
                G, pos, labels={n: str(n)}, font_size=10,
                font_color="#ffffff", font_weight="bold", ax=ax
            )

    # Edge labels for MST weights
    mst_labels = {(u, v): G[u][v]['weight'] for u, v in mst_ebunch if G.has_edge(u, v)}
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=mst_labels, font_size=7.5,
        font_color="#fbbf24",
        bbox=dict(boxstyle="round,pad=0.25", fc="#0f172a", ec="#334155", alpha=0.9),
        ax=ax
    )

    patches = [
        mpatches.Patch(color="#f472b6", label="MST Backbone"),
        mpatches.Patch(color="#334155", label="Redundant Link"),
    ]
    if selected_server is not None:
        patches.append(mpatches.Patch(color="#f59e0b", label=f"Selected (Server {selected_server})"))

    ax.legend(
        handles=patches, loc="upper left", framealpha=0.35,
        labelcolor="#e2e8f0", facecolor="#1e293b", edgecolor="#334155", fontsize=8
    )
    ax.axis("off")


def draw_dijkstra_graph(G, pos, routes, gateway, ax):
    """Draws shortest paths with high-contrast visible node numbers & latencies."""
    ax.set_facecolor("#0b0f19")
    finite_vals = [v for v in routes.values() if v != float('inf') and v > 0]
    max_dist = max(finite_vals) if finite_vals else 1

    nx.draw_networkx_edges(G, pos, width=1.1, edge_color="#334155", alpha=0.55, ax=ax)

    # Cohesive dark-to-blue colormap
    dist_cmap = LinearSegmentedColormap.from_list("dijk_nodes", ["#0284c7", "#2563eb", "#4338ca"])

    for n in G.nodes():
        d = routes.get(n, float('inf'))
        if n == gateway:
            # Gateway node: amber background with sharp dark navy text
            nx.draw_networkx_nodes(
                G, pos, nodelist=[n], node_size=780,
                node_color="#f59e0b", linewidths=2.5,
                edgecolors="#fde047", ax=ax
            )
            nx.draw_networkx_labels(
                G, pos, labels={n: f"{n}\nSRC"},
                font_size=8.5, font_color="#0f172a",
                font_weight="bold", ax=ax
            )
        elif d == float('inf'):
            # Unreachable
            nx.draw_networkx_nodes(
                G, pos, nodelist=[n], node_size=650,
                node_color="#334155", linewidths=1.5,
                edgecolors="#64748b", ax=ax
            )
            nx.draw_networkx_labels(
                G, pos, labels={n: f"{n}\n∞"},
                font_size=8.5, font_color="#cbd5e1",
                font_weight="bold", ax=ax
            )
        else:
            # Reachable: dark/royal blue node with crisp pure white text
            norm_d = min(1.0, d / max_dist)
            c = dist_cmap(norm_d)
            nx.draw_networkx_nodes(
                G, pos, nodelist=[n], node_size=680,
                node_color=[c], linewidths=1.8,
                edgecolors="#38bdf8", ax=ax
            )
            nx.draw_networkx_labels(
                G, pos, labels={n: f"{n}\n{d}ms"},
                font_size=8.0, font_color="#ffffff",
                font_weight="bold", ax=ax
            )

    ax.set_title(f"Dijkstra SSSP from Gateway Node {gateway}", color="#93c5fd", fontsize=11, pad=10)
    patches = [
        mpatches.Patch(color="#f59e0b", label=f"Gateway (Node {gateway})"),
        mpatches.Patch(color="#0284c7", label="Reachable Server"),
        mpatches.Patch(color="#334155", label="Unreachable"),
    ]
    ax.legend(
        handles=patches, loc="upper left", framealpha=0.35,
        labelcolor="#e2e8f0", facecolor="#1e293b", edgecolor="#334155", fontsize=8
    )
    ax.axis("off")


def render_matrix_heatmap(matrix, step_label, V):
    """
    Renders compact distance matrix using 2-3 cohesive colors and 100% visible text.
    Colors: Midnight Navy -> Royal Blue -> Soft Cyan.
    """
    arr = np.array([[v if v != float('inf') else np.nan for v in row]
                    for row in matrix], dtype=float)

    # Compact, well-proportioned figure size
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    fig.patch.set_facecolor("#0b0f19")
    ax.set_facecolor("#0b0f19")

    finite_vals = arr[~np.isnan(arr)]
    vmin = float(np.min(finite_vals)) if len(finite_vals) > 0 else 0
    vmax = float(np.max(finite_vals)) if len(finite_vals) > 0 else 1
    if vmax == vmin:
        vmax = vmin + 1

    # Exactly 3 harmonious colors: Midnight Navy -> Royal Blue -> Soft Sky Cyan
    matrix_cmap = LinearSegmentedColormap.from_list(
        "cloud_matrix",
        ["#0f172a", "#1d4ed8", "#38bdf8"],
        N=256
    )
    matrix_cmap.set_bad(color="#1e293b")

    im = ax.imshow(arr, cmap=matrix_cmap, aspect="equal", vmin=vmin, vmax=vmax)

    # Render cell text with guaranteed high contrast
    for i in range(V):
        for j in range(V):
            val = matrix[i][j]
            if val == float('inf'):
                txt = "∞"
                color = "#94a3b8"
            else:
                txt = str(int(val))
                norm_val = (val - vmin) / (vmax - vmin)
                r, g, b, _ = matrix_cmap(norm_val)
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                # Bright cells get bold dark navy text, dark cells get pure white text
                color = "#0a0f1d" if lum > 0.58 else "#ffffff"

            ax.text(
                j, i, txt, ha="center", va="center",
                fontsize=9.5 if V <= 8 else 8.0,
                color=color, fontweight="bold"
            )

    ax.set_xticks(range(V))
    ax.set_yticks(range(V))
    ax.set_xticklabels([f"S{i}" for i in range(V)], color="#94a3b8", fontsize=8.5)
    ax.set_yticklabels([f"S{i}" for i in range(V)], color="#94a3b8", fontsize=8.5)
    ax.set_title(step_label, color="#93c5fd", fontsize=11, pad=8, fontweight="bold")

    # Clean borders between cells
    for x in range(V + 1):
        ax.axhline(x - 0.5, color="#334155", linewidth=0.8)
        ax.axvline(x - 0.5, color="#334155", linewidth=0.8)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.outline.set_edgecolor("#334155")
    cbar.set_label("Latency (ms)", color="#94a3b8", fontsize=8)

    plt.tight_layout()
    return fig


# ==========================================
# 3. SIMULATION DATA GENERATOR
# ==========================================
def generate_simulation(V: int, E: int, tasks_count: int) -> dict:
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
    tasks = [(f"VM-{i}", random.randint(2, 24), random.randint(5, 100)) for i in range(tasks_count)]

    for u, v in G.edges():
        latency = random.randint(1, 20)
        adj_list[u].append((v, latency))
        adj_list[v].append((u, latency))
        edge_list.append((latency, u, v))
        G[u][v]['weight'] = latency

    pos = nx.spring_layout(G, seed=42, k=1.8 / (V ** 0.5))

    start = time.perf_counter()
    mst_edges, mst_cost = CloudAlgorithms.kruskal_mst(V, edge_list)
    mst_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    final_dist, fw_matrices = CloudAlgorithms.floyd_warshall(V, adj_list)
    fw_time = (time.perf_counter() - start) * 1000

    return {
        "V": V,
        "E": E,
        "tasks_count": tasks_count,
        "G": G,
        "pos": pos,
        "adj_list": adj_list,
        "edge_list": edge_list,
        "capacities": capacities,
        "tasks": tasks,
        "mst_edges": mst_edges,
        "mst_cost": mst_cost,
        "mst_time": mst_time,
        "final_dist": final_dist,
        "fw_matrices": fw_matrices,
        "fw_time": fw_time,
    }


# ==========================================
# 4. STREAMLIT APPLICATION STATE & UI
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

# Session state initialization
if "sim_data" not in st.session_state:
    st.session_state.sim_data = None
if "target_srv" not in st.session_state:
    st.session_state.target_srv = 0

with st.sidebar:
    st.markdown("## Simulation Parameters")
    st.markdown("---")
    V = st.slider("Servers (Nodes)", 3, 30, 6)
    max_edges = V * (V - 1) // 2
    E = st.slider("Links (Edges)", V - 1, max_edges, min(15, max_edges))
    TASKS = st.slider("Incoming VMs / Tasks", 3, 60, 40)
    source_node = st.slider("Gateway Node (Dijkstra src)", 0, V - 1, 0)
    st.markdown("---")
    run_btn = st.button("Run Simulation", use_container_width=True)
    st.markdown("---")
    st.markdown("<small style='color:#475569;'>DAA Cloud Project 2025</small>", unsafe_allow_html=True)

if run_btn:
    with st.spinner("Generating network topology and computing algorithms..."):
        st.session_state.sim_data = generate_simulation(V, E, TASKS)
        # Ensure target_srv is within bounds
        st.session_state.target_srv = min(st.session_state.target_srv, V - 1)

# Render Dashboard if simulation data exists
if st.session_state.sim_data is not None:
    sim = st.session_state.sim_data
    sim_V = sim["V"]
    sim_E = sim["E"]
    G = sim["G"]
    pos = sim["pos"]
    adj_list = sim["adj_list"]
    capacities = sim["capacities"]
    tasks = sim["tasks"]
    mst_edges = sim["mst_edges"]
    mst_cost = sim["mst_cost"]
    mst_time = sim["mst_time"]
    fw_matrices = sim["fw_matrices"]
    fw_time = sim["fw_time"]

    # Active source node (clamped to graph bounds)
    active_src = min(source_node, sim_V - 1)

    # Top KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Servers", sim_V)
    kpi2.metric("Network Links", sim_E)
    kpi3.metric("Tasks Queued", len(tasks))
    kpi4.metric("Gateway Node", active_src)
    st.markdown("---")

    # =========================================================
    # SECTION 1 - MST + Server Scheduling
    # =========================================================
    st.markdown("## 1 . Network Topology & MST Backbone")
    col_graph, col_sched = st.columns([3, 2], gap="large")

    # Ensure target server selection is safe
    current_target_idx = min(st.session_state.target_srv, sim_V - 1)

    with col_sched:
        st.markdown("### Server Scheduling")

        def on_server_change():
            st.session_state.target_srv = st.session_state.target_srv_select

        target_server = st.selectbox(
            "Select target server",
            options=list(range(sim_V)),
            index=current_target_idx,
            key="target_srv_select",
            on_change=on_server_change,
        )
        # Keep state synchronized
        st.session_state.target_srv = target_server
        capacity = capacities[target_server]
        st.info(f"Server {target_server} — CPU capacity: {capacity} cores")

        # Scheduling Calculations for the selected server
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

        # Bar chart comparison
        fig2, ax2 = plt.subplots(figsize=(5, 2.7))
        fig2.patch.set_facecolor("#0b0f19")
        ax2.set_facecolor("#0b0f19")
        bars = ax2.bar(
            ["DP Knapsack\n(Optimal)", "Greedy\n(Heuristic)"],
            [dp_yield, gr_yield],
            color=["#818cf8", "#34d399"],
            width=0.45, edgecolor="none"
        )
        ymax = max(dp_yield, gr_yield, 1)
        for bar, val in zip(bars, [dp_yield, gr_yield]):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.02,
                str(val), ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold"
            )
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

    with col_graph:
        fig1, ax1 = plt.subplots(figsize=(7, 5))
        # Highlights the currently selected server with amber ring
        draw_graph_mst(G, pos, mst_edges, ax1, selected_server=target_server)
        fig1.patch.set_facecolor("#0b0f19")
        plt.tight_layout()
        st.pyplot(fig1, use_container_width=True)
        plt.close(fig1)

        st.success(
            f"Kruskal MST - Backbone cost: {mst_cost} ms | "
            f"Computed in {mst_time:.3f} ms | MST edges: {len(mst_edges)}"
        )

    st.markdown("---")

    # =========================================================
    # SECTION 2 - Dijkstra SSSP
    # =========================================================
    st.markdown("## 2 . Shortest-Path Routing (Dijkstra SSSP)")
    col_dijk, col_routes = st.columns([3, 2], gap="large")

    start = time.perf_counter()
    routes = CloudAlgorithms.dijkstra(adj_list, active_src)
    dijk_time = (time.perf_counter() - start) * 1000

    with col_dijk:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        draw_dijkstra_graph(G, pos, routes, active_src, ax3)
        fig3.patch.set_facecolor("#0b0f19")
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

        st.success(f"Dijkstra from Gateway Node {active_src} — Computed in {dijk_time:.3f} ms")

    with col_routes:
        st.markdown(f"### Distance Table from Node {active_src}")
        route_df = pd.DataFrame(
            [(node, dist if dist != float('inf') else "∞")
             for node, dist in sorted(routes.items())],
            columns=["Server Node", "Shortest Latency (ms)"],
        )
        st.dataframe(route_df, use_container_width=True, hide_index=True)

        finite_dists = [d for d in routes.values() if d != float('inf') and d > 0]
        if finite_dists:
            fig4, ax4 = plt.subplots(figsize=(5, 2.7))
            fig4.patch.set_facecolor("#0b0f19")
            ax4.set_facecolor("#0b0f19")
            ax4.hist(
                finite_dists, bins=min(10, len(finite_dists)),
                color="#38bdf8", alpha=0.85, edgecolor="#0b0f19"
            )
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
    # SECTION 3 - Floyd-Warshall (APSP)
    # =========================================================
    st.markdown("## 3 . All-Pairs Shortest Paths (Floyd-Warshall)")
    MAX_V_FOR_MATRIX = 12

    if sim_V <= MAX_V_FOR_MATRIX:
        st.success(
            f"Floyd-Warshall — Computed {sim_V + 1} matrices (D(0) to D({sim_V})) in {fw_time:.3f} ms"
        )
        st.markdown(
            f"Step-by-step distance matrices **D(0)** through **D({sim_V})**. "
            f"Step $k$ allows intermediate routing through servers $0$ to $k-1$."
        )

        tab_labels = [f"D({k})" for k in range(sim_V + 1)]
        tabs = st.tabs(tab_labels)

        for k, tab in enumerate(tabs):
            with tab:
                # 2-column layout keeps matrix compact and neatly proportioned
                col_mat, col_info = st.columns([1.1, 1], gap="large")

                with col_mat:
                    fig_m = render_matrix_heatmap(fw_matrices[k], f"Matrix D({k})", sim_V)
                    st.pyplot(fig_m, use_container_width=True)
                    plt.close(fig_m)

                with col_info:
                    if k == 0:
                        st.markdown("#### Initial Adjacency Matrix D(0)")
                        st.caption("Displays direct server-to-server latencies. Symbol **∞** denotes no direct physical link.")
                    elif k == sim_V:
                        st.markdown(f"#### Final APSP Matrix D({sim_V})")
                        st.caption("Fully converged all-pairs shortest paths. Any intermediate server could be utilized.")
                    else:
                        st.markdown(f"#### Relaxation Step D({k})")
                        st.caption(f"Paths relaxed using intermediate node **Server {k - 1}** ($D[i][j] = \\min(D[i][j], D[i][{k-1}] + D[{k-1}][j])$).")

                    # Quick Stats for Step k
                    mat_k = fw_matrices[k]
                    finite_vals = [mat_k[i][j] for i in range(sim_V) for j in range(sim_V) if i != j and mat_k[i][j] != float('inf')]
                    total_pairs = sim_V * (sim_V - 1)
                    reachable_count = len(finite_vals)

                    stat_c1, stat_c2 = st.columns(2)
                    with stat_c1:
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-label">Reachable Server Pairs</div>
                            <div class="stat-val">{reachable_count} / {total_pairs}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with stat_c2:
                        min_lat = min(finite_vals) if finite_vals else 0
                        max_lat = max(finite_vals) if finite_vals else 0
                        st.markdown(f"""
                        <div class="stat-box">
                            <div class="stat-label">Latency Range</div>
                            <div class="stat-val">{min_lat} - {max_lat} ms</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Quick Pair Path Lookup
                    st.markdown("##### Pairwise Latency Query")
                    q_c1, q_c2 = st.columns(2)
                    with q_c1:
                        q_src = st.selectbox(f"From (S_src)", options=list(range(sim_V)), index=0, key=f"q_src_{k}")
                    with q_c2:
                        q_dst = st.selectbox(f"To (S_dst)", options=list(range(sim_V)), index=min(1, sim_V - 1), key=f"q_dst_{k}")

                    q_val = mat_k[q_src][q_dst]
                    if q_val == float('inf'):
                        st.warning(f"No route between Server {q_src} and Server {q_dst} at step D({k}).")
                    else:
                        st.info(f"Latency: **{int(q_val)} ms** from Server {q_src} to Server {q_dst}")

                    fmt = [['∞' if v == float('inf') else int(v) for v in row] for row in fw_matrices[k]]
                    df_mat = pd.DataFrame(fmt,
                                          index=[f"Server {i}" for i in range(sim_V)],
                                          columns=[f"Server {j}" for j in range(sim_V)])
                    with st.expander("View numeric data table"):
                        st.dataframe(df_mat, use_container_width=True)

    else:
        st.warning(
            f"Graph has V={sim_V} nodes. Reduce servers to <= {MAX_V_FOR_MATRIX} "
            "to view step-by-step matrix progression. Showing final matrix only."
        )
        col_mat, col_info = st.columns([1.1, 1], gap="large")
        with col_mat:
            fig_fin = render_matrix_heatmap(fw_matrices[-1], f"D({sim_V}) - Final APSP Matrix", sim_V)
            st.pyplot(fig_fin, use_container_width=True)
            plt.close(fig_fin)
        with col_info:
            st.info(f"APSP converged for all {sim_V} servers in {fw_time:.3f} ms.")

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#475569; font-size:0.8rem;'>"
        "Built for DAA Project Based Learning | "
        "Algorithms: Kruskal | Dijkstra | DP Knapsack | Greedy | Floyd-Warshall"
        "</p>",
        unsafe_allow_html=True,
    )

else:
    # Initial Welcome Splash Screen
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem 2rem;">
        <div style="font-size:4.5rem; margin-bottom:1rem;">☁️</div>
        <h2 style="color:#93c5fd; margin-bottom:0.5rem;">Welcome to the Cloud Placement Dashboard</h2>
        <p style="color:#94a3b8; max-width:620px; margin:0 auto 1.8rem; font-size:1.05rem;">
            Click below or use the sidebar controls to generate a cloud network simulation and visualize graph algorithms in real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        if st.button("🚀 Launch Simulation Now", key="hero_launch", use_container_width=True):
            with st.spinner("Generating topology and running algorithms..."):
                st.session_state.sim_data = generate_simulation(V, E, TASKS)
                st.rerun()

    st.markdown("""
    <div style="display:flex; justify-content:center; gap:1.8rem; flex-wrap:wrap; margin-top:2.5rem;">
        <div style="background:rgba(99,102,241,0.1); border:1px solid rgba(99,102,241,0.3); border-radius:12px; padding:1.2rem 1.8rem; min-width:160px; text-align:center;">
            <div style="font-size:2rem;">🌐</div>
            <div style="color:#93c5fd; font-weight:600; margin-top:0.5rem;">MST (Kruskal)</div>
            <div style="color:#64748b; font-size:0.8rem;">Network Backbone</div>
        </div>
        <div style="background:rgba(56,189,248,0.1); border:1px solid rgba(56,189,248,0.3); border-radius:12px; padding:1.2rem 1.8rem; min-width:160px; text-align:center;">
            <div style="font-size:2rem;">🗺️</div>
            <div style="color:#38bdf8; font-weight:600; margin-top:0.5rem;">Dijkstra SSSP</div>
            <div style="color:#64748b; font-size:0.8rem;">Shortest Routing</div>
        </div>
        <div style="background:rgba(52,211,153,0.1); border:1px solid rgba(52,211,153,0.3); border-radius:12px; padding:1.2rem 1.8rem; min-width:160px; text-align:center;">
            <div style="font-size:2rem;">📦</div>
            <div style="color:#34d399; font-weight:600; margin-top:0.5rem;">DP Knapsack</div>
            <div style="color:#64748b; font-size:0.8rem;">Optimal Scheduling</div>
        </div>
        <div style="background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3); border-radius:12px; padding:1.2rem 1.8rem; min-width:160px; text-align:center;">
            <div style="font-size:2rem;">⚡</div>
            <div style="color:#fbbf24; font-weight:600; margin-top:0.5rem;">Floyd-Warshall</div>
            <div style="color:#64748b; font-size:0.8rem;">All-Pairs SP</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
