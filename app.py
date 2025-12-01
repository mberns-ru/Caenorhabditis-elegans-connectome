# app.py

import time
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
import pandas as pd

from neural_graph import (
    build_connectome_graph,
    bfs_traversal,
    bfs_shortest_path,
    dfs_traversal,
    NeuralGraph,
)

EXCEL_PATH = "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx"

st.set_page_config(page_title="C. elegans Connectome", layout="wide")


# ---------------------------------------------------------------------
# Cached loader
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_graph_and_positions(excel_path: str):
    """
    Load full hermaphrodite connectome (CHEM + GAP), compute layout,
    and time the graph construction step.
    """
    t0 = time.perf_counter()
    G = build_connectome_graph(
        excel_path=excel_path,
        chem_sheet="hermaphrodite chemical",
        gap_sheet="hermaphrodite gap jn symmetric",
        min_weight=0.0,
        neuron_whitelist=None,
    )
    build_elapsed = time.perf_counter() - t0  # seconds

    neurons = sorted(G.vertices())

    # Kamada–Kawai layout on an undirected version of the graph
    H = nx.Graph()
    for u in G.adj:
        H.add_node(u)
        for v, _w, _etype in G.neighbors(u):
            if u == v:
                continue
            if not H.has_edge(u, v):
                H.add_edge(u, v)

    pos_raw = nx.kamada_kawai_layout(H)
    pos = {n: (float(xy[0]), float(xy[1])) for n, xy in pos_raw.items()}

    return G, neurons, pos, build_elapsed


def count_edges_for_types(G: NeuralGraph, allowed_types: set[str]) -> int:
    """
    Count edges of the selected types in the same way they are drawn.

    - CHEM: directed, each edge counts once.
    - GAP: undirected, each pair counted once (u < v).
    """
    count = 0
    seen_undirected = set()

    for u in G.adj:
        for v, _w, etype in G.neighbors(u):
            if etype not in allowed_types:
                continue
            if etype == "GAP":
                key = tuple(sorted((u, v)))
                if key in seen_undirected:
                    continue
                seen_undirected.add(key)
                count += 1
            else:  # CHEM or other directed type
                count += 1
    return count


# ---------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------
def build_plotly_figure(
    G: NeuralGraph,
    pos: dict,
    allowed_types: set[str] | None,
    highlight_nodes: set[str] | None,
    start_node: str | None,
    target_node: str | None,
    bfs_path: list[str] | None,
) -> go.Figure:
    highlight_nodes = highlight_nodes or set()

    # --------------------- Base edges -------------------------------
    edge_x, edge_y = [], []
    for u in G.adj:
        if u not in pos:
            continue
        x0, y0 = pos[u]
        for v, _w, etype in G.neighbors(u):
            if allowed_types is not None and etype not in allowed_types:
                continue
            if v not in pos:
                continue
            # avoid double-plotting GAP edges
            if etype == "GAP" and u > v:
                continue
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=0.4, color="#BBBBBB"),
        hoverinfo="none",
        name="connections",
    )

    # --------------------- Base nodes -------------------------------
    node_x, node_y, node_color, node_size, node_text = [], [], [], [], []
    for n, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(n)

        if n == start_node:
            node_color.append("#2ECC71")  # green
            node_size.append(11)
        elif n == target_node:
            node_color.append("#F1C40F")  # yellow
            node_size.append(11)
        elif n in highlight_nodes:
            node_color.append("#E63946")  # red
            node_size.append(9)
        else:
            node_color.append("#1D3557")  # dark blue
            node_size.append(6)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        name="neurons",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=0.5, color="#FFFFFF"),
        ),
    )

    traces = [edge_trace, node_trace]

    # --------------------- Optional BFS-path highlighting ----------
    if bfs_path and len(bfs_path) > 1:
        # Highlight edges on the BFS path
        path_edge_x = []
        path_edge_y = []

        for u, v in zip(bfs_path[:-1], bfs_path[1:]):
            if u not in pos or v not in pos:
                continue
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            path_edge_x += [x0, x1, None]
            path_edge_y += [y0, y1, None]

        path_edge_trace = go.Scatter(
            x=path_edge_x,
            y=path_edge_y,
            line=dict(width=3, color="rgba(255,0,0,0.8)"),
            hoverinfo="none",
            mode="lines",
            name="BFS Path",
        )
        traces.append(path_edge_trace)

        # Highlight nodes on the path (bigger + different color)
        path_node_x = []
        path_node_y = []
        path_node_text = []
        for node in bfs_path:
            if node not in pos:
                continue
            x, y = pos[node]
            path_node_x.append(x)
            path_node_y.append(y)
            path_node_text.append(node)

        path_node_trace = go.Scatter(
            x=path_node_x,
            y=path_node_y,
            mode="markers+text",
            text=path_node_text,
            textposition="top center",
            hoverinfo="text",
            marker=dict(
                size=12,
                color="rgba(255,0,0,0.9)",
                line=dict(width=1, color="rgba(0,0,0,0.8)"),
            ),
            name="BFS Path Nodes",
        )
        traces.append(path_node_trace)

    fig = go.Figure(data=traces)
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(l=10, r=10, t=40, b=10),
        height=650,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    return fig


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "highlight_nodes" not in st.session_state:
    st.session_state["highlight_nodes"] = set()
if "start_node" not in st.session_state:
    st.session_state["start_node"] = None
if "target_node" not in st.session_state:
    st.session_state["target_node"] = None
if "bfs_path" not in st.session_state:
    st.session_state["bfs_path"] = None
if "prev_start" not in st.session_state:
    st.session_state["prev_start"] = None
if "prev_target" not in st.session_state:
    st.session_state["prev_target"] = None
if "prev_conn_choice" not in st.session_state:
    st.session_state["prev_conn_choice"] = None
if "dfs_info" not in st.session_state:
    st.session_state["dfs_info"] = None
if "bfs_info" not in st.session_state:
    st.session_state["bfs_info"] = None


# ---------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------
st.title("*C. elegans* Connectome")

G, neurons, pos, build_time = load_graph_and_positions(EXCEL_PATH)

# Big left rectangle = graph, right = controls
col_graph, col_ctrl = st.columns([4, 1])

with col_ctrl:
    st.subheader("Connection Type:")
    conn_choice = st.selectbox(
        label="",
        options=["Chemical synapses", "Gap junctions"],
        label_visibility="collapsed",
        key="conn_choice",
    )

    # Reset highlights when connection type changes
    if st.session_state["prev_conn_choice"] is None:
        st.session_state["prev_conn_choice"] = conn_choice
    elif conn_choice != st.session_state["prev_conn_choice"]:
        st.session_state["prev_conn_choice"] = conn_choice
        st.session_state["highlight_nodes"] = set()
        st.session_state["bfs_path"] = None
        st.session_state["dfs_info"] = None
        st.session_state["bfs_info"] = None

    if conn_choice == "Chemical synapses":
        allowed_types = {"CHEM"}
    else:
        allowed_types = {"GAP"}

    current_edge_count = count_edges_for_types(G, allowed_types)

    st.markdown("")
    st.caption(f"{len(neurons)} Nodes / {current_edge_count} Edges")
    st.caption(f"Graph build time: {build_time * 1000:.2f} ms")

    st.markdown("---")
    st.subheader("Start Neuron:")
    start_neuron = st.selectbox(
        label="Start neuron",
        options=neurons,
        label_visibility="collapsed",
        key="dfs_start",
    )

    # Auto-reset graph & update start-node color when start neuron changes
    if st.session_state["prev_start"] is None:
        st.session_state["prev_start"] = start_neuron
        st.session_state["start_node"] = start_neuron
    elif start_neuron != st.session_state["prev_start"]:
        st.session_state["prev_start"] = start_neuron
        st.session_state["start_node"] = start_neuron
        st.session_state["highlight_nodes"] = set()
        st.session_state["bfs_path"] = None
        st.session_state["dfs_info"] = None
        st.session_state["bfs_info"] = None

    if st.button("Run DFS"):
        t0 = time.perf_counter()
        dfs_order = dfs_traversal(G, start_neuron, allowed_types=allowed_types)
        elapsed = time.perf_counter() - t0

        st.session_state["highlight_nodes"] = set(dfs_order)
        st.session_state["start_node"] = start_neuron
        st.session_state["target_node"] = None
        st.session_state["bfs_path"] = None

        coverage = (len(dfs_order) / len(neurons)) * 100 if neurons else 0.0
        st.markdown(
            f"- **DFS visited:** {len(dfs_order)} neurons "
            f"({coverage:.1f}% of graph)  \n"
            f"- **Elapsed time:** {elapsed*1000:.2f} ms"
        )

        # Store detailed DFS info
        st.session_state["dfs_info"] = {
            "start": start_neuron,
            "elapsed": elapsed,
            "order": dfs_order,
            "coverage": coverage,
        }

    #st.markdown("---")
    st.subheader("Target Neuron:")
    target_neuron = st.selectbox(
        label="Target neuron",
        options=neurons,
        label_visibility="collapsed",
        key="bfs_target",
    )

    # Auto-reset graph & update target-node color when target neuron changes
    if st.session_state["prev_target"] is None:
        st.session_state["prev_target"] = target_neuron
        st.session_state["target_node"] = target_neuron
    elif target_neuron != st.session_state["prev_target"]:
        st.session_state["prev_target"] = target_neuron
        st.session_state["target_node"] = target_neuron
        st.session_state["highlight_nodes"] = set()
        st.session_state["bfs_path"] = None
        st.session_state["dfs_info"] = None
        st.session_state["bfs_info"] = None

    # italic "start -> target" line
    st.markdown(f"*{start_neuron} -> {target_neuron}*")

    if st.button("Run BFS"):
        t0 = time.perf_counter()
        path = bfs_shortest_path(
            G,
            start=start_neuron,
            goal=target_neuron,
            allowed_types=allowed_types,
        )
        elapsed = time.perf_counter() - t0

        if path is None:
            st.session_state["highlight_nodes"] = set()
            st.session_state["start_node"] = start_neuron
            st.session_state["target_node"] = target_neuron
            st.session_state["bfs_path"] = None
            st.warning(
                f"No path found from `{start_neuron}` to `{target_neuron}` "
                "with current connection type."
            )
            st.markdown(f"- **Elapsed time:** {elapsed*1000:.2f} ms")
            st.session_state["bfs_info"] = {
                "start": start_neuron,
                "target": target_neuron,
                "elapsed": elapsed,
                "path": None,
            }
        else:
            st.session_state["highlight_nodes"] = set(path)
            st.session_state["start_node"] = start_neuron
            st.session_state["target_node"] = target_neuron
            st.session_state["bfs_path"] = path

            st.success(
                f"Shortest path from `{start_neuron}` to `{target_neuron}` "
                f"has {len(path) - 1} edge(s)."
            )
            st.markdown(
                f"- **Nodes on path:** {len(path)}  \n"
                f"- **Elapsed time:** {elapsed*1000:.2f} ms"
            )

            st.session_state["bfs_info"] = {
                "start": start_neuron,
                "target": target_neuron,
                "elapsed": elapsed,
                "path": path,
            }

with col_graph:
    fig = build_plotly_figure(
        G=G,
        pos=pos,
        allowed_types=allowed_types,
        highlight_nodes=st.session_state["highlight_nodes"],
        start_node=st.session_state["start_node"],
        target_node=st.session_state["target_node"],
        bfs_path=st.session_state["bfs_path"],
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------
# Detailed search results section
# ---------------------------------------------------------------------
st.markdown("---")
st.subheader("Search results")

tab_dfs, tab_bfs = st.tabs(["DFS details", "BFS details"])

with tab_dfs:
    info = st.session_state.get("dfs_info")
    if not info:
        st.info("Run a DFS search to see detailed results.")
    else:
        st.markdown(
            f"**Start neuron:** `{info['start']}`  \n"
            f"**Visited neurons:** {len(info['order'])} "
            f"({info['coverage']:.1f}% of graph)  \n"
            f"**Elapsed time:** {info['elapsed']*1000:.2f} ms"
        )

        df_dfs = pd.DataFrame(
            {"Visit index": list(range(len(info["order"]))), "Neuron": info["order"]}
        )
        st.markdown("**DFS visit order**")
        st.dataframe(df_dfs, use_container_width=True, hide_index=True)

        # Show unvisited nodes if any
        unvisited = [n for n in neurons if n not in info["order"]]
        if unvisited:
            st.markdown(f"**Unvisited neurons ({len(unvisited)}):**")
            st.write(", ".join(unvisited))

with tab_bfs:
    info = st.session_state.get("bfs_info")
    if not info:
        st.info("Run a BFS search to see detailed results.")
    else:
        st.markdown(
            f"**Start neuron:** `{info['start']}`  \n"
            f"**Target neuron:** `{info['target']}`  \n"
            f"**Elapsed time:** {info['elapsed']*1000:.2f} ms"
        )

        if info["path"] is None:
            st.warning("No path found in the last BFS search.")
        else:
            path = info["path"]
            st.markdown(
                f"**Path length:** {len(path) - 1} edge(s)  \n"
                f"**Nodes on path:** {len(path)}"
            )

            df_bfs = pd.DataFrame(
                {"Step": list(range(len(path))), "Neuron": path}
            )
            st.markdown("**BFS shortest path (node sequence)**")
            st.dataframe(df_bfs, use_container_width=True, hide_index=True)

            # Also show as arrow-separated string
            st.markdown("**Path:**")
            st.write(" → ".join(path))
