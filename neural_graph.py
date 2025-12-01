from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set, Iterable, Union

import pandas as pd
import numpy as np


# ----------------------------------------------------------------------
# Graph data structure
# ----------------------------------------------------------------------

@dataclass
class NeuralGraph:
    """
    Graph with adjacency lists.
    Vertices are neuron names (strings).
    Edges are (target_neuron, weight, edge_type) tuples where edge_type is
    "CHEM" or "GAP".
    """
    adj: Dict[str, List[Tuple[str, float, str]]]

    def __init__(self) -> None:
        self.adj = defaultdict(list)

    def add_edge(
        self,
        u: str,
        v: str,
        weight: float = 1.0,
        edge_type: str = "",
        directed: bool = True,
    ) -> None:
        """
        Add an edge u -> v with given weight and edge_type.
        If directed=False, also add v -> u (for gap junctions).
        """
        self.adj[u].append((v, weight, edge_type))
        if not directed:
            self.adj[v].append((u, weight, edge_type))

    def vertices(self) -> List[str]:
        return list(self.adj.keys())

    def neighbors(self, u: str) -> List[Tuple[str, float, str]]:
        return self.adj.get(u, [])


# ----------------------------------------------------------------------
# Helpers to read adjacency matrices from Cook et al. Excel file
# ----------------------------------------------------------------------

def _extract_matrix_from_sheet(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the adjacency matrix block from one of the Cook et al. sheets.

    The hermaphrodite chemical and gap junction sheets have the same layout:
    - Row 2 (0-based index) from column 3 onward contains column labels.
    - Column 2 from row 3 onward contains row labels.
    - The numeric body starts at (row 3, col 3).
    """
    header_row = 2
    rowname_col = 2
    data_row_start = 3
    data_col_start = 3

    row_labels = raw.iloc[data_row_start:, rowname_col]
    col_labels = raw.iloc[header_row, data_col_start:]

    row_mask = row_labels.notna()
    col_mask = col_labels.notna()

    row_idx = np.where(row_mask)[0] + data_row_start
    col_idx = np.where(col_mask)[0] + data_col_start

    mat = pd.DataFrame(
        raw.values[np.ix_(row_idx, col_idx)],
        index=row_labels[row_mask].astype(str).values,
        columns=col_labels[col_mask].astype(str).values,
    )

    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return mat


def _is_probable_neuron(name: str) -> bool:
    """
    Heuristic filter: keep only names that look like neurons, i.e.,
    no lowercase alphabetic characters.

    This removes many muscles / end organs like 'vm1AL', 'vBWMR22',
    'exc_cell', 'pm3L', etc. which contain lowercase letters, while
    keeping canonical neuron names like 'AVAL', 'DA1', 'VC06', 'HSNL'.

    If you want to use your explicit neuron list from the description
    table, you can ignore this heuristic and pass a neuron_whitelist set
    to build_connectome_graph instead.
    """
    if not isinstance(name, str):
        return False
    return not any(ch.isalpha() and ch.islower() for ch in name)


def build_connectome_graph(
    excel_path: str,
    chem_sheet: str = "hermaphrodite chemical",
    gap_sheet: str = "hermaphrodite gap jn symmetric",
    min_weight: float = 0.0,
    neuron_whitelist: Optional[Set[str]] = None,
) -> NeuralGraph:
    """
    Build a graph from the hermaphrodite connectome adjacency matrices.

    Chemical synapses are treated as directed edges (pre -> post).
    Gap junctions are treated as undirected edges.
    Edge weights are taken directly from the matrices.

    Parameters
    ----------
    excel_path : str
        Path to "SI 5 Connectome adjacency matrices, corrected July 2020.xlsx".
    chem_sheet : str
        Sheet name for the hermaphrodite chemical adjacency matrix.
    gap_sheet : str
        Sheet name for the hermaphrodite gap junction adjacency matrix
        (use the *symmetric* one so that weights are already symmetrized).
    min_weight : float
        Ignore edges with weight <= min_weight.
    neuron_whitelist : set of str, optional
        If provided, only nodes whose names are in this set are kept.
        If None, a heuristic is used to discard obvious muscles / end organs.

    Returns
    -------
    NeuralGraph
    """
    # --- Load and extract matrices from sheets ------------------------
    xls = pd.ExcelFile(excel_path)

    chem_raw = pd.read_excel(xls, sheet_name=chem_sheet, header=None)
    chem_mat = _extract_matrix_from_sheet(chem_raw)

    gap_raw = pd.read_excel(xls, sheet_name=gap_sheet, header=None)
    gap_mat = _extract_matrix_from_sheet(gap_raw)

    # --- Decide which labels to keep (neurons only) -------------------
    all_labels = set(chem_mat.index) | set(chem_mat.columns) | \
                 set(gap_mat.index) | set(gap_mat.columns)

    if neuron_whitelist is not None:
        neurons = all_labels & set(neuron_whitelist)
    else:
        neurons = {name for name in all_labels if _is_probable_neuron(name)}

    # Restrict matrices to neuron-only rows / columns
    chem_mat = chem_mat.loc[
        chem_mat.index.isin(neurons), chem_mat.columns.isin(neurons)
    ]
    gap_mat = gap_mat.loc[
        gap_mat.index.isin(neurons), gap_mat.columns.isin(neurons)
    ]

    # --- Build the graph ----------------------------------------------
    G = NeuralGraph()

    # Chemical synapses: directed pre -> post (row -> column)
    for pre in chem_mat.index:
        row = chem_mat.loc[pre]
        for post, w in row.items():
            if w > min_weight:
                G.add_edge(
                    u=str(pre),
                    v=str(post),
                    weight=float(w),
                    edge_type="CHEM",
                    directed=True,
                )

    # Gap junctions: undirected, using upper triangle to avoid duplicates
    gap_labels = list(gap_mat.index)
    for i, u in enumerate(gap_labels):
        row = gap_mat.iloc[i]
        for j in range(i + 1, len(gap_labels)):
            v = gap_labels[j]
            w = row.iloc[j]
            if w > min_weight:
                G.add_edge(
                    u=str(u),
                    v=str(v),
                    weight=float(w),
                    edge_type="GAP",
                    directed=False,
                )

    return G


# ----------------------------------------------------------------------
# BFS (from one or multiple start neurons)
# ----------------------------------------------------------------------

def bfs_traversal(
    G: NeuralGraph,
    start_nodes: Union[str, Iterable[str]],
    allowed_types: Optional[Set[str]] = None,
) -> List[str]:
    """
    Breadth-First Search traversal from one or more starting neurons.

    Parameters
    ----------
    G : NeuralGraph
    start_nodes : str or iterable of str
        Single starting neuron or list of starting neurons (user-selected).
    allowed_types : set of str, optional
        If provided, only traverse edges whose edge_type is in allowed_types.
        For example: {"CHEM"}, {"GAP"}, or {"CHEM", "GAP"}.

    Returns
    -------
    order : list of str
        Neurons in the order they were visited.
    """
    if isinstance(start_nodes, str):
        starts = [start_nodes]
    else:
        starts = list(start_nodes)

    visited: Set[str] = set()
    order: List[str] = []
    queue: deque[str] = deque()

    # Initialize queue with all valid start nodes
    for s in starts:
        if s in G.adj and s not in visited:
            visited.add(s)
            queue.append(s)

    while queue:
        v = queue.popleft()
        order.append(v)
        for w, _wgt, etype in G.neighbors(v):
            if allowed_types is not None and etype not in allowed_types:
                continue
            if w not in visited:
                visited.add(w)
                queue.append(w)

    return order


def bfs_shortest_path(
    G: NeuralGraph,
    start: str,
    goal: str,
    allowed_types: Optional[Set[str]] = None,
) -> Optional[List[str]]:
    """
    Unweighted shortest path between two neurons using BFS.

    Parameters
    ----------
    G : NeuralGraph
    start : str
        Starting neuron.
    goal : str
        Target neuron.
    allowed_types : set of str, optional
        If provided, only traverse edges whose edge_type is in allowed_types.

    Returns
    -------
    path : list of str or None
        Neuron names from start to goal, or None if unreachable.
    """
    if start not in G.adj or goal not in G.adj:
        return None

    queue: deque[str] = deque()
    visited: Set[str] = set()
    prev: Dict[str, Optional[str]] = {}

    queue.append(start)
    visited.add(start)
    prev[start] = None

    while queue:
        v = queue.popleft()
        if v == goal:
            break

        for w, _wgt, etype in G.neighbors(v):
            if allowed_types is not None and etype not in allowed_types:
                continue
            if w not in visited:
                visited.add(w)
                prev[w] = v
                queue.append(w)

    if goal not in visited:
        return None

    # Reconstruct path
    path: List[str] = []
    cur: Optional[str] = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


# ----------------------------------------------------------------------
# DFS (from one or multiple start neurons)
# ----------------------------------------------------------------------

def dfs_traversal(
    G: NeuralGraph,
    start_nodes: Union[str, Iterable[str]],
    allowed_types: Optional[Set[str]] = None,
) -> List[str]:
    """
    Depth-First Search traversal from one or more starting neurons.

    Parameters
    ----------
    G : NeuralGraph
    start_nodes : str or iterable of str
        Single starting neuron or list of starting neurons (user-selected).
    allowed_types : set of str, optional
        If provided, only traverse edges whose edge_type is in allowed_types.

    Returns
    -------
    order : list of str
        Neurons in the order they were visited.
    """
    if isinstance(start_nodes, str):
        starts = [start_nodes]
    else:
        starts = list(start_nodes)

    visited: Set[str] = set()
    order: List[str] = []

    def dfs(v: str) -> None:
        visited.add(v)
        order.append(v)
        for w, _wgt, etype in G.neighbors(v):
            if allowed_types is not None and etype not in allowed_types:
                continue
            if w not in visited:
                dfs(w)

    for s in starts:
        if s in G.adj and s not in visited:
            dfs(s)

    return order