from pathlib import Path
import pandas as pd
import os
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------------------------------
# Paths
# -------------------------------------------------

PROJECT_ROOT = Path(file).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_DIR = DATA_DIR / "csv"
PROCESSED_DIR.mkdir(exist_ok=True)
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# -------------------------------------------------
# IO helpers
# -------------------------------------------------
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / name)

def save_parquet(df: pd.DataFrame, name: str):
    df.to_parquet(PROCESSED_DIR / name)

def load_parquet(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / name)

# -------------------------------------------------
# Normalisation helpers
# -------------------------------------------------
def normalise(series: pd.Series) -> pd.Series:
    """Strip + uppercase + string-cast."""
    return series.astype(str).str.strip().str.upper()

# -------------------------------------------------
# ISO cleaning helpers
# -------------------------------------------------
def clean_iso_index(df: pd.DataFrame) -> pd.DataFrame:
    df.index = df.index.astype(str).str.strip().str.upper()
    return df

def clean_iso_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df

def clean_iso_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype(str).str.strip().str.upper()
    return df

# -------------------------------------------------
# Pivot helpers
# -------------------------------------------------
def pivot_binary(df: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    """Generic binary pivot-table builder."""
    return (
        df.assign(value=1)
          .pivot_table(index=index, columns=columns, values="value", fill_value=0)
    )

# -------------------------------------------------
# Mineral helpers
# -------------------------------------------------
def mineral_columns(df: pd.DataFrame):
    """Return 3-letter uppercase mineral codes."""
    return [c for c in df.columns if c.isupper() and len(c) == 3]

# -------------------------------------------------
# Graph builder (Notebook 4)
# -------------------------------------------------
def build_and_save_graph(G: nx.DiGraph, name: str):
    """Unified graph builder for global + sector dependency graphs."""
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.4, iterations=50)

    class_colors = {
        "Integrated": "gold",
        "Producer Only": "steelblue",
        "Manufacturer Only": "firebrick",
        "Neither": "grey"
    }

    node_colors = [class_colors[G.nodes[n]["dependency_class"]] for n in G.nodes()]
    node_sizes = [800 * G.nodes[n]["capability"] for n in G.nodes()]
    edge_weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.4,
                           arrows=True, arrowstyle="-|>", arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(name)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.show()

    nx.write_gexf(G, PROCESSED_DIR / f"{name}.gexf")
    print(f"Saved: {name}")
