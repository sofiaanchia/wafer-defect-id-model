"""
Load and normalize the WM-811K wafer map dataset (LSWMD.pkl).

The published pickle is a pandas DataFrame where several columns are stored
as nested numpy object arrays instead of plain scalars (a known quirk of the
original dataset release), e.g. failureType == array([['Center']], dtype=object)
instead of "Center". This module unwraps those and derives a few convenience
columns used throughout the app.
"""

import pickle

import numpy as np
import pandas as pd

FAILURE_CLASSES = [
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Random",
    "Scratch",
    "Near-full",
    "none",
]


def _unwrap(value):
    """Pull a scalar out of the dataset's nested-array cells, if needed."""
    arr = np.asarray(value)
    while arr.dtype == object and arr.size >= 1 and arr.ndim > 0:
        arr = np.asarray(arr.reshape(-1)[0])
    if arr.ndim == 0:
        return arr.item()
    return arr


def wafer_stats(wafer_map: np.ndarray) -> dict:
    """Derive die/defect counts from a raw wafer map (0=bg, 1=die, 2=defect)."""
    die_count = int((wafer_map > 0).sum())
    defect_count = int((wafer_map == 2).sum())
    return {
        "mapShape": wafer_map.shape,
        "dieCount": die_count,
        "defectCount": defect_count,
        "defectRatio": defect_count / die_count if die_count else 0.0,
    }


def load_dataset(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        df = pickle.load(f)

    df = df.copy()

    for col in ("failureType", "trainTestLabel", "dieSize", "lotName", "waferIndex"):
        if col in df.columns:
            df[col] = df[col].apply(_unwrap)

    df["failureType"] = df["failureType"].apply(
        lambda v: v if isinstance(v, str) and v else "unlabeled"
    )
    df["trainTestLabel"] = df["trainTestLabel"].apply(
        lambda v: v if isinstance(v, str) and v else "unlabeled"
    )

    df["waferMap"] = df["waferMap"].apply(np.asarray)
    stats = df["waferMap"].apply(wafer_stats).apply(pd.Series)
    df["mapShape"] = stats["mapShape"]
    df["dieCount"] = stats["dieCount"]
    df["defectCount"] = stats["defectCount"]
    df["defectRatio"] = stats["defectRatio"]

    df["dieSize"] = pd.to_numeric(df["dieSize"], errors="coerce")

    return df


def failure_class_order(df: pd.DataFrame):
    present = list(df["failureType"].unique())
    ordered = [c for c in FAILURE_CLASSES if c in present]
    ordered += sorted(c for c in present if c not in ordered)
    return ordered
