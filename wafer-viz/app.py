"""
Wafer composite visualization + defect classification app for the WM-811K
wafer defect map dataset.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

import label_store as ls
import model as M
from data_loader import failure_class_order, load_dataset

WAFER_CMAP = ListedColormap(["white", "#b0bec5", "#e53935"])  # bg, good die, defect

DB_PATH = "data/labels.db"
MODELS_DIR = Path("models")
UPLOADS_DIR = Path("data") / "uploads"
EMBEDDINGS_PATH = Path("data") / "embeddings.npy"
EMBEDDINGS_INDEX_PATH = Path("data") / "embeddings_index.json"
UPLOAD_ID_BASE = 10_000_000

st.set_page_config(page_title="Wafer Map Explorer", layout="wide")


@st.cache_data(show_spinner="Loading wafer dataset...")
def cached_load(path: str) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_resource
def get_conn():
    return ls.connect(DB_PATH)


def get_wafer_map(wafer_id: int, df: pd.DataFrame) -> np.ndarray:
    """Resolve a wafer_id to its raw map array, whether it's a dataset row
    (wafer_id = positional index into df) or an uploaded file (wafer_id >= UPLOAD_ID_BASE)."""
    if wafer_id < UPLOAD_ID_BASE:
        return df.at[wafer_id, "waferMap"]
    return np.load(UPLOADS_DIR / f"{wafer_id}.npy")


def next_upload_id() -> int:
    existing = [int(p.stem) for p in UPLOADS_DIR.glob("*.npy")] if UPLOADS_DIR.exists() else []
    return max(existing, default=UPLOAD_ID_BASE - 1) + 1


def load_production_info() -> dict | None:
    path = MODELS_DIR / "production.json"
    return json.loads(path.read_text()) if path.exists() else None


def maybe_trigger_retrain(conn) -> bool:
    """Called right after a human verifies a label (Classify's Verify button,
    Reevaluation's Confirm button) -- retraining happens automatically as a
    natural consequence of new verified ground truth existing, no manual
    "Retrain" button. Still produces only a *candidate*; promoting it to
    production stays an explicit human action (see tab_model), so a model
    can never start actually being used without someone reviewing it first.

    No-ops if a training run is already in flight, or if nothing new has
    been verified since the last run was triggered -- so rapid-fire
    confirmations in Defect Reevaluation coalesce into one run rather than
    spawning a pile of concurrent trainings.

    The lock stores the PID of the training subprocess (not just a bare
    marker file) and is checked for liveness with os.kill(pid, 0): a lock
    left behind by a crashed/killed process self-heals on the next trigger
    instead of permanently blocking retraining, while a genuinely live
    process is never double-triggered."""
    train_lock = MODELS_DIR / "train.lock"
    if train_lock.exists():
        pid_text = train_lock.read_text().strip()
        alive = True  # conservative default: an unrecognized lock format is treated as live
        if pid_text.isdigit():
            try:
                os.kill(int(pid_text), 0)
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
            else:
                alive = True
        if alive:
            return False

    state_path = MODELS_DIR / "train_trigger_state.json"
    last_count = json.loads(state_path.read_text())["last_verified_count"] if state_path.exists() else 0
    current_count = ls.verified_count(conn)
    if current_count <= last_count:
        return False

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({"last_verified_count": current_count}))
    proc = subprocess.Popen(
        "python train.py; rm -f models/train.lock",
        shell=True, cwd=str(Path(__file__).parent),
    )
    train_lock.write_text(str(proc.pid))
    return True


@st.cache_resource
def _load_embeddings_at(mtime: float):
    """Cache key includes the file's mtime, so a background subprocess (promote.py)
    or append_embedding writing a new version is picked up automatically -- no
    manual cache invalidation needed anywhere else."""
    return np.load(EMBEDDINGS_PATH), json.loads(EMBEDDINGS_INDEX_PATH.read_text())


def load_embeddings():
    """Returns (matrix, wafer_id -> row index) or (None, {}) if nothing's been computed yet."""
    if not EMBEDDINGS_PATH.exists() or not EMBEDDINGS_INDEX_PATH.exists():
        return None, {}
    return _load_embeddings_at(EMBEDDINGS_PATH.stat().st_mtime)


@st.cache_resource
def _load_embeddings_normalized_at(mtime: float):
    matrix, index = _load_embeddings_at(mtime)
    norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return norm, index


def load_embeddings_normalized():
    if not EMBEDDINGS_PATH.exists() or not EMBEDDINGS_INDEX_PATH.exists():
        return None, {}
    return _load_embeddings_normalized_at(EMBEDDINGS_PATH.stat().st_mtime)


def append_embedding(wafer_id: int, vector: np.ndarray) -> None:
    matrix, index = load_embeddings()
    if matrix is None:
        matrix = np.zeros((0, len(vector)), dtype=np.float32)
        index = {}
    key = str(wafer_id)
    vector = vector.astype(np.float32)
    if key in index:
        matrix[index[key]] = vector
    else:
        index[key] = matrix.shape[0]
        matrix = np.vstack([matrix, vector])
    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, matrix)
    EMBEDDINGS_INDEX_PATH.write_text(json.dumps(index))


def render_wafer(ax, wafer_map: np.ndarray, title: str = ""):
    ax.imshow(wafer_map, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def sidebar_data_source() -> str:
    st.sidebar.header("Dataset")
    default_path = str(Path(__file__).parent / "data" / "LSWMD.pkl")
    path = st.sidebar.text_input("Path to LSWMD.pkl", value=default_path)
    if not Path(path).exists():
        st.sidebar.warning(
            "File not found. Download LSWMD.pkl from the Kaggle 'WM-811K wafer map' "
            "dataset and place it at this path (or point to wherever you saved it)."
        )
        st.stop()
    return path


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    classes = failure_class_order(df)
    selected_classes = st.sidebar.multiselect(
        "Failure type", classes, default=classes
    )

    split_options = sorted(df["trainTestLabel"].unique())
    selected_splits = st.sidebar.multiselect(
        "Train/Test label", split_options, default=split_options
    )

    lots = sorted(df["lotName"].dropna().astype(str).unique())
    selected_lots = st.sidebar.multiselect("Lot (optional)", lots, default=[])

    die_min, die_max = float(df["dieSize"].min()), float(df["dieSize"].max())
    if die_min < die_max:
        die_range = st.sidebar.slider(
            "Die size", min_value=die_min, max_value=die_max, value=(die_min, die_max)
        )
    else:
        die_range = (die_min, die_max)

    filtered = df[
        df["failureType"].isin(selected_classes)
        & df["trainTestLabel"].isin(selected_splits)
        & df["dieSize"].between(die_range[0], die_range[1])
    ]
    if selected_lots:
        filtered = filtered[filtered["lotName"].astype(str).isin(selected_lots)]

    st.sidebar.caption(f"{len(filtered):,} / {len(df):,} wafers match filters")
    return filtered


def tab_overview(df: pd.DataFrame, filtered: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total wafers", f"{len(df):,}")
    col2.metric("Matching filters", f"{len(filtered):,}")
    col3.metric("Labeled", f"{(df['failureType'] != 'unlabeled').sum():,}")
    col4.metric("Failure classes", f"{df['failureType'].nunique():,}")

    st.subheader("Failure type distribution")
    counts = filtered["failureType"].value_counts().reindex(failure_class_order(filtered)).fillna(0)
    st.bar_chart(counts)

    with st.expander("Die size distribution"):
        st.bar_chart(filtered["dieSize"].value_counts().sort_index())
    with st.expander("Defect ratio distribution"):
        st.bar_chart(np.histogram(filtered["defectRatio"], bins=20)[0])

    st.subheader("Raw data")
    if filtered.empty:
        st.info("No wafers match the current filters.")
    else:
        display_cols = [c for c in filtered.columns if c != "waferMap"]
        max_rows = min(len(filtered), 5000)
        n_rows = st.slider("Rows to display", 1, max_rows, min(500, max_rows))
        st.dataframe(filtered[display_cols].head(n_rows), width="stretch")


def tab_composite(filtered: pd.DataFrame):
    st.subheader("Composite montage")
    classes = failure_class_order(filtered)
    if not classes:
        st.info("No wafers match the current filters.")
        return

    per_class = st.slider("Samples per class", min_value=1, max_value=20, value=8)
    sort_by = st.selectbox(
        "Sort samples within class by", ["random", "defectRatio", "dieSize"]
    )

    for cls in classes:
        subset = filtered[filtered["failureType"] == cls]
        if sort_by == "random":
            subset = subset.sample(n=min(per_class, len(subset)), random_state=0)
        else:
            subset = subset.sort_values(sort_by, ascending=False).head(per_class)

        st.markdown(f"**{cls}**  ({len(filtered[filtered['failureType'] == cls]):,} total)")
        fig, axes = plt.subplots(1, len(subset), figsize=(1.4 * len(subset), 1.6))
        if len(subset) == 1:
            axes = [axes]
        for ax, (_, row) in zip(axes, subset.iterrows()):
            render_wafer(ax, row["waferMap"])
        st.pyplot(fig)
        plt.close(fig)


def tab_explorer(filtered: pd.DataFrame):
    st.subheader("Single wafer explorer")
    if filtered.empty:
        st.info("No wafers match the current filters.")
        return

    max_idx = len(filtered) - 1
    idx = st.number_input("Row index within filtered set", 0, max_idx, 0)
    row = filtered.iloc[int(idx)]

    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(4, 4))
        render_wafer(ax, row["waferMap"])
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        st.write(
            {
                "failureType": row["failureType"],
                "trainTestLabel": row["trainTestLabel"],
                "lotName": row["lotName"],
                "waferIndex": row["waferIndex"],
                "dieSize": row["dieSize"],
                "mapShape": row["mapShape"],
                "dieCount": row["dieCount"],
                "defectCount": row["defectCount"],
                "defectRatio": round(row["defectRatio"], 4),
            }
        )

    same_lot = filtered[filtered["lotName"] == row["lotName"]]
    if len(same_lot) > 1:
        st.markdown(f"**Other wafers in lot `{row['lotName']}`** ({len(same_lot)} total)")
        same_lot = same_lot.sort_values("waferIndex").head(25)
        fig, axes = plt.subplots(1, len(same_lot), figsize=(1.4 * len(same_lot), 1.6))
        if len(same_lot) == 1:
            axes = [axes]
        for ax, (_, r) in zip(axes, same_lot.iterrows()):
            render_wafer(ax, r["waferMap"], title=str(r["waferIndex"]))
        st.pyplot(fig)
        plt.close(fig)


def tab_classify(df: pd.DataFrame, filtered: pd.DataFrame, conn):
    st.subheader("Classify a wafer")
    st.caption(
        "Pick a wafer (or upload a new one), see the model's prediction, then "
        "confirm it or refine it into a more specific in-house subtype."
    )

    verified_by = st.text_input("Verified by (optional)", key="verified_by_name")

    source = st.radio(
        "Wafer source", ["Pick from dataset", "Upload a wafer map (.npy)"], horizontal=True
    )

    wafer_id = None
    wafer_map = None

    if source == "Pick from dataset":
        if filtered.empty:
            st.info("No wafers match the current filters.")
            return
        max_idx = len(filtered) - 1
        idx = st.number_input("Row index within filtered set", 0, max_idx, 0, key="classify_row_idx")
        row = filtered.iloc[int(idx)]
        wafer_id = int(row.name)
        wafer_map = row["waferMap"]
    else:
        uploaded = st.file_uploader("Wafer map (.npy, 2D array of 0/1/2)", type=["npy"])
        if uploaded is None:
            st.info("Upload a .npy file to classify it.")
            return
        try:
            arr = np.load(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that file as a numpy array: {e}")
            return
        if arr.ndim != 2:
            st.error(f"Expected a 2D array, got shape {arr.shape}.")
            return
        if not np.all(np.isin(arr, [0, 1, 2])):
            st.error("Wafer map values must all be 0 (background), 1 (die), or 2 (defect).")
            return
        wafer_map = arr.astype(np.uint8)

        st.write(f"Preview — shape {wafer_map.shape}")
        fig, ax = plt.subplots(figsize=(3, 3))
        render_wafer(ax, wafer_map)
        st.pyplot(fig)
        plt.close(fig)

        upload_key = f"upload_id::{uploaded.name}::{uploaded.size}"
        if upload_key not in st.session_state:
            if st.button("Save this upload"):
                UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                new_id = next_upload_id()
                np.save(UPLOADS_DIR / f"{new_id}.npy", wafer_map)
                st.session_state[upload_key] = new_id
                st.rerun()
            return
        wafer_id = st.session_state[upload_key]

    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        render_wafer(ax, wafer_map, title=f"wafer_id {wafer_id}")
        st.pyplot(fig)
        plt.close(fig)

    prod_info = load_production_info()
    predicted_name = None
    if prod_info is None:
        with c2:
            st.warning("No production model yet. Train and promote one in the Model tab first.")
    else:
        clf = M.Classifier(prod_info["checkpoint"])
        cached_label = ls.get_label(conn, wafer_id)
        if cached_label and cached_label["predicted_type_id"] is not None:
            predicted_name = ls.type_name(conn, cached_label["predicted_type_id"])
            predicted_conf = cached_label["predicted_confidence"]
        else:
            pred = clf.predict(wafer_map)
            predicted_name, predicted_conf = pred["class_name"], pred["confidence"]
            type_id = ls.get_or_create_type(conn, predicted_name, None)
            ls.upsert_prediction(conn, wafer_id, type_id, predicted_conf, prod_info["version"])
            append_embedding(wafer_id, clf.embed(wafer_map))

        with c2:
            st.metric("Model prediction", predicted_name, f"{predicted_conf:.1%} confidence")
            cam = clf.grad_cam(wafer_map)
            resized = M.preprocess(wafer_map).squeeze(0).numpy()
            fig2, ax2 = plt.subplots(figsize=(3, 3))
            ax2.imshow(resized, cmap="gray")
            ax2.imshow(cam, cmap="jet", alpha=0.5)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_title("Grad-CAM: what drove this prediction", fontsize=8)
            st.pyplot(fig2)
            plt.close(fig2)

    st.divider()
    st.markdown("**Confirm or refine the defect type**")

    taxonomy = ls.list_taxonomy(conn)
    umbrella_nodes = [t for t in taxonomy if t["parent_id"] is None]
    umbrella_names = [t["name"] for t in umbrella_nodes] + ["+ Create new umbrella type"]
    default_idx = umbrella_names.index(predicted_name) if predicted_name in umbrella_names else 0
    umbrella_choice = st.selectbox(
        "Umbrella defect type", umbrella_names, index=default_idx, key=f"umbrella_{wafer_id}"
    )

    umbrella_id = None
    if umbrella_choice == "+ Create new umbrella type":
        umbrella_choice = st.text_input("New umbrella type name", key=f"new_umbrella_{wafer_id}")
    else:
        umbrella_id = next(t["id"] for t in umbrella_nodes if t["name"] == umbrella_choice)

    subtype_choice = ""
    if umbrella_id is not None:
        subtype_nodes = [t for t in taxonomy if t["parent_id"] == umbrella_id]
        subtype_names = ["(none)"] + [t["name"] for t in subtype_nodes] + ["+ Create new subtype"]
        subtype_pick = st.selectbox(
            "Subtype (optional, in-house granularity)", subtype_names, key=f"subtype_{wafer_id}"
        )
        if subtype_pick == "+ Create new subtype":
            subtype_choice = st.text_input("New subtype name", key=f"new_subtype_{wafer_id}")
        elif subtype_pick != "(none)":
            subtype_choice = subtype_pick

    if st.button("Verify", type="primary", key=f"verify_{wafer_id}"):
        if not umbrella_choice:
            st.error("Pick or name an umbrella defect type first.")
        else:
            final_umbrella_id = umbrella_id or ls.get_or_create_type(conn, umbrella_choice, None)
            final_type_id = final_umbrella_id
            if subtype_choice:
                final_type_id = ls.get_or_create_type(conn, subtype_choice, final_umbrella_id)
            ls.verify_label(conn, wafer_id, final_type_id, verified_by=verified_by)
            triggered = maybe_trigger_retrain(conn)
            label = f"{umbrella_choice}" + (f" → {subtype_choice}" if subtype_choice else "")
            msg = f"Verified wafer {wafer_id} as {label}."
            if triggered:
                msg += " New verified data detected — retraining started automatically in the background (see Model tab)."
            st.success(msg)


def tab_reevaluation(df: pd.DataFrame, conn):
    st.subheader("Defect reevaluation")
    st.caption(
        "Pick a confirmed defect type (umbrella or in-house subtype) and find "
        "other wafers that likely belong to it, ranked by embedding similarity."
    )

    taxonomy = ls.list_taxonomy(conn)
    by_id = {t["id"]: t for t in taxonomy}

    def display_name(t):
        if t["parent_id"] is None:
            return t["name"]
        return f"{by_id[t['parent_id']]['name']} → {t['name']}"

    verified_by_type = {t["id"]: ls.verified_examples_for_type(conn, t["id"]) for t in taxonomy}
    candidate_nodes = [t for t in taxonomy if verified_by_type[t["id"]]]
    if not candidate_nodes:
        st.info("No verified examples yet. Verify at least one wafer in the Classify tab first.")
        return

    options = {display_name(t): t["id"] for t in candidate_nodes}
    choice = st.selectbox("Defect type", list(options.keys()))
    type_id = options[choice]

    matrix, index = load_embeddings_normalized()
    if matrix is None:
        st.warning(
            "No embeddings computed yet. Promote a model in the Model tab first "
            "— that generates embeddings for the whole dataset."
        )
        return

    verified_ids = verified_by_type[type_id]
    ref_rows = [index[str(wid)] for wid in verified_ids if str(wid) in index]
    if not ref_rows:
        st.warning(
            "The verified example(s) for this type don't have embeddings yet "
            "(likely added after the last promotion). Promote again to refresh embeddings."
        )
        return

    centroid = matrix[ref_rows].mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
    sims = matrix @ centroid

    reverse_index = {v: int(k) for k, v in index.items()}
    exclude = set(verified_ids)

    order = np.argsort(-sims)
    ranked = []
    for row in order:
        wid = reverse_index[row]
        if wid in exclude:
            continue
        ranked.append((wid, float(sims[row])))
        if len(ranked) >= 200:
            break

    if not ranked:
        st.info("No other candidates found.")
        return

    k = st.slider("Candidates to show", 1, min(50, len(ranked)), min(12, len(ranked)))

    cols = st.columns(4)
    for i, (wid, score) in enumerate(ranked[:k]):
        with cols[i % 4]:
            wafer_map = get_wafer_map(wid, df)
            fig, ax = plt.subplots(figsize=(2, 2))
            render_wafer(ax, wafer_map)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"id {wid} · sim {score:.3f}")
            if st.button(f"Confirm as {choice}", key=f"confirm_{type_id}_{wid}"):
                verified_by = st.session_state.get("verified_by_name", "")
                ls.verify_label(conn, wid, type_id, verified_by=verified_by)
                triggered = maybe_trigger_retrain(conn)
                msg = f"Confirmed wafer {wid}."
                if triggered:
                    msg += " Retraining started automatically in the background."
                st.success(msg)
                st.rerun()


def tab_model(conn):
    st.subheader("Model")
    st.caption(
        "Architecture, data prep, training, and evaluation follow MathWorks' "
        "\"Classify Anomalies on Wafer Defect Maps Using Deep Learning\" example."
    )

    st.markdown("### Architecture")
    st.markdown(
        "4 convolutional blocks (8 → 16 → 32 → 64 filters), each a 3×3 conv + "
        "batchnorm + ReLU followed by 2×2 max pooling, then dropout (0.3) and a "
        "linear classifier over the umbrella defect classes. Input is a single-channel "
        f"{M.INPUT_SIZE}×{M.INPUT_SIZE} map. Grad-CAM overlays use the last pre-pool "
        "convolutional activation, matching `gradCAM`'s default reduction-layer choice."
    )

    st.markdown("### Data preparation")
    st.markdown(
        "Wafer maps ({0,1,2}-valued: background/die/defect) are resized to "
        f"{M.INPUT_SIZE}×{M.INPUT_SIZE} and normalized to [0,1]. Training uses WM-811K's "
        "own Training/Test split as verified ground truth, plus any human-verified labels "
        "collected in the app. Classes other than the majority get 5 augmented copies each "
        "(horizontal flip, vertical flip, 90°-multiple rotation) — a fixed 6x multiplier, "
        "matching the article's `oversampleWaferDefectClasses`."
    )

    st.markdown("### Training")
    st.markdown(
        "Adam optimizer with weight decay, cross-entropy loss; the checkpoint with the "
        "lowest validation loss across training is kept. Evaluation mirrors the article's "
        "methodology: row-normalized confusion matrix, per-class precision/recall/F1 from "
        "the confusion matrix, and precision-recall curves with AUC swept over every "
        "distinct prediction score (matching MATLAB's `rocmetrics`, not a coarse fixed grid)."
    )

    st.divider()
    st.markdown("### Evaluation (production model)")

    prod_info = load_production_info()
    if prod_info is None:
        st.info("No production model yet.")
        return

    st.caption(f"v{prod_info['version']}")
    eval_path = Path(prod_info["eval"]) if "eval" in prod_info else None
    ev = json.loads(eval_path.read_text()) if eval_path and eval_path.exists() else None

    cols = st.columns(3)
    cols[0].metric("Trained on", f"{prod_info.get('n_verified', '—')} verified wafers")
    if ev is None:
        st.caption("No evaluation report available for this version.")
        return

    cols[1].metric("Test accuracy", f"{ev.get('accuracy', 0):.1%}")
    cols[2].metric("Test set size", ev.get("n_test", "—"))

    classes = ev["classes"]
    supports = {cls: m["support"] for cls, m in ev["per_class"].items()}
    zero_support = [cls for cls, s in supports.items() if s == 0]
    if zero_support:
        st.warning(
            f"No test examples for: {', '.join(zero_support)} — their confusion-matrix "
            "rows and P/R/F1 are correctly all-zero, not a rendering error. Verify more "
            "wafers of these types (or wait for a larger training run) for a meaningful read."
        )
    predicted_classes = {classes[c] for row in ev["confusion_matrix"] for c, v in enumerate(row) if v > 0}
    if len(predicted_classes) == 1 and ev["n_test"] > 1:
        st.warning(
            f"This model predicts \"{next(iter(predicted_classes))}\" for every single test "
            "wafer regardless of true class — it hasn't learned to discriminate yet (typical "
            "of a model trained on very little data / very few epochs), not a metrics bug."
        )

    cm = pd.DataFrame(ev["confusion_matrix_normalized"], index=classes, columns=classes)
    st.caption("Confusion matrix (row-normalized) — rows are true class, columns are predicted class")
    st.dataframe(cm.style.format("{:.2f}").background_gradient(cmap="Reds", axis=1), width="stretch")

    metrics_df = pd.DataFrame(
        {
            cls: {
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "pr_auc": m["pr_auc"],
            }
            for cls, m in ev["per_class"].items()
        }
    ).T
    st.caption("Per-class precision / recall / F1 / PR-AUC")
    st.dataframe(metrics_df.style.format("{:.2f}"), width="stretch")

    st.caption("Precision-recall curves for all classes")
    fig, ax = plt.subplots(figsize=(5, 4))
    for cls, m in ev["per_class"].items():
        ax.plot(m["pr_curve"]["recall"], m["pr_curve"]["precision"], label=cls, linewidth=1.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    st.pyplot(fig)
    plt.close(fig)

    gradcam_dir = MODELS_DIR / f"candidate_{prod_info['version']}_gradcam"
    if gradcam_dir.exists():
        images = sorted(gradcam_dir.glob("*.png"))[:8]
        if images:
            st.caption("Grad-CAM samples (what the model keys on)")
            img_cols = st.columns(4)
            for i, img_path in enumerate(images):
                img_cols[i % 4].image(str(img_path), caption=img_path.stem, width="stretch")


def main():
    st.title("Wafer Map Explorer — WM-811K")
    path = sidebar_data_source()
    df = cached_load(path)
    filtered = sidebar_filters(df)

    conn = get_conn()
    ls.seed_taxonomy(conn, failure_class_order(df))

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Overview",
            "Composite Dictionary",
            "Wafer Explorer",
            "Classify",
            "Defect Reevaluation",
            "Model",
        ]
    )
    with tab1:
        tab_overview(df, filtered)
    with tab2:
        tab_composite(filtered)
    with tab3:
        tab_explorer(filtered)
    with tab4:
        tab_classify(df, filtered, conn)
    with tab5:
        tab_reevaluation(df, conn)
    with tab6:
        tab_model(conn)


if __name__ == "__main__":
    main()
