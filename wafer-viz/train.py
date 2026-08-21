"""
Standalone training CLI for the wafer defect classifier. Not imported by
app.py at request time -- the app launches this as a background subprocess
automatically whenever a wafer gets newly verified (see
app.py:maybe_trigger_retrain), or you can run it directly:

    python train.py [--epochs 30] [--limit N] [--data data/LSWMD.pkl]

Trains ONLY on verified ground truth: the dataset's own Training/Test-split
labeled wafers, plus any human-verified rows in label_store's wafer_labels
table. Always produces a new *candidate* checkpoint + evaluation report
under models/ -- never touches models/production.json. Promoting a
candidate to production (and running the subsequent batch prediction /
embedding pass) is a separate, explicit human action taken in the app's
Model tab -- that gate stays manual even though triggering a *retrain* no
longer requires a click, so a model can never start actually being used
without a human reviewing its metrics first.

Evaluation methodology (confusion matrix, precision/recall/F1, PR curves +
AUC) mirrors MathWorks' "Classify Anomalies on Wafer Defect Maps Using Deep
Learning" example: https://www.mathworks.com/help/images/classify-anomalies-on-wafer-defect-maps-using-deep-learning.html
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import label_store as ls
import model as M
from data_loader import failure_class_order, load_dataset

MODELS_DIR = Path("models")


def log(msg: str) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.datetime.utcnow().isoformat()}  {msg}"
    print(line, flush=True)
    with open(MODELS_DIR / "train.log", "a") as f:
        f.write(line + "\n")


def build_ground_truth(df, conn) -> tuple[list[int], list[str]]:
    """Returns (wafer_ids, umbrella_class_names) for every verified example."""
    wafer_ids, labels = [], []

    # WM-811K's own labeled subset counts as pre-verified ground truth.
    dataset_labeled = df[df["trainTestLabel"].isin(["Training", "Test"])]
    for wafer_id, failure_type in dataset_labeled["failureType"].items():
        wafer_ids.append(int(wafer_id))
        labels.append(failure_type)

    # Human-verified rows from the label store, resolved up to their umbrella class.
    taxonomy = {t["id"]: t for t in ls.list_taxonomy(conn)}

    def umbrella_name(type_id: int) -> str:
        node = taxonomy[type_id]
        while node["parent_id"] is not None:
            node = taxonomy[node["parent_id"]]
        return node["name"]

    cur = conn.execute("SELECT wafer_id, verified_type_id FROM wafer_labels WHERE verified = 1")
    seen = set(wafer_ids)
    for wafer_id, verified_type_id in cur.fetchall():
        if wafer_id in seen:
            continue  # dataset ground truth for this wafer already included above
        wafer_ids.append(wafer_id)
        labels.append(umbrella_name(verified_type_id))

    return wafer_ids, labels


def stratified_split(labels: list[str], val_frac=0.05, test_frac=0.05, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_val = max(1, int(n * val_frac)) if n >= 3 else 0
        n_test = max(1, int(n * test_frac)) if n >= 3 else 0
        val_idx.extend(idx[:n_val])
        test_idx.extend(idx[n_val:n_val + n_test])
        train_idx.extend(idx[n_val + n_test:])
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def augment(wafer_map: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m = wafer_map
    if rng.random() < 0.5:
        m = np.fliplr(m)
    if rng.random() < 0.5:
        m = np.flipud(m)
    k = rng.integers(0, 4)
    if k:
        m = np.rot90(m, k=k)
    return m


class WaferDataset(Dataset):
    def __init__(self, wafer_maps, labels, class_to_idx, oversample=False, augment_data=False, seed=0):
        self.class_to_idx = class_to_idx
        self.augment_data = augment_data
        self.rng = np.random.default_rng(seed)

        if oversample:
            # Matches the MathWorks example's oversampleWaferDefectClasses: every
            # image NOT in the majority class gets exactly 5 augmented copies
            # appended (flip/flip/90-degree-rotation), a fixed 6x multiplier --
            # not a full rebalance to the majority count, which would repeat a
            # rare class's handful of source images hundreds of times over and
            # invite overfitting on near-duplicates.
            counts: dict = {}
            for lbl in labels:
                counts[lbl] = counts.get(lbl, 0) + 1
            majority_class = max(counts, key=counts.get)

            maps, lbls = list(wafer_maps), list(labels)
            for m, lbl in zip(wafer_maps, labels):
                if lbl == majority_class:
                    continue
                for _ in range(5):
                    maps.append(augment(m, self.rng))
                    lbls.append(lbl)
            self.wafer_maps, self.labels = maps, lbls
        else:
            self.wafer_maps, self.labels = list(wafer_maps), list(labels)

    def __len__(self):
        return len(self.wafer_maps)

    def __getitem__(self, i):
        m = self.wafer_maps[i]
        if self.augment_data:
            m = augment(m, self.rng)
        x = M.preprocess(m)
        y = self.class_to_idx[self.labels[i]]
        return x, y


def precision_recall_curve(is_pos: np.ndarray, scores: np.ndarray) -> tuple[list, list, float]:
    """Full-resolution PR curve: every distinct score is a threshold, matching
    what MATLAB's rocmetrics / sklearn's precision_recall_curve compute --
    not a coarse fixed grid, which under- or over-estimates AUC depending on
    how the (often very imbalanced) class's scores happen to fall between
    grid points."""
    order = np.argsort(-scores, kind="mergesort")
    is_pos_sorted = is_pos[order]
    tps = np.cumsum(is_pos_sorted)
    fps = np.cumsum(~is_pos_sorted)
    n_pos = int(is_pos_sorted.sum())

    precision = np.divide(tps, tps + fps, out=np.ones_like(tps, dtype=float), where=(tps + fps) > 0)
    recall = tps / n_pos if n_pos else np.zeros_like(tps, dtype=float)

    # as the threshold relaxes (more items included), recall rises 0 -> 1: already
    # ascending, so we only need to prepend the conventional (recall=0, precision=1) anchor
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    auc = float(np.trapezoid(precision, recall))
    return recall.tolist(), precision.tolist(), auc


def compute_metrics(y_true: np.ndarray, probs: np.ndarray, classes: list) -> dict:
    """Pure function: (true class indices, predicted probabilities, class names)
    -> the full evaluation report. Deliberately has no dependency on the model,
    dataloader, or torch, so it can be unit-tested against hand-computed
    ground truth independent of anything about training. Mirrors MathWorks'
    confusionmat / precision-recall-F1 / rocmetrics formulas exactly:
    https://www.mathworks.com/help/images/classify-anomalies-on-wafer-defect-maps-using-deep-learning.html
    """
    y_true = np.asarray(y_true)
    y_pred = probs.argmax(axis=1)
    n = len(classes)

    confusion = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        confusion[t, p] += 1
    row_sums = confusion.sum(axis=1, keepdims=True)
    confusion_norm = np.divide(confusion, row_sums, out=np.zeros_like(confusion, dtype=float), where=row_sums != 0)

    per_class = {}
    for i, cls in enumerate(classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        is_pos = (y_true == i)
        recalls, precisions, auc = precision_recall_curve(is_pos, probs[:, i])

        per_class[cls] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "pr_auc": auc,
            "pr_curve": {"recall": recalls, "precision": precisions},
            "support": int(is_pos.sum()),
        }

    return {
        "classes": classes,
        "confusion_matrix": confusion.tolist(),
        "confusion_matrix_normalized": confusion_norm.tolist(),
        "per_class": per_class,
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
        "n_test": int(len(y_true)),
    }


def evaluate(net, loader, dev, classes) -> dict:
    net.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev)
            logits = net(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_true.extend(y.numpy().tolist())
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.array(all_true)
    return compute_metrics(y_true, probs, classes)


def save_gradcam_gallery(net, classes, test_maps, test_labels, out_dir: Path, per_class=3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    dev = M.device()
    net.to(dev)

    ckpt_path = out_dir.parent / (out_dir.name + "_tmp.pt")
    M.save_checkpoint(str(ckpt_path), net, classes, version="tmp-for-gradcam")
    clf = M.Classifier(str(ckpt_path))
    ckpt_path.unlink(missing_ok=True)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    by_class = {}
    for m, lbl in zip(test_maps, test_labels):
        by_class.setdefault(lbl, []).append(m)

    for cls, maps in by_class.items():
        for i, wafer_map in enumerate(maps[:per_class]):
            pred = clf.predict(wafer_map)
            cam = clf.grad_cam(wafer_map, target_class=class_to_idx[cls])
            fig, axes = plt.subplots(1, 2, figsize=(4, 2))
            axes[0].imshow(wafer_map, cmap="gray", interpolation="nearest")
            axes[0].set_title("wafer", fontsize=7)
            axes[0].axis("off")
            axes[1].imshow(cam, cmap="jet")
            axes[1].set_title(f"true={cls} pred={pred['class_name']}", fontsize=6)
            axes[1].axis("off")
            fname = out_dir / f"{cls}_{i}_{'correct' if pred['class_name'] == cls else 'wrong'}.png"
            fig.savefig(fname, dpi=100, bbox_inches="tight")
            plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/LSWMD.pkl")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=None, help="cap ground-truth examples, for smoke testing")
    ap.add_argument("--db", default="data/labels.db")
    args = ap.parse_args()

    log("Loading dataset...")
    df = load_dataset(args.data)
    df = df.reset_index(drop=True)

    conn = ls.connect(args.db)
    ls.seed_taxonomy(conn, failure_class_order(df))

    log("Building ground-truth set from verified labels...")
    wafer_ids, labels = build_ground_truth(df, conn)
    if args.limit:
        wafer_ids, labels = wafer_ids[: args.limit], labels[: args.limit]
    classes = sorted(set(labels))
    log(f"Ground truth: {len(wafer_ids)} examples across {len(classes)} classes: {classes}")

    wafer_maps = [df.at[wid, "waferMap"] for wid in wafer_ids]

    train_i, val_i, test_i = stratified_split(labels)
    log(f"Split sizes -> train: {len(train_i)}, val: {len(val_i)}, test: {len(test_i)}")

    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_ds = WaferDataset(
        [wafer_maps[i] for i in train_i], [labels[i] for i in train_i],
        class_to_idx, oversample=True, augment_data=True,
    )
    val_ds = WaferDataset(
        [wafer_maps[i] for i in val_i], [labels[i] for i in val_i],
        class_to_idx, oversample=False, augment_data=False,
    )
    test_ds = WaferDataset(
        [wafer_maps[i] for i in test_i], [labels[i] for i in test_i],
        class_to_idx, oversample=False, augment_data=False,
    )

    dev = M.device()
    log(f"Training on device: {dev}")
    net = M.WaferCNN(num_classes=len(classes)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size) if len(test_ds) else None

    best_val_loss = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        net.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            logits = net(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / max(1, len(train_ds))

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(dev), y.to(dev)
                logits = net(x)
                val_loss += loss_fn(logits, y).item() * x.size(0)
        val_loss = val_loss / max(1, len(val_ds))

        log(f"epoch {epoch + 1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)

    version = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    ckpt_path = MODELS_DIR / f"candidate_{version}.pt"
    M.save_checkpoint(str(ckpt_path), net, classes, version)
    log(f"Saved candidate checkpoint: {ckpt_path}")

    eval_report = {"version": version, "trained_at": datetime.datetime.utcnow().isoformat(),
                    "n_train": len(train_ds), "n_val": len(val_ds)}
    if test_loader is not None:
        eval_report.update(evaluate(net, test_loader, dev, classes))
        log(f"Test accuracy: {eval_report.get('accuracy'):.4f}")
    else:
        log("No test split available (too few examples) -- skipping evaluation.")

    eval_path = MODELS_DIR / f"candidate_{version}_eval.json"
    eval_path.write_text(json.dumps(eval_report, indent=2))
    log(f"Saved evaluation report: {eval_path}")

    if len(test_i):
        gradcam_dir = MODELS_DIR / f"candidate_{version}_gradcam"
        save_gradcam_gallery(
            net, classes,
            [wafer_maps[i] for i in test_i], [labels[i] for i in test_i],
            gradcam_dir,
        )
        log(f"Saved Grad-CAM gallery: {gradcam_dir}")

    candidate_pointer = {
        "checkpoint": str(ckpt_path),
        "eval": str(eval_path),
        "version": version,
        "classes": classes,
        "n_verified": len(wafer_ids),
    }
    (MODELS_DIR / "candidate.json").write_text(json.dumps(candidate_pointer, indent=2))
    log("DONE. Candidate ready for review in the app's Model tab (not yet promoted to production).")


if __name__ == "__main__":
    main()
