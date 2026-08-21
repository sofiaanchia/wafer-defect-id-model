"""
Promotes a trained candidate model to production, then regenerates
embeddings + predictions for the whole dataset using it.

Run as a background subprocess from the app's "Promote to production"
button (see app.py's Model tab) -- this is the ONLY code path that writes
models/production.json, and it only runs on an explicit human click.

    python promote.py [--candidate models/candidate.json] [--data data/LSWMD.pkl]
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import label_store as ls
import model as M
from data_loader import load_dataset

MODELS_DIR = Path("models")
LOG_PATH = MODELS_DIR / "promote.log"


def log(msg: str) -> None:
    line = f"{datetime.datetime.utcnow().isoformat()}  {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


class AllWafersDataset(Dataset):
    def __init__(self, df):
        self.wafer_ids = df.index.to_numpy()
        self.wafer_maps = df["waferMap"].to_numpy()

    def __len__(self):
        return len(self.wafer_ids)

    def __getitem__(self, i):
        return int(self.wafer_ids[i]), M.preprocess(self.wafer_maps[i])


def _collate(batch):
    ids = [b[0] for b in batch]
    x = torch.stack([b[1] for b in batch])
    return ids, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="models/candidate.json")
    ap.add_argument("--data", default="data/LSWMD.pkl")
    ap.add_argument("--db", default="data/labels.db")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None, help="cap dataset rows processed, for smoke testing")
    args = ap.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)

    candidate = json.loads(Path(args.candidate).read_text())
    log(f"Promoting candidate {candidate['version']} to production...")

    production = dict(candidate)
    production["promoted_at"] = datetime.datetime.utcnow().isoformat()
    (MODELS_DIR / "production.json").write_text(json.dumps(production, indent=2))
    log("production.json updated.")

    log("Loading dataset for batch embedding/prediction refresh...")
    df = load_dataset(args.data)
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.iloc[: args.limit]
    conn = ls.connect(args.db)
    taxonomy = {t["name"]: t["id"] for t in ls.list_taxonomy(conn) if t["parent_id"] is None}

    clf = M.Classifier(candidate["checkpoint"])
    dev, net, classes = clf.dev, clf.model, clf.classes

    ds = AllWafersDataset(df)
    loader = DataLoader(ds, batch_size=args.batch_size, collate_fn=_collate)

    embeddings = np.zeros((len(ds), 576), dtype=np.float32)
    index = {}
    pred_records = []

    done = 0
    with torch.no_grad():
        for wafer_ids, x in loader:
            x = x.to(dev)
            logits, emb = net(x, return_embedding=True)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            emb = emb.cpu().numpy()
            for j, wid in enumerate(wafer_ids):
                index[str(wid)] = done + j
                embeddings[done + j] = emb[j]
                idx = int(probs[j].argmax())
                type_id = taxonomy.get(classes[idx])
                if type_id is not None:
                    pred_records.append((wid, type_id, float(probs[j, idx]), candidate["version"]))
            done += len(wafer_ids)
            if done % (args.batch_size * 20) == 0:
                log(f"  processed {done}/{len(ds)} wafers")

    log(f"Writing {len(pred_records)} predictions (skips wafers already human-verified)...")
    ls.upsert_predictions_bulk(conn, pred_records)

    np.save("data/embeddings.npy", embeddings)
    Path("data/embeddings_index.json").write_text(json.dumps(index))
    log(f"DONE. Refreshed embeddings + predictions for {done} wafers.")


if __name__ == "__main__":
    main()
