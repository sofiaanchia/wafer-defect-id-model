# Wafer Composite Defect Pattern Classifier: *Convolutional Neural Network (CNN) + UI*
### Developed by Sofia Anchia

A Streamlit app for exploring, visualizing, and classifying wafer defect maps from the [WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) dataset; with a human-in-the-loop labeling workflow that retrains and improves the underlying CNN as you verify predictions.

## Tabs + Features

- **Overview** — dataset-wide stats, failure type / die size / defect ratio distributions, and a filterable raw data table.
- **Composite Dictionary** — montage view of sample wafer maps grouped by failure type.
- **Wafer Explorer** — inspect a single wafer map and browse other wafers from the same lot.
- **Classify** — pick a wafer (or upload your own `.npy` map) to get a model prediction with a Grad-CAM overlay showing what drove its prediction, then confirm or refine it into a more specific in-house subtype(s).
- **Defect Reevaluation** — given a confirmed defect type, surface other wafers likely to belong to it via nearest-neighbor search over the model's embedding space.
- **Model** — architecture/training/evaluation summary for the current production model: confusion matrix, per-class precision/recall/F1, and PR curves with AUC.

## How the labeling loop works

1. Verifying a wafer's label in **Classify** or **Defect Reevaluation** confirms the classification as objective/ground truth.
2. New verified labels automatically trigger a background training run, producing a *candidate* model (this never touches the live model).
3. Promoting a candidate (an additional, new) defect composite to production is a separate, explicit action in the **Model** tab. This keeps a human in the loop before any newly trained model is actually used for predictions.

Predicted labels and human-verified labels are tracked separately (SQLite, via `label_store.py`), so an automated retrain can never overwrite a label a person has confirmed.

## Model development: CNN (Convolutional Neural Network)

Architecture, data preparation, training, and evaluation methodology follow MathWorks' *"Classify Anomalies on Wafer Defect Maps Using Deep Learning"* example:
https://www.mathworks.com/help/images/classify-anomalies-on-wafer-defect-maps-using-deep-learning.html

- 4 convolutional blocks (8 → 16 → 32 → 64 filters), each with batchnorm, ReLU, and max pooling, followed by dropout and a linear classifier over the umbrella defect classes.
- In-house subtypes aren't modeled as separate output classes (too few verified examples per defect subtype; instead they're found via nearest-neighbor search over the network's second to last layer embeddings.
  - ex. <umbrella type> Edge vs. <subtype> Edge - Inking)
- Evaluation follows the article's methodology:
  - row-normalized confusion matrix, per-class precision/recall/F1, and precision-recall curves with AUC covering over every distinct prediction score.

## Setup

```bash
git clone <repo-url>
cd wafer-viz
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download `LSWMD.pkl` from the [WM-811K wafer map dataset on Kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) and place it at `data/LSWMD.pkl` (or point the app at wherever you saved it).

## Run

```bash
streamlit run app.py
```

## Tech stack

Streamlit, PyTorch, pandas, NumPy, Matplotlib, SQLite.

