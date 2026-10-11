# Smart City Digital Twin

A live digital twin that simulates and forecasts five coupled urban systems —
**traffic, air pollution, electricity, floods and emergency-vehicle routing** — with
deep learning (CNN-LSTM, Transformer), explainable AI (SHAP, Grad-CAM), risk tiers,
and cascade propagation between systems. Extends DeepCrowd (Jiang et al., IEEE TKDE 2021)
from single-domain crowd prediction to a multi-domain, explainable, action-oriented twin.

```
smart_city_twin/
├── sctwin/
│   ├── city.py       # road graph over Vellore coords, zones, sensors, hospitals, elevation
│   ├── datasets.py   # real-data loaders + coupled synthetic city simulator
│   ├── models.py     # Persistence, LSTM, CNN-LSTM, MM-STFormer (ours), Flood U-Net
│   ├── train.py      # scaler, training loop w/ early stopping, MAE/RMSE/MAPE
│   ├── xai.py        # SHAP feature attribution, Grad-CAM
│   └── twin.py       # risk tiers, cascade engine, emergency routing
├── scripts/run_all.py  # trains + evaluates everything -> results/
├── app.py              # live Streamlit + Folium dashboard
└── data/README.md      # where to put the real datasets
```

---

## Step-by-step build guide

### Step 1 — Set up the environment
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
CPU is enough. Full run ≈ 10–20 min on a laptop; a GPU is optional.

### Step 2 — Understand the city model (`sctwin/city.py`)
- 7×7 grid of intersections laid on Vellore's lat/lon → **49 nodes, 84 roads**.
- Every road = one traffic **sensor** with free-flow speed (60 km/h arterials, 40 otherwise).
- Grid split into **9 zones** — each zone has PM2.5, electricity load, flood depth.
- Elevation surface: low zones flood first. 3 hospitals are placed on the graph.
- *Upgrade path:* swap the grid for a real OSM graph with `osmnx` — the rest of the code only needs `G`, `zones`, `hospitals`.

### Step 3 — Get the data (`sctwin/datasets.py`, `data/README.md`)
- Download METR-LA, Beijing AQ, UCI Power, Sen1Floods11 into `data/` (table in `data/README.md`).
- If a file is missing, the loader uses the **coupled synthetic simulator**, which encodes the cascades:
  `rain → flood → road closure → congestion → PM2.5`, `temperature → load`, `flood → outage`.
- Every series is windowed: **12 h of history → 6 h forecast**.

### Step 4 — Build the models (`sctwin/models.py`)
| Model | Role |
|---|---|
| Persistence | naive lower bound |
| LSTM | recurrent baseline |
| CNN-LSTM | DeepCrowd-style conv + recurrent baseline |
| **MM-STFormer (proposed)** | graph neighbourhood mixing (adjacency `A`) → Transformer encoder (temporal attention) → **gated fusion** of exogenous signals (weather, calendar, cross-domain traffic) → residual on last observation |
| Flood U-Net | water segmentation from 2-band SAR (VV, VH) |

Ablations `w/o exogenous fusion` and `w/o graph mixing` prove each component helps.

### Step 5 — Train & evaluate everything
```bash
python scripts/run_all.py --quick   # 2-min smoke test
python scripts/run_all.py           # full experiment for the paper
```
Outputs in `results/`:
- `metrics.json`, `results_table.md` — MAE / RMSE / MAPE per model per domain, per-horizon MAE
- `figures/forecasts.png`, `horizon_mae.png`, `shap_groups.png`, `flood_gradcam.png`, `cascade_routing.png`
- `models/*.pt` — trained weights

### Step 6 — Explainability (`sctwin/xai.py`)
- **SHAP (GradientExplainer)** on MM-STFormer → % contribution of *own history / weather / calendar / traffic*.
  This is how you show that traffic actually drives the pollution forecast (multimodal gap closed).
- **Grad-CAM** on the U-Net bottleneck → which SAR regions triggered the flood mask.

### Step 7 — Risk tiers + cascades + routing (`sctwin/twin.py`)
- Forecasts → **LOW / MEDIUM / HIGH** tiers (PM2.5 uses CPCB breakpoints 60/90 µg/m³).
- `CascadeEngine.apply()` closes flooded roads, spills diverted traffic onto neighbours, raises PM2.5
  with a coefficient **fitted from data**, trips substations in deep floods.
- `emergency_route()` = Dijkstra on *forecast* travel times, avoiding closed roads, to the fastest hospital.
  Compared against static shortest-distance routing evaluated under true conditions.

### Step 8 — Run the live twin
```bash
streamlit run app.py
```
Slider = simulation hour; "what-if" slider injects extra flooding; pick where the emergency call is.
The map shows congestion-coloured roads, closed roads (dashed), zone risk, hospitals and the ambulance route;
the side panel lists cascade alerts.

### Step 9 — Rerun on real data for the paper
Put the real files in `data/`, rerun Step 5, and report *both* the real-data tables (traffic, pollution,
electricity, flood) and the simulated-city tables (cascades + routing, which need co-located data).
Run 3 seeds (`--seed 1/2/3`) and report mean ± std.

### Step 10 — Paper & slides
Use `results/results_table.md` for tables and `results/figures/` for figures.
Research gaps (from the proposal) → where they're addressed:

| Gap | Addressed by |
|---|---|
| Single dataset, no live testing | 4 public datasets + live Streamlit twin |
| No long-range temporal attention | Transformer encoder in MM-STFormer |
| No multimodal learning | gated exogenous fusion (weather + traffic → pollution) |
| Black-box | SHAP + Grad-CAM |
| No cascade forecasting | `CascadeEngine` |
| No actionable output | risk tiers + emergency routing |
