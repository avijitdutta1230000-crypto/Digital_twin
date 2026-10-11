# Putting the real datasets here

The code runs without these (it uses the coupled synthetic city), but for the
paper you should run at least Traffic + Pollution on the real data.

| Domain | Dataset | What to place in `data/` | Where to get it |
|---|---|---|---|
| Traffic | METR-LA | `metr-la.h5` | DCRNN repo (github.com/liyaguang/DCRNN) → Google Drive link in its README; also on Kaggle ("METR-LA") |
| Pollution | Beijing Multi-Site Air Quality (UCI #501) | the 12 `PRSA_Data_*.csv` files | archive.ics.uci.edu/dataset/501 |
| Electricity | UCI Household Power Consumption (#235) | `household_power_consumption.txt` | archive.ics.uci.edu/dataset/235 |
| Floods | Sen1Floods11 | `sen1floods11/S1Hand/*.tif` and `sen1floods11/LabelHand/*.tif` | github.com/cloudtostreet/Sen1Floods11 (GCS bucket, hand-labelled split, ~1 GB) + `pip install rasterio` |
| Routing | OSM road network (optional upgrade) | — | `pip install osmnx`; `osmnx.graph_from_place("Vellore, India")` |

When a real file is found the loader uses it automatically and `results/metrics.json`
records it under `data_sources`. Note: on real data the cross-domain link
(traffic → pollution) is unavailable because METR-LA (Los Angeles) and Beijing AQ are
different cities — say so in the paper and keep the cascade experiments on the
co-located simulated city.
