"""Data layer.

Each loader first looks for the REAL public dataset under data/ (see data/README.md).
If it is missing, it falls back to the coupled synthetic city simulator below so the
whole pipeline still runs. Every loader returns a dict with a `source` key saying
which one was used — report that honestly in the paper.
"""
import os
import glob
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


# ----------------------------------------------------------------------------
# Coupled synthetic city simulator (hourly)
# ----------------------------------------------------------------------------
def simulate_city(city, days=150, seed=0):
    """Generates mutually-consistent traffic, weather, pollution, power and flood series.

    Couplings (these are the cascades the twin must learn):
      rain  -> zone flood depth -> road closures -> traffic speed drop
      traffic congestion (per zone) -> PM2.5 / NO2 rise
      temperature + hour-of-day -> electricity demand; floods -> outages (demand dips)
    """
    rng = np.random.default_rng(seed)
    T = days * 24
    t = np.arange(T)
    hour, dow = t % 24, (t // 24) % 7
    S, Z = city.n_sensors, city.n_zones
    sz = city.sensor_zone()

    # weather
    temp = 28 + 5 * np.sin(2 * np.pi * (hour - 9) / 24) + 2 * np.sin(2 * np.pi * t / (24 * 60)) + rng.normal(0, 0.8, T)
    wind = np.clip(2.5 + rng.gamma(2, 0.8, T) - 1.5, 0.2, None)
    rain = np.zeros(T)
    for s in rng.choice(T - 24, size=days // 3, replace=False):   # storm events
        dur = rng.integers(3, 14)
        rain[s:s + dur] += rng.gamma(2.0, 9.0) * np.hanning(dur + 2)[1:-1]
    rain += rng.exponential(0.3, T) * (rng.random(T) < 0.05)

    # flood depth per zone (cm): accumulates with rain, drains over time, worse in low zones
    zone_elev = np.array([np.mean([city.G.nodes[n]["elev"] for n in city.zones[z]]) for z in range(Z)])
    vuln = (zone_elev.max() - zone_elev) / (np.ptp(zone_elev) + 1e-6)            # 0..1
    flood = np.zeros((T, Z))
    for i in range(1, T):
        flood[i] = np.clip(0.8 * flood[i - 1] + rain[i] * (0.15 + 0.6 * vuln) - 0.8, 0, 120)

    # traffic speed per sensor (km/h)
    free = np.array([d["free_speed"] for _, _, d in city.G.edges(data=True)])
    rush = np.exp(-((hour - 9) ** 2) / 3) + 1.2 * np.exp(-((hour - 18.5) ** 2) / 4)
    weekend = (dow >= 5).astype(float)
    demand = (rush * (1 - 0.5 * weekend))[:, None] * rng.uniform(0.6, 1.0, S)[None]
    A = city.sensor_adjacency()
    incidents = np.zeros((T, S))
    for s in rng.choice(T, size=days * 2):
        e = rng.integers(S)
        incidents[s:s + rng.integers(1, 4), e] += rng.uniform(0.3, 0.6)
    incidents = incidents @ A.T * 2.0                                            # spill to neighbours
    flood_on_road = np.clip(flood[:, sz] / 80.0, 0, 0.5)
    cong = np.clip(0.55 * demand + incidents + np.clip(0.02 * rain[:, None], 0, 0.25) + flood_on_road, 0, 0.95)
    speed = free[None] * (1 - cong) + rng.normal(0, 1.5, (T, S))
    speed = np.clip(speed, 3, None)

    # pollution per zone: driven by zone congestion, dispersed by wind, washed out by rain
    zone_cong = np.stack([cong[:, sz == z].mean(1) for z in range(Z)], 1)
    pm25 = np.zeros((T, Z))
    pm25[0] = 40
    for i in range(1, T):
        src = 30 + 160 * zone_cong[i] + 6 * (hour[i] in (22, 23, 0, 1, 2))
        pm25[i] = 0.8 * pm25[i - 1] + 0.2 * src / (0.5 + 0.2 * wind[i]) - 0.6 * rain[i]
        pm25[i] = np.clip(pm25[i] + rng.normal(0, 2, Z), 5, None)
    no2 = 15 + 60 * zone_cong + rng.normal(0, 3, (T, Z))

    # electricity per zone (MW)
    base = rng.uniform(8, 15, Z)
    load = base[None] * (0.7 + 0.25 * np.sin(2 * np.pi * (hour - 14) / 24)[:, None]
                         + 0.03 * (temp[:, None] - 28) + 0.1 * (hour[:, None] >= 18) * (hour[:, None] <= 22))
    outage = flood > 30
    load = np.where(outage, load * 0.3, load) + rng.normal(0, 0.3, (T, Z))

    ts = pd.date_range("2025-06-01", periods=T, freq="h")
    return dict(time=ts, speed=speed.astype(np.float32), rain=rain, temp=temp, wind=wind,
                flood=flood.astype(np.float32), pm25=pm25.astype(np.float32), no2=no2.astype(np.float32),
                load=load.astype(np.float32), sensor_zone=sz, zone_vuln=vuln, source="synthetic")


# ----------------------------------------------------------------------------
# Real-dataset loaders
# ----------------------------------------------------------------------------
def load_traffic(sim):
    """METR-LA (data/metr-la.h5, 207 sensors, 5-min speed). Resampled hourly."""
    p = os.path.join(DATA, "metr-la.h5")
    if os.path.exists(p):
        df = pd.read_hdf(p).replace(0, np.nan).ffill().bfill().resample("1h").mean()
        return dict(X=df.values.astype(np.float32), time=df.index, source="METR-LA")
    return dict(X=sim["speed"], time=sim["time"], source="synthetic")


def load_pollution(sim):
    """Beijing Multi-Site AQ (data/PRSA_Data_*.csv, 12 stations)."""
    files = sorted(glob.glob(os.path.join(DATA, "PRSA_Data_*.csv")))
    if files:
        frames = []
        for f in files:
            d = pd.read_csv(f)
            d["time"] = pd.to_datetime(d[["year", "month", "day", "hour"]])
            frames.append(d.set_index("time")[["PM2.5", "NO2", "TEMP", "WSPM", "RAIN"]])
        pm = pd.concat([f["PM2.5"] for f in frames], axis=1).interpolate().bfill()
        w = frames[0][["TEMP", "WSPM", "RAIN"]].interpolate().bfill()
        return dict(Y=pm.values.astype(np.float32), weather=w.values.astype(np.float32),
                    exo_traffic=None, time=pm.index, source="Beijing-MultiSite")
    weather = np.stack([sim["temp"], sim["wind"], sim["rain"]], 1).astype(np.float32)
    sz = sim["sensor_zone"]
    zone_speed = np.stack([sim["speed"][:, sz == z].mean(1) for z in range(sim["pm25"].shape[1])], 1)
    return dict(Y=sim["pm25"], weather=weather, exo_traffic=zone_speed.astype(np.float32),
                time=sim["time"], source="synthetic")


def load_power(sim):
    """UCI Household Power Consumption (data/household_power_consumption.txt)."""
    p = os.path.join(DATA, "household_power_consumption.txt")
    if os.path.exists(p):
        d = pd.read_csv(p, sep=";", na_values="?", low_memory=False)
        d["time"] = pd.to_datetime(d["Date"] + " " + d["Time"], dayfirst=True)
        s = d.set_index("time")["Global_active_power"].resample("1h").mean().interpolate()
        return dict(Y=s.values[:, None].astype(np.float32), weather=None, time=s.index, source="UCI-Household")
    weather = np.stack([sim["temp"], sim["rain"]], 1).astype(np.float32)
    return dict(Y=sim["load"], weather=weather, time=sim["time"], source="synthetic")


def make_flood_chips(n=600, size=64, seed=0):
    """Sen1Floods11-like chips: 2-band SAR backscatter (VV, VH in dB) + water mask.

    Real data: put Sen1Floods11 hand-labelled chips in data/sen1floods11/{S1Hand,LabelHand}
    and install rasterio; this function will use them instead.
    """
    root = os.path.join(DATA, "sen1floods11")
    if os.path.isdir(root):
        import rasterio
        xs, ys = [], []
        for f in sorted(glob.glob(os.path.join(root, "S1Hand", "*.tif")))[:n]:
            lab = f.replace("S1Hand", "LabelHand")
            with rasterio.open(f) as a, rasterio.open(lab) as b:
                x, y = a.read()[:, :size * 4:4, :size * 4:4], b.read(1)[:size * 4:4, :size * 4:4]
            xs.append(np.nan_to_num(x)); ys.append((y == 1).astype(np.float32))
        return np.stack(xs).astype(np.float32), np.stack(ys)[:, None], "Sen1Floods11"
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    X = np.zeros((n, 2, size, size), np.float32)
    Y = np.zeros((n, 1, size, size), np.float32)
    for i in range(n):
        m = np.zeros((size, size), bool)
        for _ in range(rng.integers(0, 4)):               # flood blobs
            cx, cy, r = rng.uniform(0, size, 2).tolist() + [rng.uniform(5, 18)]
            m |= ((xx - cx) ** 2 / rng.uniform(0.5, 2) + (yy - cy) ** 2) < r ** 2
        if rng.random() < 0.5:                             # river
            c = rng.uniform(10, 54); w = rng.uniform(2, 5)
            m |= np.abs(xx - (c + 6 * np.sin(yy / rng.uniform(6, 12)))) < w
        speckle = rng.gamma(1.0, 1.0, (2, size, size))          # single-look speckle (hard)
        land = np.array([-9.0, -16.0])[:, None, None] + rng.normal(0, 1.5)
        water = np.array([-16.0, -23.0])[:, None, None] + rng.normal(0, 1.5)
        dark = np.zeros((size, size), bool)                     # look-alikes: tarmac, shadow, dry sand
        for _ in range(rng.integers(0, 3)):
            cx, cy = rng.uniform(0, size, 2); dark |= (np.abs(xx - cx) < rng.uniform(3, 10)) & (np.abs(yy - cy) < rng.uniform(3, 10))
        lab = np.where(dark[None], water + 2.0, land)
        X[i] = np.where(m[None], water, lab) + 10 * np.log10(speckle + 1e-3)
        Y[i, 0] = m
    return X, Y, "synthetic"


# ----------------------------------------------------------------------------
def windows(X, n_in=12, n_out=6, exo=None):
    """Slice a (T, N) series into supervised (B, n_in, F) -> (B, n_out, N) samples."""
    feats = X if exo is None else np.concatenate([X, exo], 1)
    xs, ys = [], []
    for i in range(len(X) - n_in - n_out + 1):
        xs.append(feats[i:i + n_in]); ys.append(X[i + n_in:i + n_in + n_out])
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def time_features(time):
    h = np.asarray(time.hour); d = np.asarray(time.dayofweek)
    return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                     np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], 1).astype(np.float32)
