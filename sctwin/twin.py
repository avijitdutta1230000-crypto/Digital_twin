"""Digital-twin core: risk tiers, cascade propagation and emergency routing."""
import numpy as np
import networkx as nx

# Risk thresholds (Indian CPCB AQI breakpoints for PM2.5; others are design choices)
TIERS = {
    "traffic": [(0.35, "LOW"), (0.6, "MEDIUM"), (1.01, "HIGH")],     # congestion ratio 1 - v/v_free
    "pm25":    [(60, "LOW"), (90, "MEDIUM"), (1e9, "HIGH")],          # ug/m3 (CPCB satisfactory / moderate)
    "flood":   [(10, "LOW"), (30, "MEDIUM"), (1e9, "HIGH")],          # cm of water
    "power":   [(0.85, "LOW"), (0.95, "MEDIUM"), (1e9, "HIGH")],      # load / zone capacity
}


def tier(kind, v):
    for thr, name in TIERS[kind]:
        if v < thr:
            return name
    return "HIGH"


class CascadeEngine:
    """Propagates a forecast shock in one system to the others.

    flood (zone) --closure--> roads --reroute--> traffic congestion --emissions--> PM2.5
    flood (zone) --substation trip--> power outage
    Coefficients are estimated from data in scripts/run_all.py (linear regression of
    the downstream variable on the upstream one), not hard-coded guesses.
    """
    def __init__(self, city, coef_cong_to_pm=90.0, flood_close_cm=30.0):
        self.city, self.k_pm, self.close_cm = city, coef_cong_to_pm, flood_close_cm

    def apply(self, flood_cm, congestion, pm25, load_ratio):
        G = self.city.G
        events, closed = [], []
        for u, v, d in G.edges(data=True):
            z = G.nodes[u]["zone"]
            d["closed"] = bool(flood_cm[z] >= self.close_cm)
            if d["closed"]:
                closed.append(d["sensor"])
        cong = congestion.copy()
        if closed:
            A = self.city.sensor_adjacency()
            spill = A[:, closed].sum(1)                            # neighbours absorb diverted flow
            cong = np.clip(cong + 0.15 * np.clip(spill, 0, 1.5), 0, 0.95)
            cong[closed] = 0.95
            events.append(f"{len(closed)} road segments closed by flooding -> traffic diverted")
        sz = self.city.sensor_zone()
        d_cong = np.array([(cong - congestion)[sz == z].mean() for z in range(self.city.n_zones)])
        pm_new = pm25 + self.k_pm * np.clip(d_cong, 0, None)
        if (pm_new - pm25).max() >= 1:
            events.append(f"PM2.5 up to +{(pm_new - pm25).max():.0f} ug/m3 from rerouted traffic")
        load_new = load_ratio.copy()
        trip = flood_cm >= 45
        if trip.any():
            load_new[trip] = 0.0
            events.append(f"{int(trip.sum())} zone substation(s) tripped -> outage")
        return cong, pm_new, load_new, closed, events


def emergency_route(city, congestion, src, dst=None):
    """Fastest route for an ambulance using *forecast* edge travel times.

    Emergency vehicles get 40% of congestion delay removed (siren priority) but
    cannot cross closed (flooded) roads. If dst is None, nearest hospital by time.
    """
    G = city.G
    H = nx.Graph()
    for u, v, d in G.edges(data=True):
        if d["closed"]:
            continue
        speed = d["free_speed"] * (1 - 0.6 * congestion[d["sensor"]])
        H.add_edge(u, v, t=60 * d["length_km"] / max(speed, 3))
    targets = [dst] if dst is not None else city.hospitals
    best = None
    for h in targets:
        try:
            tt, path = nx.single_source_dijkstra(H, src, h, weight="t")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if best is None or tt < best[0]:
            best = (tt, path, h)
    return best   # (minutes, [nodes], hospital) or None


def naive_route(city, congestion, src, dst):
    """Shortest-distance route (what a static map gives), evaluated under real conditions."""
    G = city.G
    path = nx.shortest_path(G, src, dst, weight="length_km")
    tt = 0.0
    for u, v in zip(path, path[1:]):
        d = G.edges[u, v]
        if d["closed"]:
            return float("inf"), path
        tt += 60 * d["length_km"] / max(d["free_speed"] * (1 - 0.6 * congestion[d["sensor"]]), 3)
    return tt, path
