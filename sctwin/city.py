"""City model: a grid road network with zones, sensors, hospitals and low-lying areas.

The city is a lattice of intersections laid over real-world coordinates (default:
Vellore, Tamil Nadu). Every intersection is a node; every road is an edge with a
length, free-flow speed and a traffic sensor id. Zones group intersections and
carry the pollution / electricity / flood state.
"""
from dataclasses import dataclass, field
import numpy as np
import networkx as nx


@dataclass
class City:
    rows: int = 7
    cols: int = 7
    origin: tuple = (12.9165, 79.1325)   # Vellore (lat, lon)
    spacing_deg: float = 0.006            # ~650 m between intersections
    seed: int = 7
    G: nx.Graph = field(init=False)
    zones: dict = field(init=False)
    hospitals: list = field(init=False)
    elevation: np.ndarray = field(init=False)

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        G = nx.grid_2d_graph(self.rows, self.cols)
        G = nx.convert_node_labels_to_integers(G, label_attribute="rc")
        for n, d in G.nodes(data=True):
            r, c = d["rc"]
            d["lat"] = self.origin[0] + r * self.spacing_deg
            d["lon"] = self.origin[1] + c * self.spacing_deg
            d["zone"] = (r * 3 // self.rows) * 3 + (c * 3 // self.cols)   # 3x3 zones
        for i, (u, v, d) in enumerate(G.edges(data=True)):
            d["sensor"] = i
            d["length_km"] = 0.65
            arterial = G.nodes[u]["rc"][0] == self.rows // 2 or G.nodes[u]["rc"][1] == self.cols // 2
            d["free_speed"] = 60.0 if arterial else 40.0
            d["closed"] = False
        self.G = G
        nz = max(nx.get_node_attributes(G, "zone").values()) + 1
        self.zones = {z: [n for n, d in G.nodes(data=True) if d["zone"] == z] for z in range(nz)}
        # smooth random terrain; low elevation => flood prone
        r, c = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij")
        elev = 220 + 8 * np.sin(r / 2.0) + 6 * np.cos(c / 2.5) + rng.normal(0, 1.5, r.shape)
        self.elevation = elev
        for n, d in G.nodes(data=True):
            d["elev"] = float(elev[d["rc"]])
        self.hospitals = [0, self.rows * self.cols - 1, (self.rows // 2) * self.cols + self.cols // 2]

    @property
    def n_sensors(self):
        return self.G.number_of_edges()

    @property
    def n_zones(self):
        return len(self.zones)

    def edge_list(self):
        return list(self.G.edges())

    def sensor_adjacency(self):
        """Line-graph adjacency between road sensors (roads that share an intersection)."""
        L = nx.line_graph(self.G)
        idx = {e: self.G.edges[e]["sensor"] for e in self.G.edges()}
        A = np.eye(self.n_sensors, dtype=np.float32)
        for a, b in L.edges():
            ia = idx.get(a, idx.get((a[1], a[0])))
            ib = idx.get(b, idx.get((b[1], b[0])))
            A[ia, ib] = A[ib, ia] = 1.0
        return A / A.sum(1, keepdims=True)

    def sensor_zone(self):
        return np.array([self.G.nodes[u]["zone"] for u, v in self.G.edges()])

    def node_low_lying(self, q=0.25):
        thr = np.quantile(self.elevation, q)
        return {n for n, d in self.G.nodes(data=True) if d["elev"] <= thr}
