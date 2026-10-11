"""End-to-end experiment runner: trains every module, evaluates, writes results/ for the paper.

    python scripts/run_all.py            # full run
    python scripts/run_all.py --quick    # smoke test (few epochs)
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

from sctwin.city import City
from sctwin import datasets as D
from sctwin.models import Persistence, LSTMForecaster, CNNLSTM, MMSTFormer, FloodUNet
from sctwin.train import Scaler, split, fit, predict, metrics, horizon_metrics
from sctwin.xai import shap_feature_importance, grouped, grad_cam
from sctwin.twin import CascadeEngine, emergency_route, naive_route, tier

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
EPOCHS = 3 if args.quick else 40
torch.manual_seed(args.seed); np.random.seed(args.seed)
ROOT = os.path.join(os.path.dirname(__file__), "..")
OUT, FIG, MOD = (os.path.join(ROOT, p) for p in ("results", "results/figures", "results/models"))
for p in (OUT, FIG, MOD): os.makedirs(p, exist_ok=True)
N_IN, N_OUT = 12, 6
R = {"config": dict(n_in=N_IN, n_out=N_OUT, epochs=EPOCHS, seed=args.seed)}

city = City()
sim = D.simulate_city(city, days=240, seed=args.seed)
A = city.sensor_adjacency()
print(f"City: {city.G.number_of_nodes()} intersections, {city.n_sensors} road sensors, {city.n_zones} zones")


def run_domain(name, Y, exo, adj, exo_groups):
    """Train the 4 models + 2 ablations for one domain; return metrics and the best model."""
    print(f"\n=== {name}: target {Y.shape}, exogenous {exo.shape} ===")
    N = Y.shape[1]
    F = np.concatenate([Y, exo], 1)
    tr, va, te = split(len(F))
    sc = Scaler().fit(F[tr])
    Fz = sc.tf(F)
    X, Yw = D.windows(Fz[:, :N], N_IN, N_OUT, exo=Fz[:, N:])
    tr, va, te = split(len(X))
    y_true = sc.inv(Yw[te], slice(0, N))
    E = exo.shape[1]
    candidates = {
        "Persistence": lambda: Persistence(N, N_OUT),
        "LSTM": lambda: LSTMForecaster(N + E, N, N_OUT),
        "CNN-LSTM": lambda: CNNLSTM(N + E, N, N_OUT),
        "MM-STFormer (ours)": lambda: MMSTFormer(N, E, N_OUT, A=adj),
        "  w/o exogenous fusion": lambda: MMSTFormer(N, 0, N_OUT, A=adj),
        "  w/o graph mixing": lambda: MMSTFormer(N, E, N_OUT, A=np.eye(N, dtype=np.float32)),
    }
    res, preds, models = {}, {}, {}
    for mname, make in candidates.items():
        torch.manual_seed(args.seed)
        m = make()
        Xtr, Xva, Xte = X[tr], X[va], X[te]
        if "w/o exogenous" in mname:
            Xtr, Xva, Xte = Xtr[..., :N], Xva[..., :N], Xte[..., :N]
        print(f"  training {mname.strip()} ({sum(p.numel() for p in m.parameters()):,} params)")
        m, secs = fit(m, Xtr, Yw[tr], Xva, Yw[va], epochs=EPOCHS, verbose=False)
        p = sc.inv(predict(m, Xte), slice(0, N))
        r = metrics(y_true, p); r["train_s"] = round(secs, 1); r["horizon_MAE"] = horizon_metrics(y_true, p)
        res[mname] = r; preds[mname] = p; models[mname] = m
        print(f"    MAE={r['MAE']:.3f} RMSE={r['RMSE']:.3f} MAPE={r['MAPE']:.2f}%")
    best = models["MM-STFormer (ours)"]
    torch.save(best.state_dict(), os.path.join(MOD, f"{name}_mmstformer.pt"))
    # explainability on the proposed model
    names = [f"{name}_{i}" for i in range(N)] + [n for g in exo_groups.values() for n in g]
    groups = {"own history": names[:N], **exo_groups}
    imp, imp_t, method = shap_feature_importance(best, X[tr][:100], X[te][:120], names)
    shap = dict(method=method, grouped_pct=grouped(imp, groups),
                temporal=np.asarray(imp_t).sum(1).tolist())
    return dict(metrics=res, shap=shap, test_slice=te, y_true=y_true, preds=preds, scaler=sc, model=best, X=X)


T_feat = D.time_features(sim["time"])
traffic = D.load_traffic(sim); pol = D.load_pollution(sim); pw = D.load_power(sim)
R["data_sources"] = dict(traffic=traffic["source"], pollution=pol["source"], electricity=pw["source"])
if traffic["source"] != "synthetic":
    A = np.eye(traffic["X"].shape[1], dtype=np.float32)   # supply METR-LA adj_mx.pkl here if available

# ---------------- Traffic ----------------
tf_exo = np.concatenate([T_feat, sim["rain"][:, None].astype(np.float32)], 1) if traffic["source"] == "synthetic" \
    else D.time_features(traffic["time"])
grp = {"calendar": ["sin_h", "cos_h", "sin_d", "cos_d"]}
if tf_exo.shape[1] == 5: grp["rain"] = ["rain"]
T_res = run_domain("traffic", traffic["X"], tf_exo, A, grp)

# ---------------- Pollution (multimodal: weather + traffic) ----------------
Z = pol["Y"].shape[1]
Zadj = np.ones((Z, Z), np.float32) / Z * 0.5 + np.eye(Z, dtype=np.float32) * 0.5
pexo = [pol["weather"], D.time_features(pol["time"])]
pgrp = {"weather": ["temp", "wind", "rain"], "calendar": ["sin_h", "cos_h", "sin_d", "cos_d"]}
if pol["exo_traffic"] is not None:
    pexo.append(pol["exo_traffic"]); pgrp["traffic (cross-domain)"] = [f"zspeed_{i}" for i in range(Z)]
P_res = run_domain("pollution", pol["Y"], np.concatenate(pexo, 1), Zadj, pgrp)

# ---------------- Electricity ----------------
Ze = pw["Y"].shape[1]
eadj = np.ones((Ze, Ze), np.float32) / Ze * 0.5 + np.eye(Ze, dtype=np.float32) * 0.5
eexo = [D.time_features(pw["time"])] + ([pw["weather"]] if pw["weather"] is not None else [])
egrp = {"calendar": ["sin_h", "cos_h", "sin_d", "cos_d"]}
if pw["weather"] is not None: egrp["weather"] = ["temp", "rain"]
E_res = run_domain("electricity", pw["Y"], np.concatenate(eexo, 1), eadj, egrp)

for k, r in (("traffic", T_res), ("pollution", P_res), ("electricity", E_res)):
    R[k] = dict(metrics=r["metrics"], shap=r["shap"])

# ---------------- Flood segmentation ----------------
print("\n=== flood segmentation ===")
Xf, Yf, fsrc = D.make_flood_chips(n=200 if args.quick else 800, seed=args.seed)
R["data_sources"]["flood"] = fsrc
mu, sd = Xf.mean((0, 2, 3), keepdims=True), Xf.std((0, 2, 3), keepdims=True)
Xf = (Xf - mu) / sd
ntr = int(0.8 * len(Xf))
unet = FloodUNet()
opt = torch.optim.Adam(unet.parameters(), 1e-3)
bce = torch.nn.BCEWithLogitsLoss()
for ep in range(2 if args.quick else 12):
    unet.train(); perm = np.random.permutation(ntr)
    for i in range(0, ntr, 16):
        b = perm[i:i + 16]
        opt.zero_grad(); l = bce(unet(torch.tensor(Xf[b])), torch.tensor(Yf[b])); l.backward(); opt.step()
unet.eval()
with torch.no_grad():
    pf = (torch.sigmoid(unet(torch.tensor(Xf[ntr:]))).numpy() > 0.5)
yt = Yf[ntr:] > 0.5
inter, union = (pf & yt).sum(), (pf | yt).sum()
thr_base = (Xf[ntr:, 0:1] < 0)          # classic threshold baseline on VV backscatter
R["flood"] = dict(IoU=float(inter / union), F1=float(f1_score(yt.ravel(), pf.ravel())),
                  baseline_threshold_IoU=float((thr_base & yt).sum() / (thr_base | yt).sum()))
print("  ", R["flood"])
torch.save(unet.state_dict(), os.path.join(MOD, "flood_unet.pt"))
k = ntr + int(np.argmax(Yf[ntr:].mean((1, 2, 3))))
cam, prob = grad_cam(unet, Xf[k], unet.b)
fig, ax = plt.subplots(1, 4, figsize=(12, 3.2))
for a, im, t in zip(ax, [Xf[k, 0], Yf[k, 0], prob, cam], ["SAR VV (norm.)", "Ground truth", "U-Net prob.", "Grad-CAM"]):
    a.imshow(im, cmap="gray" if t.startswith("SAR") else "viridis"); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "flood_gradcam.png"), dpi=200); plt.close()

# ---------------- Risk-tier accuracy (PM2.5, 6 h ahead) ----------------
yt6 = P_res["y_true"][:, -1].ravel(); pp6 = P_res["preds"]["MM-STFormer (ours)"][:, -1].ravel()
t_true = [tier("pm25", v) for v in yt6]; t_pred = [tier("pm25", v) for v in pp6]
R["risk_tier_pm25_6h"] = dict(accuracy=accuracy_score(t_true, t_pred),
                             macro_F1=f1_score(t_true, t_pred, average="macro"),
                             distribution={t: t_true.count(t) for t in ("LOW", "MEDIUM", "HIGH")})

# ---------------- Cascade coefficient + emergency routing ----------------
if traffic["source"] == "synthetic":
    sz = sim["sensor_zone"]
    free = np.array([d["free_speed"] for *_, d in city.G.edges(data=True)])
    cong = 1 - sim["speed"] / free
    zc = np.stack([cong[:, sz == z].mean(1) for z in range(city.n_zones)], 1)
    k_pm = float(np.polyfit(zc.ravel(), sim["pm25"].ravel(), 1)[0])
    eng = CascadeEngine(city, coef_cong_to_pm=k_pm)
    te0 = N_IN + T_res["test_slice"].start
    pred1 = T_res["preds"]["MM-STFormer (ours)"][:, 0]                 # 1-h-ahead speed forecast
    rng = np.random.default_rng(args.seed)
    rows = []
    stormy = [i for i in range(len(pred1)) if sim["flood"][te0 + i].max() > 30]
    calm = [i for i in range(len(pred1)) if sim["flood"][te0 + i].max() <= 30]
    for label, pool in (("normal", calm), ("flood", stormy)):
        for i in rng.choice(pool, size=min(300, len(pool)), replace=False) if pool else []:
            t = te0 + i
            true_c = np.clip(1 - sim["speed"][t] / free, 0, 0.95)
            fc_c = np.clip(1 - pred1[i] / free, 0, 0.95)
            fc_c, _, _, closed, _ = eng.apply(sim["flood"][t], fc_c, sim["pm25"][t], np.ones(city.n_zones))
            src = int(rng.integers(city.G.number_of_nodes()))
            twin = emergency_route(city, fc_c, src)
            if twin is None: continue
            _, path, h = twin
            # evaluate twin path under TRUE conditions
            tt = sum(60 * city.G.edges[u, v]["length_km"] /
                     max(city.G.edges[u, v]["free_speed"] * (1 - 0.6 * true_c[city.G.edges[u, v]["sensor"]]), 3)
                     for u, v in zip(path, path[1:]))
            nt, _ = naive_route(city, true_c, src, h)
            rows.append((label, tt, nt))
    out = {}
    for label in ("normal", "flood"):
        r = [x for x in rows if x[0] == label]
        if not r: continue
        tw = np.array([x[1] for x in r]); nv = np.array([x[2] for x in r])
        ok = np.isfinite(nv)
        out[label] = dict(trips=len(r), twin_mean_min=float(tw.mean()),
                          static_mean_min=float(nv[ok].mean()) if ok.any() else None,
                          static_blocked_pct=float(100 * (~ok).mean()),
                          twin_reached_pct=100.0,
                          time_saved_pct_on_passable=float(100 * (1 - tw[ok].sum() / nv[ok].sum())) if ok.any() else None)
    R["emergency_routing"] = out; R["cascade_k_pm_per_congestion"] = k_pm
    print("\n=== routing ===\n", json.dumps(out, indent=1))

    # cascade demo figure: one storm hour
    t = te0 + (stormy[0] if stormy else 0)
    c0 = np.clip(1 - sim["speed"][t] / free, 0, .95)
    c1, pm1, ld1, closed, ev = eng.apply(sim["flood"][t], c0, sim["pm25"][t], np.ones(city.n_zones))
    R["cascade_example"] = dict(hour=str(sim["time"][t]), events=ev)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for u, v, d in city.G.edges(data=True):
        x = [city.G.nodes[u]["lon"], city.G.nodes[v]["lon"]]; y = [city.G.nodes[u]["lat"], city.G.nodes[v]["lat"]]
        col = "black" if d["closed"] else plt.cm.RdYlGn_r(c1[d["sensor"]])
        ax.plot(x, y, color=col, lw=4 if d["closed"] else 2.5, ls="--" if d["closed"] else "-")
    r_ = emergency_route(city, c1, 22)
    if r_:
        p = r_[1]; ax.plot([city.G.nodes[n]["lon"] for n in p], [city.G.nodes[n]["lat"] for n in p], "b-", lw=6, alpha=.35, label=f"ambulance route ({r_[0]:.1f} min)")
    for h in city.hospitals: ax.plot(city.G.nodes[h]["lon"], city.G.nodes[h]["lat"], "P", ms=13, color="red")
    ax.set_title("Cascade: flood closures (dashed) -> congestion -> ambulance reroute", fontsize=9)
    ax.legend(loc="lower right", fontsize=8); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "cascade_routing.png"), dpi=200); plt.close()

# ---------------- Figures ----------------
fig, axs = plt.subplots(1, 3, figsize=(14, 3.6))
for a, (name, r) in zip(axs, (("Traffic speed (km/h)", T_res), ("PM2.5 (ug/m3)", P_res), ("Load (MW)", E_res))):
    y = r["y_true"][:168, 0, 0]
    a.plot(y, "k", lw=1.6, label="ground truth")
    for m, c in (("CNN-LSTM", "tab:orange"), ("MM-STFormer (ours)", "tab:blue")):
        a.plot(r["preds"][m][:168, 0, 0], c=c, lw=1, label=m)
    a.set_title(name); a.set_xlabel("test hour")
axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "forecasts.png"), dpi=200); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
for a, (name, r) in zip(axs, (("Traffic", T_res), ("Pollution", P_res), ("Electricity", E_res))):
    for m in ("Persistence", "LSTM", "CNN-LSTM", "MM-STFormer (ours)"):
        a.plot(range(1, N_OUT + 1), r["metrics"][m]["horizon_MAE"], "o-", label=m)
    a.set_title(f"{name}: MAE vs horizon"); a.set_xlabel("hours ahead")
axs[0].legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "horizon_mae.png"), dpi=200); plt.close()

fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
for a, (name, r) in zip(axs, (("Traffic", T_res), ("Pollution", P_res), ("Electricity", E_res))):
    g = r["shap"]["grouped_pct"]; a.barh(list(g), list(g.values()), color="tab:purple"); a.set_title(f"{name}: SHAP share (%)")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "shap_groups.png"), dpi=200); plt.close()

json.dump(R, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)
with open(os.path.join(OUT, "results_table.md"), "w") as f:
    for dom in ("traffic", "pollution", "electricity"):
        f.write(f"\n### {dom} ({R['data_sources'].get(dom, '')}), {N_OUT}-h horizon\n\n| Model | MAE | RMSE | MAPE % |\n|---|---|---|---|\n")
        for m, r in R[dom]["metrics"].items():
            f.write(f"| {m} | {r['MAE']:.3f} | {r['RMSE']:.3f} | {r['MAPE']:.2f} |\n")
print("\nDone. See results/metrics.json, results/results_table.md, results/figures/")
