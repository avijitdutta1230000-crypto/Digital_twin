"""Explainability: SHAP for the forecasters, Grad-CAM for the flood segmenter."""
import numpy as np
import torch


def shap_feature_importance(model, X_background, X_explain, feature_names, nsamples=50):
    """Mean |SHAP| per input feature (aggregated over time steps and outputs).

    Uses shap.GradientExplainer (expected gradients); falls back to
    gradient x input if shap is not installed.
    """
    model.eval()
    Xb, Xe = torch.tensor(X_background), torch.tensor(X_explain)

    class Agg(torch.nn.Module):            # explain the mean forecast across the horizon/nodes
        def __init__(s, m): super().__init__(); s.m = m
        def forward(s, x): return s.m(x).mean((1, 2)).unsqueeze(1)

    try:
        import shap
        sv = shap.GradientExplainer(Agg(model), Xb).shap_values(Xe, nsamples=nsamples)
        sv = np.asarray(sv).reshape(Xe.shape)
        method = "SHAP (GradientExplainer)"
    except Exception as ex:  # pragma: no cover
        Xe.requires_grad_(True)
        Agg(model)(Xe).sum().backward()
        sv = (Xe.grad * (Xe - Xb.mean(0))).detach().numpy()
        method = f"gradient x input (shap unavailable: {ex.__class__.__name__})"
    imp = np.abs(sv).mean((0, 1))
    return dict(zip(feature_names, imp.tolist())), np.abs(sv).mean(0), method


def grouped(importance, groups):
    """Sum per-feature importance into named groups, normalised to 100%."""
    out = {g: sum(importance[f] for f in fs) for g, fs in groups.items()}
    tot = sum(out.values()) or 1.0
    return {g: 100 * v / tot for g, v in out.items()}


def grad_cam(model, x, layer):
    """Grad-CAM heatmap for a segmentation network (target = predicted water area)."""
    acts, grads = {}, {}
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    model.eval()
    xt = torch.tensor(x[None]).requires_grad_(True)
    out = torch.sigmoid(model(xt))
    out.sum().backward()
    h1.remove(); h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0].detach().numpy()
    cam = np.kron(cam, np.ones((x.shape[1] // cam.shape[0], x.shape[2] // cam.shape[1])))
    return cam / (cam.max() + 1e-9), out[0, 0].detach().numpy()
