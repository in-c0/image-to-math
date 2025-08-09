#!/usr/bin/env python3
# math-to-image.py
# Reconstruct images from:
#   (A) Analytic JSON (tilewise poly+trig) exported by image-to-math.py
#   (B) Neural NPZ (MLP weights) exported by image-to-math.py
#
# Examples:
#   python math-to-image.py --analytic debug/coeffs.json --out recon.png --res 1024x1024
#   python math-to-image.py --nn debug/nn_weights.npz --out recon_nn.png --res 1920x1080

import argparse, os, sys, json
import numpy as np
import matplotlib.pyplot as plt

def parse_res(s):
    if s is None: return None
    w, h = s.lower().split('x'); return (int(w), int(h))

def save_img(img_arr, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(6,6), dpi=140)
    plt.imshow(np.clip(img_arr,0,1)); plt.axis("off")
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()
    print(f"Saved {out_path}")

# -------- Analytic (JSON) --------

def hann2d(h, w):
    hx = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(w)/(w-1))
    hy = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(h)/(h-1))
    return np.outer(hy, hx)

def eval_basis_names(u, v, names):
    """Return matrix T (M, H*W) for basis listed in names, given u,v grids in [-1,1]."""
    terms = []
    # NOTE: names are LaTeX-like; we parse patterns we emitted
    # We support: 1, u^i v^j, sin(k\pi u), cos(k\pi u), sin(k\pi v), cos(k\pi v),
    #             sin(k\pi u)cos(k\pi v), cos(k\pi u)sin(k\pi v)
    for name in names:
        if name == "1":
            terms.append(np.ones_like(u))
            continue
        if name.startswith("u^") or name.startswith("v^") or "u^" in name or "v^" in name:
            # Parse u^i v^j (possibly missing one factor)
            ui, vj = 0, 0
            parts = name.split()
            # crude parse: tokens like 'u^2' 'v^1'
            for tok in parts:
                if tok.startswith("u^"):
                    ui = int(tok[2:])
                if tok.startswith("v^"):
                    vj = int(tok[2:])
            terms.append((u**ui) * (v**vj))
            continue
        # trig
        # sin(k\pi u), cos(k\pi u)
        if "\\sin(" in name or "\\cos(" in name:
            # identify k and axes
            # convert to python eval: sin(pi*k*u) etc
            k = None
            if "\\sin(" in name:
                s = name.split("\\sin(")[1]
            else:
                s = name.split("\\cos(")[1]
            # s like "3\\pi u)" or "3\\pi u)\\cos(2\\pi v)"
            # We'll just compute term by term below:
            su = "\\sin" in name and "u)" in name.split("\\sin(")[1]
            cu = "\\cos" in name and "u)" in name.split("\\cos(")[1]
            sv = name.count("\\sin(") > (1 if su else 0) and "v)" in name.split("\\sin(")[-1]
            cv = name.count("\\cos(") > (1 if cu else 0) and "v)" in name.split("\\cos(")[-1]

            # Helper to extract the first integer before '\pi' for each segment
            def first_k(segment):
                seg = segment.split("\\pi")[0]
                # trim non-digits
                digits = "".join(ch for ch in seg if ch.isdigit())
                return int(digits) if digits else 1

            # Build term = (sin/cos on u) * (sin/cos on v) as applicable
            term = np.ones_like(u)
            if "\\sin(" in name and "u)" in name.split("\\sin(")[1]:
                ku = first_k(name.split("\\sin(")[1])
                term = term * np.sin(np.pi*ku*u)
            if "\\cos(" in name and "u)" in name.split("\\cos(")[1]:
                ku = first_k(name.split("\\cos(")[1])
                term = term * np.cos(np.pi*ku*u)
            # Mixed: following '*' portions for v
            if ")\\cos(" in name and "v)" in name.split(")\\cos(")[1]:
                kv = first_k(name.split(")\\cos(")[1])
                term = term * np.cos(np.pi*kv*v)
            if ")\\sin(" in name and "v)" in name.split(")\\sin(")[1]:
                kv = first_k(name.split(")\\sin(")[1])
                term = term * np.sin(np.pi*kv*v)
            terms.append(term)
            continue

        raise ValueError(f"Unsupported basis name: {name}")

    T = np.stack([t.ravel() for t in terms], axis=0)
    return T

def reconstruct_from_analytic_json(json_path, out_res=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    W0, H0 = data["img_w"], data["img_h"]
    W, H = out_res if out_res else (W0, H0)

    recon = np.zeros((H, W, 3), dtype=np.float64)
    acc   = np.zeros((H, W, 1), dtype=np.float64)

    for tile in data["tiles"]:
        x0, y0, tw, th = tile["x0"], tile["y0"], tile["w"], tile["h"]
        # Map original tile extent to new resolution (simple scale)
        sx = W / W0; sy = H / H0
        x0n = int(round(x0 * sx)); y0n = int(round(y0 * sy))
        twn = max(1, int(round(tw * sx))); thn = max(1, int(round(th * sy)))

        u = np.linspace(-1, 1, twn)
        v = np.linspace(-1, 1, thn)
        U, V = np.meshgrid(u, v)
        T = eval_basis_names(U, V, data["basis_names"])  # (M, thn*twn)

        # Compose per-channel
        tile_out = np.zeros((thn*twn, 3), dtype=np.float64)
        for ci, ch in enumerate(("R","G","B")):
            w = np.array(tile["coeffs"][ch], dtype=np.float64)  # (M,)
            tile_out[:, ci] = w @ T

        tile_out = tile_out.reshape(thn, twn, 3)
        win = hann2d(thn, twn)[..., None]
        y1, x1 = min(y0n+thn, H), min(x0n+twn, W)
        recon[y0n:y1, x0n:x1] += tile_out[:(y1-y0n), :(x1-x0n)] * win[:(y1-y0n), :(x1-x0n)]
        acc[y0n:y1, x0n:x1]   += win[:(y1-y0n), :(x1-x0n)]

    recon = recon / np.clip(acc, 1e-8, None)
    return np.clip(recon, 0, 1)

# -------- Neural (NPZ) --------

def fourier_features_xy(xy, bands):
    freqs = 2**np.arange(bands, dtype=np.float32) * np.pi
    t = xy[..., None] * freqs
    return np.concatenate([np.sin(t), np.cos(t)], axis=-1).reshape(xy.shape[0], -1)

def mlp_forward_np(X, W1,b1, W2,b2, W3,b3):
    H1 = np.tanh(X @ W1 + b1)
    H2 = np.tanh(H1 @ W2 + b2)
    Y  = 1/(1+np.exp(-(H2 @ W3 + b3)))
    return Y

def reconstruct_from_nn_npz(npz_path, out_res=None):
    Z = np.load(npz_path, allow_pickle=True)
    W1, b1 = Z["W1"], Z["b1"]
    W2, b2 = Z["W2"], Z["b2"]
    W3, b3 = Z["W3"], Z["b3"]
    bands  = int(Z["bands"][0])
    # original size stored, but we render at requested size
    if out_res is None:
        W, H = int(Z["img_w"][0]), int(Z["img_h"][0])
    else:
        W, H = out_res

    # Build coords/features
    x = np.linspace(-1, 1, W); y = np.linspace(-1, 1, H)
    Xg, Yg = np.meshgrid(x, y)
    xy = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)
    feats = np.concatenate([xy, fourier_features_xy(xy, bands=bands)], axis=1)

    Y = mlp_forward_np(feats, W1,b1, W2,b2, W3,b3)
    return np.clip(Y.reshape(H, W, 3), 0, 1)

# -------- CLI --------

def main():
    ap = argparse.ArgumentParser(description="Math → Image renderer (analytic JSON or neural NPZ)")
    ap.add_argument("--analytic", help="path to analytic coeffs JSON")
    ap.add_argument("--nn", help="path to neural weights NPZ")
    ap.add_argument("--res", type=parse_res, default=None, help="output resolution WxH (optional)")
    ap.add_argument("--out", default="reconstructed.png")
    args = ap.parse_args()

    if not args.analytic and not args.nn:
        print("Provide --analytic JSON or --nn NPZ.", file=sys.stderr); sys.exit(1)

    if args.analytic:
        img = reconstruct_from_analytic_json(args.analytic, out_res=args.res)
        save_img(img, args.out)

    if args.nn:
        # If both given, append suffix
        out = args.out if not args.analytic else os.path.splitext(args.out)[0] + "_nn.png"
        img = reconstruct_from_nn_npz(args.nn, out_res=args.res)
        save_img(img, out)

if __name__ == "__main__":
    main()
