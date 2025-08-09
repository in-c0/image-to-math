#!/usr/bin/env python3
# image-to-math.py
# One CLI for:
#   - Analytic (poly+trig tiled) reconstruction with JSON/LaTeX export
#   - Neural implicit reconstruction (tiny MLP) at full resolution
#
# Examples:
#   python image-to-math.py                           # uses input.jpg, analytic xhigh, exports JSON + global LaTeX
#   python image-to-math.py img.png --method analytic --detail high --res 1024x1024 --export-json debug/coeffs.json --final-latex debug/final.tex --no-gui
#   python image-to-math.py img.png --method nn --nn-iters 600 --nn-train-side 384 --no-gui
#   python image-to-math.py img.png --method both --detail med --nn-iters 300 --no-gui

import argparse, os, sys, json, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
from tqdm import tqdm
import helper

# --------------------------
# Common helpers
# --------------------------

def parse_res(s):
    if s is None: return None
    try:
        w, h = s.lower().split('x')
        return (int(w), int(h))
    except Exception as e:
        raise argparse.ArgumentTypeError("Resolution must be WxH, e.g., 1024x1024")

PRESETS = {
    #        TILE  OVERLAP  POLY  FREQ   L2
    "xlow":   (192,   0.25,    2,    8, 1e-2),  # fastest, blurriest
    "low":    (160,   0.33,    2,   12, 5e-3),
    "med":    (128,   0.50,    2,   16, 1e-3),
    "high":   (96,    0.50,    2,   24, 5e-4),
    "xhigh":  (64,    0.50,    2,   32, 3e-4),  # slowest, sharpest
}

def load_img_or_die(path, target_res):
    if not os.path.exists(path):
        print(f"Error: input image not found: {path}", file=sys.stderr)
        sys.exit(1)
    img = Image.open(path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    if target_res:
        img = img.resize(target_res, Image.LANCZOS)
    return img

def save_img(img_arr, title, out_path, no_gui=False):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(6,6), dpi=140)
    plt.imshow(np.clip(img_arr, 0, 1))
    plt.axis("off"); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight")
    if not no_gui: plt.show()
    plt.close()

# --------------------------
# Analytic (poly+trig tiled)
# --------------------------

def build_basis_from_grids(X, Y, poly_deg=2, max_freq=16):
    """
    X, Y: meshgrids in local normalized coordinates u,v in [-1,1]
    Returns:
      T: (M, H*W) basis eval matrix, basis_names: list of LaTeX-safe names
    """
    terms = []
    names = []
    # Polynomials
    for i in range(poly_deg + 1):
        for j in range(poly_deg + 1 - i):
            terms.append((X**i) * (Y**j))
            names.append("1" if (i==0 and j==0) else f"u^{i} v^{j}")
    # Sines/cosines + mixed terms (LaTeX names use \pi not Unicode π)
    for k in range(1, max_freq + 1):
        su, cu = np.sin(np.pi * k * X), np.cos(np.pi * k * X)
        sv, cv = np.sin(np.pi * k * Y), np.cos(np.pi * k * Y)
        terms += [su, cu, sv, cv, su*cv, cu*sv]
        names += [
            f"\\sin({k}\\pi u)", f"\\cos({k}\\pi u)",
            f"\\sin({k}\\pi v)", f"\\cos({k}\\pi v)",
            f"\\sin({k}\\pi u)\\cos({k}\\pi v)",
            f"\\cos({k}\\pi u)\\sin({k}\\pi v)"
        ]
    T = np.stack([t.ravel() for t in terms], axis=0)  # (M, N)
    return T, names

def ridge_solve(T, y_flat, l2):
    A = T @ T.T
    b = T @ y_flat
    A += l2 * np.eye(A.shape[0])
    return np.linalg.solve(A, b)  # (M,)

def hann2d(h, w):
    hx = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(w)/(w-1))
    hy = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(h)/(h-1))
    return np.outer(hy, hx)

def draw_tile_grid(img, tile, overlap, color=(255,0,0), alpha=128):
    H, W = img.size[1], img.size[0]
    step = max(1, int(tile*(1-overlap)))
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    d = ImageDraw.Draw(overlay)
    for x in range(0, W, step):
        d.line([(x,0),(x,H)], fill=color+(alpha,), width=1)
    for y in range(0, H, step):
        d.line([(0,y),(W,y)], fill=color+(alpha,), width=1)
    return Image.alpha_composite(img.convert("RGBA"), overlay)

def reconstruct_tiled_analytic(arr, tile, overlap, poly, freq, l2, collect_coeffs=False, debug=False):
    """
    Returns:
      recon: (H,W,3)
      err_map: (H,W) or None
      coeffs: dict or None
    """
    H, W = arr.shape[:2]
    step = max(1, int(tile * (1 - overlap)))
    out = np.zeros_like(arr, dtype=np.float64)
    acc = np.zeros((H, W, 1), dtype=np.float64)
    err_map = np.zeros((H, W), dtype=np.float64) if debug else None

    coeffs = {
        "tile": tile, "overlap": overlap, "poly": poly, "freq": freq, "l2": l2,
        "img_w": W, "img_h": H,
        "basis_names": None,
        "tiles": []
    } if collect_coeffs else None

    xs = list(range(0, W, step))
    ys = list(range(0, H, step))
    total = len(xs) * len(ys)

    pbar = tqdm(total=total, desc="Analytic reconstruct (tiled)")
    k = 0
    for y0 in ys:
        for x0 in xs:
            y1 = min(y0 + tile, H); x1 = min(x0 + tile, W)
            th, tw = y1 - y0, x1 - x0
            tile_img = arr[y0:y1, x0:x1, :]

            # pad edge tiles to (tile,tile)
            if th != tile or tw != tile:
                pad = np.zeros((tile, tile, 3), dtype=arr.dtype)
                pad[:th, :tw] = tile_img
                tile_img = pad
                th_pad, tw_pad = tile, tile
            else:
                th_pad, tw_pad = th, tw

            # Build local coordinate grids u,v in [-1,1]
            u = np.linspace(-1, 1, tw_pad)
            v = np.linspace(-1, 1, th_pad)
            X, Y = np.meshgrid(u, v)

            Tmat, basis_names = build_basis_from_grids(X, Y, poly_deg=poly, max_freq=freq)
            if collect_coeffs and coeffs["basis_names"] is None:
                coeffs["basis_names"] = basis_names

            flat = tile_img.reshape(-1, 3)
            recon_flat = np.zeros_like(flat)
            wR = wG = wB = None
            for c in range(3):
                wcoef = ridge_solve(Tmat, flat[:, c], l2)
                recon_flat[:, c] = wcoef @ Tmat
                if collect_coeffs:
                    if c == 0: wR = wcoef
                    elif c == 1: wG = wcoef
                    else: wB = wcoef

            if collect_coeffs:
                coeffs["tiles"].append({
                    "k": k,
                    "x0": int(x0), "y0": int(y0), "w": int(tw), "h": int(th),
                    "coeffs": {"R": wR.tolist(), "G": wG.tolist(), "B": wB.tolist()}
                })

            recon_tile = recon_flat.reshape(th_pad, tw_pad, 3)
            win_full = hann2d(th_pad, tw_pad)[..., None]
            win = win_full if (th == th_pad and tw == tw_pad) else win_full[:th, :tw]

            out[y0:y1, x0:x1] += recon_tile[:th, :tw] * win
            acc[y0:y1, x0:x1] += win

            if debug:
                diff = (recon_tile[:th, :tw] - tile_img[:th, :tw])**2
                err_map[y0:y1, x0:x1] += diff.mean(axis=2)

            k += 1
            pbar.update(1)
    pbar.close()

    recon = out / np.clip(acc, 1e-8, None)
    return np.clip(recon, 0, 1), (err_map if debug else None), coeffs

def write_latex_tile(coeffs, k, channel, out_path):
    tiles = {t["k"]: t for t in coeffs["tiles"]}
    if k not in tiles:
        raise ValueError(f"Tile {k} not found (0..{len(tiles)-1})")
    t = tiles[k]
    names = coeffs["basis_names"]
    w = t["coeffs"][channel]

    pieces = []
    for coef, name in zip(w, names):
        # name already LaTeX-safe; coef numeric
        pieces.append(f"{coef:+.6f}\\,{name}")
    body = " + ".join(pieces)

    lines = [
        r"\[",
        rf"I^{{({channel})}}_{{k={k}}}(u,v) \;=\; {body},\quad u,v\in[-1,1]\,.",
        r"\]",
        "",
        r"% Tile placement in full image (pixels):",
        rf"% x0={t['x0']}, y0={t['y0']}, w={t['w']}, h={t['h']}",
        r"% Local coords: u = 2(x-x0)/(w-1)-1,  v = 2(y-y0)/(h-1)-1",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def write_latex_global(coeffs, out_path, expand=False):
    basis = coeffs["basis_names"]
    K = len(coeffs["tiles"])
    lines = [
        r"\section*{Global Reconstruction Formula}",
        r"Let the image domain be $\Omega=\{0,\dots,W-1\}\times\{0,\dots,H-1\}$.",
        rf"Here $W={coeffs['img_w']}$, $H={coeffs['img_h']}$, tile size $t={coeffs['tile']}$, overlap $\alpha={coeffs['overlap']}$.",
        r"We cover $\Omega$ with overlapping tiles $T_k$ (index $k=0,\dots," + str(K-1) + r"$), stride $s=\lfloor t(1-\alpha)\rfloor$.",
        r"For a pixel $(x,y)\in\Omega$ and tile $T_k$ with top-left $(x_k,y_k)$ and size $(w_k,h_k)$, define local coords:",
        r"\[ u_k(x)=\frac{2(x-x_k)}{w_k-1}-1,\qquad v_k(y)=\frac{2(y-y_k)}{h_k-1}-1. \]",
        r"Let $w_k(x,y)$ be a separable Hann window on $T_k$; we normalize by $\sum_k w_k(x,y)$.",
        r"Let $\{\phi_j\}_{j=1}^M$ be the analytic basis over $(u,v)\in[-1,1]^2$.",
        r"The final reconstructed image (per channel $c\in\{R,G,B\}$) is:",
        r"\paragraph{Mathtext-friendly:}",
        r"\[ \hat I^{(c)}(x,y) = \frac{\sum_k w_k(x,y)\,\sum_{j=1}^M a^{(c)}_{k,j}\,\phi_j(u_k(x),v_k(y))}{\sum_k w_k(x,y)} \]",        
        r"\paragraph{Full LaTeX:}"
        r"\[ \hat I^{(c)}(x,y) \;=\; \frac{\displaystyle \sum_{k} w_k(x,y)\;\sum_{j=1}^M a^{(c)}_{k,j}\,\phi_j\!\big(u_k(x),v_k(y)\big)}{\displaystyle \sum_{k} w_k(x,y)}\,. \]",
    ]
    if not expand:
        lines += [
            r"\paragraph{Basis order.} In our implementation the basis is:",
            r"\[ \{ " + ", ".join(basis) + r" \}\,.",
            r"\]",
            r"Numerical $a^{(c)}_{k,j}$ and tile placements are provided in the exported JSON.",
        ]
    else:
        # Warning: massive output on large images
        lines += [r"\section*{Expanded Numeric Form (illustrative, large)}"]
        for c in ["R","G","B"]:
            lines += [rf"\subsection*{{Channel {c}}}", r"\[ \hat I^{("+c+r")}(x,y)=\frac{1}{\sum_k w_k}\left("]
            for t in coeffs["tiles"]:
                coefs = t["coeffs"][c]
                inner = " + ".join([f"{coefs[j]:+.6f}\\,{basis[j]}" for j in range(len(basis))])
                lines += [r"w_k(x,y)\,\left(" + inner + r"\right) +"]
            lines += [r"\right) \]"]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# --------------------------
# Neural implicit (tiny MLP)
# --------------------------

def make_coords(W, H):
    x = np.linspace(-1, 1, W)
    y = np.linspace(-1, 1, H)
    X, Y = np.meshgrid(x, y)
    return X, Y

def fourier_features(xy, bands=6):
    # xy: (N,2) in [-1,1]
    freqs = 2**np.arange(bands, dtype=np.float32) * np.pi
    t = xy[..., None] * freqs  # (N,2,B)
    return np.concatenate([np.sin(t), np.cos(t)], axis=-1).reshape(xy.shape[0], -1)

class TinyMLP:
    def __init__(self, din, dh=96, dout=3, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.W1 = rng.normal(0, 0.5/np.sqrt(din), (din, dh)); self.b1 = np.zeros((dh,))
        self.W2 = rng.normal(0, 0.5/np.sqrt(dh), (dh, dh));   self.b2 = np.zeros((dh,))
        self.W3 = rng.normal(0, 0.5/np.sqrt(dh), (dh, dout)); self.b3 = np.zeros((dout,))
    def forward(self, X):
        H1 = np.tanh(X @ self.W1 + self.b1)
        H2 = np.tanh(H1 @ self.W2 + self.b2)
        Y  = 1/(1+np.exp(-(H2 @ self.W3 + self.b3)))  # sigmoid to [0,1]
        return Y, (X, H1, H2)
    def step(self, X, Yt, lr=5e-3):
        Yp, (Xc, H1, H2) = self.forward(X)
        dY = (Yp - Yt)
        dW3 = H2.T @ dY; db3 = dY.sum(axis=0)
        dH2 = dY @ self.W3.T * (1 - H2**2)
        dW2 = H1.T @ dH2; db2 = dH2.sum(axis=0)
        dH1 = dH2 @ self.W2.T * (1 - H1**2)
        dW1 = Xc.T @ dH1; db1 = dH1.sum(axis=0)
        # updates
        for P, G in [(self.W3,dW3),(self.b3,db3),(self.W2,dW2),(self.b2,db2),(self.W1,dW1),(self.b1,db1)]:
            P -= lr * G / X.shape[0]
        return float(np.mean((Yp - Yt)**2))

def bilinear_sample(img, Xs, Ys):
    # Xs, Ys in pixel coords
    H, W = img.shape[:2]
    x0 = np.clip(np.floor(Xs).astype(int), 0, W-1); x1 = np.clip(x0+1, 0, W-1)
    y0 = np.clip(np.floor(Ys).astype(int), 0, H-1); y1 = np.clip(y0+1, 0, H-1)
    wx = (Xs - x0)[:, None]; wy = (Ys - y0)[:, None]
    c00 = img[y0, x0]; c10 = img[y0, x1]
    c01 = img[y1, x0]; c11 = img[y1, x1]
    return (1-wx)*(1-wy)*c00 + wx*(1-wy)*c10 + (1-wx)*wy*c01 + wx*wy*c11

def neural_reconstruct(arr, train_side=256, iters=400, batch=8192, bands=6, lr=5e-3, seed=0, return_state=False):
    ...
    mlp = TinyMLP(din=feats.shape[1], dh=96, dout=3, rng=np.random.default_rng(seed))
    ...
    H, W = arr.shape[:2]
    scale = max(H, W) / train_side if max(H, W) > train_side else 1.0
    Ht, Wt = max(1, int(round(H/scale))), max(1, int(round(W/scale)))

    Xt, Yt = make_coords(Wt, Ht)
    xy = np.stack([Xt.ravel(), Yt.ravel()], axis=1).astype(np.float32)
    # Target colors from original full-res by bilinear sampling
    Xpix = (xy[:,0] + 1) * (W-1) / 2.0
    Ypix = (xy[:,1] + 1) * (H-1) / 2.0
    targets = bilinear_sample(arr, Xpix, Ypix).astype(np.float32)

    feats = np.concatenate([xy, fourier_features(xy, bands=bands)], axis=1)
    mlp = TinyMLP(din=feats.shape[1], dh=96, dout=3, rng=np.random.default_rng(seed))
    rng = np.random.default_rng(seed+1)
    N = feats.shape[0]

    pbar = tqdm(total=iters, desc="Neural train")
    for it in range(iters):
        idx = rng.integers(0, N, size=(min(batch, N),))
        loss = mlp.step(feats[idx], targets[idx], lr=lr)
        if (it+1) % 50 == 0:
            pbar.set_postfix({"mse": f"{loss:.5f}"})
        pbar.update(1)
    pbar.close()

    # Render at full resolution
    Xf, Yf = make_coords(W, H)
    xy_full = np.stack([Xf.ravel(), Yf.ravel()], axis=1).astype(np.float32)
    feats_full = np.concatenate([xy_full, fourier_features(xy_full, bands=bands)], axis=1)
    Yfull, _ = mlp.forward(feats_full)
    recon = Yfull.reshape(H, W, 3)
    if return_state:
        state = {"W1": mlp.W1, "b1": mlp.b1, "W2": mlp.W2, "b2": mlp.b2, "W3": mlp.W3, "b3": mlp.b3,
                 "din": feats.shape[1], "dh": mlp.W1.shape[1]}
        return np.clip(recon, 0, 1), state
    return np.clip(recon, 0, 1)

# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="Image → Math: Analytic (poly+trig tiles) and Neural implicit")
    ap.add_argument("image", nargs="?", default="input.jpg", help="input image path (default: input.jpg)")
    ap.add_argument("--method", choices=["analytic","nn","both"], default=None,
                    help="which method to run (default: analytic for no args, else analytic)")

    # Common
    ap.add_argument("--res", type=parse_res, default=None, help="resize to WxH (e.g., 1024x1024)")
    ap.add_argument("--out", default="reconstruction_tiled.png", help="output image for analytic/both")
    ap.add_argument("--out-nn", default="reconstruction_nn.png", help="output image for neural")
    ap.add_argument("--no-gui", action="store_true", help="don’t call plt.show()")
    ap.add_argument("--debug", action="store_true", help="save tile grid + error heatmap (analytic only)")

    # Analytic knobs
    ap.add_argument("--detail", choices=["xlow","low","med","high","xhigh"], default=None,
                    help="analytic preset (default xhigh if no-args, else med)")
    ap.add_argument("--tile", type=int, default=None)
    ap.add_argument("--overlap", type=float, default=None)
    ap.add_argument("--poly", type=int, default=None)
    ap.add_argument("--freq", type=int, default=None)
    ap.add_argument("--l2", type=float, default=None)
    ap.add_argument("--export-json", default=None, help="save analytic coefficients + metadata as JSON")
    ap.add_argument("--final-latex", default=None, help="path for global analytic formula LaTeX")
    ap.add_argument("--expand-global", action="store_true", help="inline all coeffs in global LaTeX (HUGE)")
    ap.add_argument("--latex-tile", type=int, default=None, help="emit LaTeX for a specific tile index")
    ap.add_argument("--channel", choices=["R","G","B"], default="R", help="channel for --latex-tile")

    # Neural knobs
    ap.add_argument("--nn-train-side", type=int, default=256, help="max train grid side")
    ap.add_argument("--nn-iters", type=int, default=400, help="training iterations")
    ap.add_argument("--nn-bands", type=int, default=6, help="positional encoding bands")
    ap.add_argument("--nn-batch", type=int, default=8192, help="batch size")
    ap.add_argument("--nn-lr", type=float, default=5e-3, help="learning rate")
    ap.add_argument("--nn-seed", type=int, default=0, help="random seed")

    # Neural export
    ap.add_argument("--export-nn", default=None, help="save neural implicit weights/meta to NPZ")


    args = ap.parse_args()
    no_args_run = (len(sys.argv) == 1)

    # Defaults
    if args.method is None:
        args.method = "analytic"  # default
    if args.detail is None:
        args.detail = "xhigh" if no_args_run else "med"
    if args.export_json is None and no_args_run:
        args.export_json = "debug/coeffs.json"
    if args.final_latex is None and no_args_run:
        args.final_latex = "debug/final_formula.tex"

    # Load image
    img = load_img_or_die(args.image, args.res)
    arr = np.asarray(img, dtype=np.float64) / 255.0

    # Run methods
    ran_any = False

    if args.method in ("analytic", "both"):
        TILE, OVERLAP, POLY, FREQ, L2 = PRESETS[args.detail]
        if args.tile is not None: TILE = args.tile
        if args.overlap is not None: OVERLAP = args.overlap
        if args.poly is not None: POLY = args.poly
        if args.freq is not None: FREQ = args.freq
        if args.l2 is not None: L2 = args.l2
        print(f"Preset={args.detail} → tile={TILE}, overlap={OVERLAP}, poly={POLY}, freq={FREQ}, l2={L2}")

        collect = bool(args.export_json or args.final_latex or args.latex_tile is not None)
        recon, errmap, coeffs = reconstruct_tiled_analytic(
            arr, TILE, OVERLAP, POLY, FREQ, L2, collect_coeffs=collect, debug=args.debug
        )

        save_img(arr, "Original", os.path.join(os.path.dirname(args.out) or ".", "original.png"), args.no_gui)
        save_img(recon, "Analytic (poly+trig tiled)", args.out, args.no_gui)
        # --- Side-by-side viewer (Original | Analytic Recon | Formula) ---
        # note: for analytic, coeffs is available when you requested any export
        # if not, we still pass sizes/tile/overlap so the formula shows meaningful metadata.
        meta = helper.AnalyticMeta(
            img_w=arr.shape[1], img_h=arr.shape[0],
            tile=TILE, overlap=OVERLAP,
            basis_names=(coeffs["basis_names"] if coeffs and "basis_names" in coeffs else None)
        )
        helper.show_triptych(
            arr, recon, meta,
            mid_title="Analytic (poly+trig tiled)",
            no_gui=args.no_gui
        )

        print(f"Saved {args.out}")
        ran_any = True

        # Debug extras
        if args.debug:
            os.makedirs("debug", exist_ok=True)
            grid = draw_tile_grid(img, TILE, OVERLAP); grid.save("debug/tile_grid_overlay.png")
            if errmap is not None:
                save_img(errmap, "Per-tile error (normalized)", "debug/error_heatmap.png", args.no_gui)
            print("Saved debug/tile_grid_overlay.png" + (" and debug/error_heatmap.png" if errmap is not None else ""))

        # Exports
        if args.export_json:
            os.makedirs(os.path.dirname(args.export_json) or ".", exist_ok=True)
            with open(args.export_json, "w", encoding="utf-8") as f:
                json.dump(coeffs, f, ensure_ascii=False, indent=2)
            print(f"Saved coefficients JSON → {args.export_json}")

        if args.latex_tile is not None:
            os.makedirs("debug", exist_ok=True)
            out_tex = f"debug/tile_{args.latex_tile}_{args.channel}.tex"
            write_latex_tile(coeffs, args.latex_tile, args.channel, out_tex)
            print(f"Wrote LaTeX for tile {args.latex_tile} channel {args.channel} → {out_tex}")

        if args.final_latex:
            os.makedirs(os.path.dirname(args.final_latex) or ".", exist_ok=True)
            write_latex_global(coeffs, args.final_latex, expand=args.expand_global)
            print(f"Wrote global LaTeX formula → {args.final_latex}")



    if args.method in ("nn", "both"):
        t0 = time.time()
        # Train
        recon_nn, mlp_state = neural_reconstruct(
            arr,
            train_side=args.nn_train_side,
            iters=args.nn_iters,
            batch=args.nn_batch,
            bands=args.nn_bands,
            lr=args.nn_lr,
            seed=args.nn_seed,
            return_state=True  # NEW
        )
        print(f"Neural finished in {time.time()-t0:.1f}s")
        save_img(recon_nn, "Neural implicit (full-res render)", args.out_nn, args.no_gui)
        meta = helper.NNMeta(
            bands=args.nn_bands,
            din=mlp_state.get("din"), dh=mlp_state.get("dh")
        )
        helper.show_triptych(
            arr, recon_nn, meta,
            mid_title="Neural implicit (full-res render)",
            no_gui=args.no_gui
        )

        print(f"Saved {args.out_nn}")

        # Export weights/meta
        if args.export_nn:
            os.makedirs(os.path.dirname(args.export_nn) or ".", exist_ok=True)
            np.savez(
                args.export_nn,
                W1=mlp_state["W1"], b1=mlp_state["b1"],
                W2=mlp_state["W2"], b2=mlp_state["b2"],
                W3=mlp_state["W3"], b3=mlp_state["b3"],
                bands=np.array([args.nn_bands], dtype=np.int32),
                din=np.array([mlp_state["din"]], dtype=np.int32),
                dh=np.array([mlp_state["dh"]], dtype=np.int32),
                dout=np.array([3], dtype=np.int32),
                img_w=np.array([arr.shape[1]], dtype=np.int32),
                img_h=np.array([arr.shape[0]], dtype=np.int32),
                note="coords=(x,y) in [-1,1]^2; features=[xy, sin/cos(2^k*pi*xy)]"
            )
            print(f"Saved neural weights/meta → {args.export_nn}")


    if not ran_any:
        print("Nothing ran. Check --method.", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main()
