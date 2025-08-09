# viewer.py
# Streamlit viewer for Original / Analytic(JSON) / Neural(NPZ)
# - Load original image (file upload)
# - Load analytic JSON (from image-to-math.py --export-json)
# - Load neural NPZ (from image-to-math.py --export-nn)
# - Set output resolution; render and compare
# - Optional: compute analytic from image directly (slow), with presets

import io, os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st

st.set_page_config(page_title="Image ⇄ Math Viewer", layout="wide")

# ---------------------------
# Helpers (math-to-image)
# ---------------------------
def parse_wh(s):
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        return None

def hann2d(h, w):
    hx = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(w)/(w-1))
    hy = 0.5 - 0.5 * np.cos(2*np.pi*np.arange(h)/(h-1))
    return np.outer(hy, hx)

def eval_basis_names(U, V, names):
    terms = []
    for name in names:
        if name == "1":
            terms.append(np.ones_like(U)); continue
        if ("u^" in name) or ("v^" in name):
            ui, vj = 0, 0
            # names are like "u^2 v^1" or "u^2" or "v^1"
            toks = name.split()
            for tok in toks:
                if tok.startswith("u^"):
                    ui = int(tok[2:])
                if tok.startswith("v^"):
                    vj = int(tok[2:])
            terms.append((U**ui) * (V**vj)); continue
        # trig forms using \pi:
        term = np.ones_like(U)
        if "\\sin(" in name and "u)" in name.split("\\sin(")[1]:
            ku = int("".join(ch for ch in name.split("\\sin(")[1].split("\\pi")[0] if ch.isdigit()) or "1")
            term = term * np.sin(np.pi * ku * U)
        if "\\cos(" in name and "u)" in name.split("\\cos(")[1]:
            ku = int("".join(ch for ch in name.split("\\cos(")[1].split("\\pi")[0] if ch.isdigit()) or "1")
            term = term * np.cos(np.pi * ku * U)
        if ")\\cos(" in name and "v)" in name.split(")\\cos(")[1]:
            kv = int("".join(ch for ch in name.split(")\\cos(")[1].split("\\pi")[0] if ch.isdigit()) or "1")
            term = term * np.cos(np.pi * kv * V)
        if ")\\sin(" in name and "v)" in name.split(")\\sin(")[1]:
            kv = int("".join(ch for ch in name.split(")\\sin(")[1].split("\\pi")[0] if ch.isdigit()) or "1")
            term = term * np.sin(np.pi * kv * V)
        terms.append(term)
    return np.stack([t.ravel() for t in terms], axis=0)

def render_from_analytic_json(json_obj, out_wh=None):
    data = json_obj
    W0, H0 = data["img_w"], data["img_h"]
    if out_wh is None:
        W, H = W0, H0
    else:
        W, H = out_wh

    recon = np.zeros((H, W, 3), dtype=np.float64)
    acc   = np.zeros((H, W, 1), dtype=np.float64)

    for tile in data["tiles"]:
        x0, y0, tw, th = tile["x0"], tile["y0"], tile["w"], tile["h"]
        sx = W / W0; sy = H / H0
        x0n = int(round(x0 * sx)); y0n = int(round(y0 * sy))
        twn = max(1, int(round(tw * sx))); thn = max(1, int(round(th * sy)))

        u = np.linspace(-1, 1, twn); v = np.linspace(-1, 1, thn)
        U, V = np.meshgrid(u, v)
        T = eval_basis_names(U, V, data["basis_names"])

        tile_out = np.zeros((thn*twn, 3), dtype=np.float64)
        for ci, ch in enumerate(("R","G","B")):
            w = np.array(tile["coeffs"][ch], dtype=np.float64)
            tile_out[:, ci] = w @ T
        tile_out = tile_out.reshape(thn, twn, 3)

        win = hann2d(thn, twn)[..., None]
        y1, x1 = min(y0n+thn, H), min(x0n+twn, W)
        hclip, wclip = (y1-y0n), (x1-x0n)
        recon[y0n:y1, x0n:x1] += tile_out[:hclip, :wclip] * win[:hclip, :wclip]
        acc[y0n:y1, x0n:x1]   += win[:hclip, :wclip]

    recon = recon / np.clip(acc, 1e-8, None)
    return np.clip(recon, 0, 1)

def mlp_forward_np(X, W1,b1, W2,b2, W3,b3):
    H1 = np.tanh(X @ W1 + b1)
    H2 = np.tanh(H1 @ W2 + b2)
    Y  = 1/(1+np.exp(-(H2 @ W3 + b3)))
    return Y

def fourier_features_xy(xy, bands):
    freqs = 2**np.arange(bands, dtype=np.float32) * np.pi
    t = xy[..., None] * freqs
    return np.concatenate([np.sin(t), np.cos(t)], axis=-1).reshape(xy.shape[0], -1)

def render_from_nn_npz(npz_bytes, out_wh):
    import numpy as np
    with np.load(io.BytesIO(npz_bytes), allow_pickle=True) as Z:
        W1, b1, W2, b2, W3, b3 = Z["W1"], Z["b1"], Z["W2"], Z["b2"], Z["W3"], Z["b3"]
        bands = int(Z["bands"][0])
    W, H = out_wh
    x = np.linspace(-1, 1, W); y = np.linspace(-1, 1, H)
    Xg, Yg = np.meshgrid(x, y)
    xy = np.stack([Xg.ravel(), Yg.ravel()], axis=1).astype(np.float32)
    feats = np.concatenate([xy, fourier_features_xy(xy, bands=bands)], axis=1)
    Y = mlp_forward_np(feats, W1,b1, W2,b2, W3,b3)
    return np.clip(Y.reshape(H, W, 3), 0, 1)

# ---------------------------
# UI
# ---------------------------
st.title("🧮 Image ⇄ Math Viewer")

with st.sidebar:
    st.header("Inputs")
    up_img = st.file_uploader("Original image (optional, for reference)", type=["png","jpg","jpeg"])
    up_json = st.file_uploader("Analytic JSON (coeffs.json)", type=["json"])
    up_npz  = st.file_uploader("Neural NPZ (nn_weights.npz)", type=["npz"])
    res_str = st.text_input("Output resolution", value="1024x1024")
    out_wh = parse_wh(res_str) or (1024, 1024)

    st.header("Compute Analytic (optional)")
    compute_from_image = st.checkbox("Compute analytic from image (slow)")
    preset = st.selectbox("Preset", ["xlow","low","med","high","xhigh"], index=4)
    run_button = st.button("Render")

# Load original preview
orig_img = None
if up_img is not None:
    orig_img = Image.open(up_img).convert("RGB")
    orig_img = ImageOps.exif_transpose(orig_img)

col1, col2, col3 = st.columns(3)

if run_button:
    # Column 1: Original
    with col1:
        st.subheader("Original")
        if orig_img is not None:
            st.image(orig_img, use_column_width=True)
        else:
            st.info("No original provided.")

    # Column 2: Analytic
    with col2:
        st.subheader("Analytic")
        if up_json is not None:
            import json as _json
            data = _json.load(io.TextIOWrapper(up_json, encoding="utf-8"))
            aa = render_from_analytic_json(data, out_wh=out_wh)
            st.image((aa*255).astype(np.uint8), use_column_width=True)
            st.download_button("Download Analytic PNG", data=Image.fromarray((aa*255).astype(np.uint8)).to_bytes() if hasattr(Image.Image, "to_bytes") else None)
        elif compute_from_image and orig_img is not None:
            # On-the-fly compute (quick-n-dirty): small wrapper around your CLI logic with defaults.
            # For Streamlit, we keep it simple: use a medium preset (costly otherwise).
            st.warning("Computing analytic from the image inside Streamlit can be slow. This is a minimal demo.")
            # Minimal in-app compute: single global basis (fast-ish but blurrier).
            # Encourage users to use CLI for tiled high quality.
            img_resized = orig_img.resize(out_wh, Image.LANCZOS)
            arr = np.asarray(img_resized, dtype=np.float64)/255.0
            H, W = arr.shape[:2]
            u = np.linspace(-1,1,W); v = np.linspace(-1,1,H)
            U,V = np.meshgrid(u,v)
            # simple basis (poly=2, freq=8) for demo
            def build_basis_demo(U,V, poly=2, freq=8):
                terms=[]; names=[]
                for i in range(poly+1):
                    for j in range(poly+1-i):
                        terms.append((U**i)*(V**j)); names.append("1" if (i==0 and j==0) else f"u^{i} v^{j}")
                for k in range(1,freq+1):
                    su,cu=np.sin(np.pi*k*U),np.cos(np.pi*k*U)
                    sv,cv=np.sin(np.pi*k*V),np.cos(np.pi*k*V)
                    terms += [su,cu,sv,cv,su*cv,cu*sv]
                    names += [f"\\sin({k}\\pi u)",f"\\cos({k}\\pi u)",f"\\sin({k}\\pi v)",f"\\cos({k}\\pi v)",f"\\sin({k}\\pi u)\\cos({k}\\pi v)",f"\\cos({k}\\pi u)\\sin({k}\\pi v)"]
                T = np.stack([t.ravel() for t in terms], axis=0)
                return T
            T = build_basis_demo(U,V, poly=2, freq=8)
            def ridge(T, y, l2=1e-3):
                A = T @ T.T; b = T @ y.ravel()
                A += l2*np.eye(T.shape[0]); return np.linalg.solve(A,b)
            rec = np.zeros_like(arr)
            for c in range(3):
                w = ridge(T, arr[...,c], l2=1e-3)
                rec[...,c] = (w @ T).reshape(H,W)
            aa = np.clip(rec,0,1)
            st.image((aa*255).astype(np.uint8), use_column_width=True)
        else:
            st.info("Drop an Analytic JSON or enable compute-from-image.")

    # Column 3: Neural
    with col3:
        st.subheader("Neural")
        if up_npz is not None:
            npz_bytes = up_npz.read()
            nn_img = render_from_nn_npz(npz_bytes, out_wh=out_wh)
            st.image((nn_img*255).astype(np.uint8), use_column_width=True)
        else:
            st.info("Drop a Neural NPZ to view.")

else:
    st.info("📥 Upload an Analytic JSON and/or Neural NPZ, set resolution, then click **Render**.")
    if orig_img is not None:
        st.image(orig_img, caption="Original (preview)", use_column_width=True)
