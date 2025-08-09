# helper.py
# Shared viewer + formula helpers for image-to-math.py and math-to-image.py

from dataclasses import dataclass
from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams.update({"text.usetex": False})
import textwrap
# ---------- Meta types ----------

@dataclass
class AnalyticMeta:
    img_w: int
    img_h: int
    tile: int
    overlap: float
    basis_names: Optional[Sequence[str]] = None  # for M

@dataclass
class NNMeta:
    bands: int
    din: Optional[int] = None
    dh: Optional[int]  = None

# ---------- Utilities ----------

def _ensure_float01(arr):
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.dtype.kind in "ui":
        a = a.astype(np.float32) / 255.0
    return np.clip(a, 0.0, 1.0)

def save_img(img_arr, out_path, *, title=None, no_gui=False):
    """Consistent saver used by both CLIs."""
    import os
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig = plt.figure(figsize=(6, 6), dpi=140)
    ax = fig.add_subplot(111)
    ax.imshow(_ensure_float01(img_arr))
    ax.axis("off")
    if title: ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    if not no_gui:
        plt.show()
    plt.close(fig)

def prepare_original(path, res=None):
    """Load optional original image (PIL if available in caller)."""
    if not path:
        return None
    from PIL import Image, ImageOps
    im = Image.open(path).convert("RGB")
    im = ImageOps.exif_transpose(im)
    if res:
        im = im.resize(res, Image.LANCZOS)
    return np.asarray(im, dtype=np.float32) / 255.0

# ---------- Formula formatting (single dispatch) ----------

mpl.rcParams.update({
    "text.usetex": False,          # force mathtext (no external LaTeX)
    "mathtext.default": "regular", # simpler glyphs
})

def _format_analytic_formula(meta: AnalyticMeta) -> str:
    M = len(meta.basis_names) if (meta.basis_names is not None) else "M"
    line1 = (
        r"$\hat I^{(c)}(x,y)=\frac{\sum_k w_k(x,y)\sum_{j=1}^{%s}"
        r" a^{(c)}_{k,j}\phi_j(u_k(x),v_k(y))}{\sum_k w_k(x,y)}$" % M
    )
    line2 = r"$u_k(x)=\frac{2(x-x_k)}{w_k-1}-1,\quad v_k(y)=\frac{2(y-y_k)}{h_k-1}-1$"
    line3 = rf"$H={meta.img_h},\ W={meta.img_w};\ t={meta.tile},\ \alpha={meta.overlap}$"
    return "\n".join([line1, line2, line3])

def _format_nn_formula(meta: NNMeta) -> str:
    header = f"MLP: din={meta.din}, dh={meta.dh}, dout=3" if (meta.din or meta.dh) else None
    core = (
        r"$\hat I(x,y)=\sigma\!\big(f_3(\tanh(f_2(\tanh(f_1(z)))))\big)$" "\n"
        r"$z=[x,y,\{\sin(2^k\pi x),\cos(2^k\pi x),\sin(2^k\pi y),\cos(2^k\pi y)\}_{k=0}^{%d}]$" % (meta.bands-1)
    )
    return (header + "\n" + core) if header else core

def format_formula(meta) -> str:
    if isinstance(meta, AnalyticMeta):
        return _format_analytic_formula(meta)
    if isinstance(meta, NNMeta):
        return _format_nn_formula(meta)
    raise TypeError("Unsupported meta type for formula formatting.")

# ---------- Unified viewer ----------

def show_triptych(original_arr, recon_arr, meta, *, mid_title="Reconstruction", no_gui=False):
    """Single UI: [ Original | Reconstruction | Formula ]"""
    if no_gui:
        return
    orig = _ensure_float01(original_arr) if original_arr is not None else None
    recon = _ensure_float01(recon_arr)
    formula_text = format_formula(meta)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=140)

    # Original
    axes[0].axis("off")
    if orig is not None:
        axes[0].imshow(orig)
        axes[0].set_title("Original")
    else:
        axes[0].text(0.5, 0.5, "Original not provided", ha="center", va="center")
        axes[0].set_title("Original")

    # Reconstruction
    axes[1].imshow(recon); axes[1].set_title(mid_title); axes[1].axis("off")

    # Formula panel
    axes[2].axis("off")
    wrapped = []
    for line in formula_text.split("\n"):
        # If the line contains any math, don't rewrap it—let mathtext handle it.
        if "$" in line:
            wrapped.append(line)
        else:
            wrapped.append(textwrap.fill(line, width=54))
    axes[2].text(
        0.02, 0.98, "\n".join(wrapped),
        va="top", ha="left", transform=axes[2].transAxes,
        wrap=False,                     #  don't set to True, it breaks $...$
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.4", fc="w", ec="0.85"),
    )
    axes[2].set_title("Formula")

    fig.tight_layout()
    plt.show()
    plt.close(fig)
