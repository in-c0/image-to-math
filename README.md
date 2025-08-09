# Image ⇄ Math (Piecewise Polynomial + Sinusoid Approximation)

This is Stage 1 of (*Link).

Run image-to-math.py to convert an image into a mathematical description; using a sum of low-order polynomials and sinusoids on tiled patches, then reconstructing it back from that math.



## Installation

### CPU
pip install -r requirements.txt

### or: GPU (NVIDIA, CUDA 11.8)
pip install -r requirements-gpu.txt


## Run

### Analytic/NN reconstruction + LaTeX/JSON export
python image-to-math.py  input.jpg  --method analytic
python image-to-math.py  input.jpg  --method nn
python image-to-math.py  input.jpg  --method both

### Re-render from saved math (JSON/NPZ)
python math-to-image.py --analytic debug/coeffs.json
python math-to-image.py --nn debug/nn_weights.npz

### Viewer (WIP)
pip install streamlit numpy pillow matplotlib
streamlit run viewer.py