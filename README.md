# CPU
pip install -r requirements.txt

# or: GPU (NVIDIA, CUDA 11.8)
pip install -r requirements-gpu.txt


Run:

# Analytic/NN reconstruction + LaTeX/JSON export
python image-to-math.py  input.jpg  --method analytic
python image-to-math.py  input.jpg  --method nn
python image-to-math.py  input.jpg  --method both

# Re-render from saved math (JSON/NPZ)
python math-to-image.py --analytic debug/coeffs.json
python math-to-image.py --nn debug/nn_weights.npz

# Viewer
pip install streamlit numpy pillow matplotlib
streamlit run viewer.py
