# Edge-ready Perception Pipeline

PyTorch → ONNX → FastAPI → Docker. FER2013 baseline + robustness + Kalman smoothing + ONNXRuntime API.

A modular perception pipeline designed for **robust inference under noisy conditions**.
The project demonstrates end-to-end ML deployment workflows, from ONNX model inference to containerized API services, with Kalman-based post-processing for stability.

## Features
- **Model Inference**
  - Exported PyTorch model to ONNX and executed inference via ONNX Runtime.
  - Simulated noisy inputs (Gaussian blur, occlusion, variance shifts).

- **Post-processing**
  - Applied **Kalman smoothing** to stabilize predictions under degraded conditions.
  - Quantified robustness improvements with spectral/variance analysis and condition number regularization.

- **Deployment**
  - Containerized services with **Docker** (CPU/GPU modes).
  - Lightweight REST API using **FastAPI** for easy evaluation.

- **Profiling & Stability**
  - Latency and resource profiling for inference pipelines.
  - Robust accuracy curves under varying noise levels.


### Robustness Evaluation

We tested the baseline CNN model under several perturbations.
Dataset: FER2013 (or FakeData for pipeline validation).

| Scenario      | Accuracy |
|---------------|----------|
| clean         | 0.51     |
| gaussian_0.10 | 0.40     |
| blur_k5       | 0.45     |
| occlusion_25  | 0.30     |
| bright_1.4    | 0.49     |

![Robustness Results](results/robust_bar.png)

### Kalman Smoothing

Applied a simple 1D Kalman filter on the class probability sequence.
- Raw accuracy: 0.51
- Smoothed accuracy: 0.74

![Kalman smoothing trace](results/smooth_trace.png)

### ONNX Export

Exported PyTorch model to ONNX:
- File: `models/fer2013_cnn.onnx`
- Verified with onnxruntime (max error < 1e-5)

Test inference results saved in:
- `results/infer_onnx.csv`


## Quick Start
```bash
pip install -r requirements.txt
uvicorn src.train.infer_api:app --host 0.0.0.0 --port 8080
# curl -F "file=@/path/to/img.jpg" http://127.0.0.1:8080/infer_form
```
