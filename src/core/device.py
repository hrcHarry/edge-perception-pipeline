import os
try:
    import torch
except Exception:
    torch = None

def torch_device():
    if torch is None:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def ort_providers():
    # CUDA → CPU fallback；環境變數 ORT_CUDA_OFF=1 可強制走 CPU
    off = os.environ.get("ORT_CUDA_OFF", "0") == "1"
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        avail = ["CPUExecutionProvider"]
    wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if off:
        return ["CPUExecutionProvider"] if "CPUExecutionProvider" in avail else []
    return [p for p in wanted if p in avail] or ["CPUExecutionProvider"]

