import os
import numpy as np
import torch
import torch.nn.functional as F
import onnx, onnxruntime as ort

from src.core.models import SimpleCNN

DEVICE = torch.device("cpu")
PT_WEIGHTS = "models/fer2013_cnn.pth"
ONNX_PATH   = "models/fer2013_cnn.onnx"

def load_model():
    model = SimpleCNN(num_classes=7).to(DEVICE)
    state = torch.load(PT_WEIGHTS, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def export_onnx(model):
    os.makedirs("models", exist_ok=True)
    dummy = torch.randn(1, 1, 48, 48, device=DEVICE)  # (N,C,H,W)
    torch.onnx.export(
        model, dummy, ONNX_PATH,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
        opset_version=13, do_constant_folding=True
    )
    onnx.checker.check_model(onnx.load(ONNX_PATH))
    print(f"Exported: {ONNX_PATH}")

def verify_equivalence(model):
    x = torch.randn(8, 1, 48, 48, device=DEVICE)
    with torch.no_grad():
        pt_logits = model(x).cpu().numpy()

    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_logits = sess.run(["logits"], {"input": x.cpu().numpy()})[0]

    max_abs = np.max(np.abs(pt_logits - ort_logits))
    pt_prob  = torch.softmax(torch.from_numpy(pt_logits), dim=1).numpy()
    ort_prob = torch.softmax(torch.from_numpy(ort_logits), dim=1).numpy()
    max_abs_prob = np.max(np.abs(pt_prob - ort_prob))

    print(f"Max |logits_pt - logits_ort| = {max_abs:.3e}")
    print(f"Max |prob_pt - prob_ort|     = {max_abs_prob:.3e}")
    ok = (max_abs < 1e-4) and (max_abs_prob < 1e-5)
    print("Verification:", "OK" if ok else "CHECK THRESHOLDS")
    return ok

if __name__ == "__main__":
    model = load_model()
    export_onnx(model)
    verify_equivalence(model)

