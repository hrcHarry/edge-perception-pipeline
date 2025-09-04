import argparse, os, csv, glob
import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from torchvision import transforms

TFM = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),        # (C,H,W) in [0,1]
])

def load_session(onnx_path: str):
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

def infer_one(sess, img_path: str):
    img = Image.open(img_path).convert("L")
    x = TFM(img).unsqueeze(0).numpy()          # (1,1,48,48)
    logits = sess.run(["logits"], {"input": x})[0]  # (1,C)
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()[0]  # (C,)
    top1 = int(probs.argmax())
    conf = float(probs[top1])
    return top1, conf, probs

def write_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "top1", "confidence"])
        w.writerows(rows)

def load_class_names(path: str | None):
    if not path: return None
    with open(path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f if ln.strip()]
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/fer2013_cnn.onnx")
    ap.add_argument("--image", help="單張圖片路徑")
    ap.add_argument("--folder", help="資料夾，遞迴讀取常見影像副檔名")
    ap.add_argument("--classes", help="類別名稱檔，每行一個名稱（可選）")
    ap.add_argument("--out", default="results/infer_onnx.csv")
    args = ap.parse_args()

    sess = load_session(args.onnx)
    class_names = load_class_names(args.classes)

    targets = []
    if args.image:
        targets.append(args.image)
    if args.folder:
        exts = ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]
        for ext in exts:
            targets += glob.glob(os.path.join(args.folder, "**", ext), recursive=True)
    if not targets:
        raise SystemExit("請提供 --image 或 --folder")

    rows = []
    for p in sorted(targets):
        try:
            top1, conf, _ = infer_one(sess, p)
            label = class_names[top1] if class_names and top1 < len(class_names) else str(top1)
            print(f"{p} -> {label} ({conf:.4f})")
            rows.append([p, label, f"{conf:.6f}"])
        except Exception as e:
            print(f"[ERR] {p}: {e}")

    write_csv(rows, args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()

