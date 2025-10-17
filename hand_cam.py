import argparse
import os
import sys
from typing import Optional

import cv2
from ultralytics import YOLO

try:
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover
    hf_hub_download = None


DEFAULT_CAMERA_INDEX = 0


def try_download_hand_weights(prefer_size: str = "n") -> Optional[str]:
    """
    Attempt to download a YOLOv8 hand-detection checkpoint from Hugging Face.

    This tries a few known community repositories. If none succeed, returns None.
    """
    if hf_hub_download is None:
        return None

    # Candidate repos ordered by expected model size preference.
    small_first = [
        "keremberke/yolov8n-hand-detection",
        "keremberke/yolov8s-hand-detection",
    ]
    larger = [
        "keremberke/yolov8m-hand-detection",
        "keremberke/yolov8l-hand-detection",
    ]

    repo_candidates = (small_first + larger) if prefer_size in {"n", "s"} else (larger + small_first)

    # Common file name patterns seen across community YOLO repos
    filename_candidates = [
        "best.pt",
        "weights/best.pt",
        "yolov8n-hand-detection.pt",
        "yolov8s-hand-detection.pt",
        "yolov8m-hand-detection.pt",
        "yolov8l-hand-detection.pt",
    ]

    for repo_id in repo_candidates:
        for filename in filename_candidates:
            try:
                local_path = hf_hub_download(repo_id=repo_id, filename=filename)
                if os.path.isfile(local_path):
                    return local_path
            except Exception:
                continue

    return None


def load_model(weights_path: Optional[str], prefer_size: str = "n") -> YOLO:
    """
    Load YOLO model. If weights_path is None, try to auto-download a hand detector.
    As a last resort, load the generic COCO model (may not detect hands reliably).
    """
    if weights_path and os.path.isfile(weights_path):
        return YOLO(weights_path)

    # Try auto-download
    downloaded = try_download_hand_weights(prefer_size=prefer_size)
    if downloaded is not None:
        return YOLO(downloaded)

    # Fallback to generic YOLOv8n (COCO). This likely won't detect hands as a class,
    # but keeps the demo running. Users should provide/fine-tune a hand model.
    return YOLO("yolov8n.pt")


def draw_detections(frame, result, names):
    if not hasattr(result, "boxes") or result.boxes is None:
        return frame

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()

    for (x1, y1, x2, y2), score, cls_id in zip(boxes_xyxy, scores, class_ids):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        class_id = int(cls_id)
        label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)

        # If the model is a hand-specific model, label likely is 'hand'.
        # Otherwise, display whatever label is available.
        color = (0, 255, 0) if label.lower() == "hand" else (255, 165, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{label}: {score:.2f}",
            (x1, max(0, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time hand detection with YOLOv8")
    parser.add_argument("--source", type=int, default=DEFAULT_CAMERA_INDEX, help="Webcam index (default: 0)")
    parser.add_argument("--weights", type=str, default="", help="Path to hand model .pt (optional)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--pref", type=str, default="n", choices=["n", "s", "m", "l"], help="Preferred model size for auto-download")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="", help="Device id (e.g. '0' or 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()

    model = load_model(args.weights if args.weights else None, prefer_size=args.pref)
    # Override model settings
    model.overrides["conf"] = float(args.conf)
    if args.device:
        model.to(args.device)

    # Try to set a reasonable resolution for the webcam
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("[ERROR] Unable to open webcam. Check the index or permissions.")
        sys.exit(1)

    # Optional: try to increase resolution (may be ignored by some cameras)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Firedect - YOLOv8 Hand Detection (press 'q' to quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Failed to read frame from camera.")
                break

            # Run inference; stream=False returns a list of results
            results = model.predict(source=frame, imgsz=args.imgsz, verbose=False)
            vis_frame = frame.copy()

            if results:
                r0 = results[0]
                names = getattr(model, "names", {})
                vis_frame = draw_detections(vis_frame, r0, names)

            # If the model is likely not hand-specific, hint the user
            if getattr(model, "names", {}).get(0, "").lower() != "hand":
                cv2.putText(
                    vis_frame,
                    "Model may not be hand-specific. Provide --weights for better results.",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


