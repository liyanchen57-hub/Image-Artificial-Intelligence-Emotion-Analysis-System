from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.web_app import WebConfig, serve_web_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the web realtime emotion dashboard.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/full_run_resnet_cbam_mps/checkpoints/best_model.pth",
    )
    parser.add_argument("--model", type=str, default="resnet_cbam", choices=["cnn_cbam", "resnet_cbam"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--frame-width", type=int, default=960)
    parser.add_argument("--frame-height", type=int, default=540)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument("--history-length", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    config = WebConfig(
        camera=args.camera,
        checkpoint=checkpoint,
        model=args.model,
        device=args.device,
        host=args.host,
        port=args.port,
        min_detection_confidence=args.min_detection_confidence,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        jpeg_quality=args.jpeg_quality,
        history_length=args.history_length,
    )
    serve_web_app(config)


if __name__ == "__main__":
    main()
