import os
import argparse
import time
import cv2
import numpy as np

from inference import RGBTInference


def draw_detections_on_frame(frame_bgr: np.ndarray, detections, class_names, color=(0, 0, 255)) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    for det in detections:
        bbox = det['bbox']
        confidence = det['confidence']
        class_id = det['class_id']
        temperature = det['temperature']

        x_center = bbox[0] * w
        y_center = bbox[1] * h
        width = bbox[2] * w
        height = bbox[3] * h

        x1 = int(max(0, x_center - width / 2))
        y1 = int(max(0, y_center - height / 2))
        x2 = int(min(w - 1, x1 + width))
        y2 = int(min(h - 1, y1 + height))

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        class_name = class_names[class_id] if class_id < len(class_names) else f'Class {class_id}'
        label = f'{class_name} {confidence:.2f} | {temperature:.1f}C'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), (255, 255, 255), -1)
        cv2.putText(frame_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    return frame_bgr


def open_video_source(src: str | int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(src)
    return cap


def parse_source(arg: str) -> str | int:
    # If arg is an integer index, treat as webcam index
    try:
        return int(arg)
    except ValueError:
        return arg


def main():
    parser = argparse.ArgumentParser(description='Real-time RGB+Thermal Video Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model .pth')
    parser.add_argument('--rgb_source', type=str, required=True, help='RGB video source (file path or camera index)')
    parser.add_argument('--thermal_source', type=str, required=True, help='Thermal video source (file path or camera index)')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--input_size', type=int, default=640, help='Model input size')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.45, help='NMS threshold')
    parser.add_argument('--display', action='store_true', help='Display real-time window')
    parser.add_argument('--save_path', type=str, default='', help='Optional output video path')
    parser.add_argument('--max_frames', type=int, default=0, help='Stop after N frames (0 = unlimited)')

    args = parser.parse_args()

    # Initialize inference
    infer = RGBTInference(
        model_path=args.model_path,
        num_classes=args.num_classes,
        input_size=args.input_size,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        device=args.device
    )

    # Open sources
    rgb_src = parse_source(args.rgb_source)
    thr_src = parse_source(args.thermal_source)
    cap_rgb = open_video_source(rgb_src)
    cap_thr = open_video_source(thr_src)

    if not cap_rgb.isOpened():
        raise RuntimeError(f'Could not open RGB source: {args.rgb_source}')
    if not cap_thr.isOpened():
        raise RuntimeError(f'Could not open Thermal source: {args.thermal_source}')

    # Video writer setup
    writer = None
    if args.save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap_rgb.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH) or args.input_size)
        height = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT) or args.input_size)
        writer = cv2.VideoWriter(args.save_path, fourcc, fps, (width, height))

    frame_count = 0
    prev_time = time.time()

    while True:
        ok_rgb, frame_rgb_bgr = cap_rgb.read()
        ok_thr, frame_thr = cap_thr.read()
        if not ok_rgb or not ok_thr:
            break

        # Convert frames: RGB as RGB, Thermal as grayscale
        frame_rgb_rgb = cv2.cvtColor(frame_rgb_bgr, cv2.COLOR_BGR2RGB)
        if frame_thr.ndim == 3 and frame_thr.shape[2] == 3:
            frame_thr_gray = cv2.cvtColor(frame_thr, cv2.COLOR_BGR2GRAY)
        else:
            frame_thr_gray = frame_thr

        detections = infer.predict(frame_rgb_rgb, frame_thr_gray)

        # Draw on RGB frame for visualization
        vis_bgr = draw_detections_on_frame(frame_rgb_bgr.copy(), detections, infer.class_names)

        # FPS
        curr_time = time.time()
        fps = 1.0 / max(1e-6, (curr_time - prev_time))
        prev_time = curr_time
        cv2.putText(vis_bgr, f'FPS: {fps:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if args.display:
            cv2.imshow('RGB + Thermal Detections', vis_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if writer is not None:
            writer.write(vis_bgr)

        frame_count += 1
        if args.max_frames > 0 and frame_count >= args.max_frames:
            break

    cap_rgb.release()
    cap_thr.release()
    if writer is not None:
        writer.release()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
