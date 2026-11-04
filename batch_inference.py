import os
import argparse
import cv2
from inference import RGBTInference


def main():
    parser = argparse.ArgumentParser(description='Batch inference for RGB+Thermal image pairs')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--rgb_dir', type=str, required=True, help='Directory with RGB images')
    parser.add_argument('--thermal_dir', type=str, required=True, help='Directory with thermal images')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save results')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--num_classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--input_size', type=int, default=640, help='Input size')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.45, help='NMS threshold')
    parser.add_argument('--pair_by_index', action='store_true', help='Pair RGB and thermal by sorted index if names do not match')
    parser.add_argument('--rgb_single', type=str, default='', help='Use a single RGB image for all thermal images')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    infer = RGBTInference(
        model_path=args.model_path,
        num_classes=args.num_classes,
        input_size=args.input_size,
        conf_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        device=args.device
    )

    # Build mapping of available files by basename (without extension)
    def list_images(d):
        return {os.path.splitext(f)[0]: os.path.join(d, f)
                for f in os.listdir(d)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))}

    rgb_map = list_images(args.rgb_dir)
    thr_map = list_images(args.thermal_dir)

    common_keys = sorted(set(rgb_map.keys()) & set(thr_map.keys()))
    pairs = []
    if common_keys:
        print(f'Found {len(common_keys)} matched pairs')
        for key in common_keys:
            pairs.append((rgb_map[key], thr_map[key], key))
    elif args.rgb_single:
        if not os.path.isfile(args.rgb_single):
            print(f'RGB single image not found: {args.rgb_single}')
            return
        print(f'No name matches found. Using single RGB image for all thermal images: {args.rgb_single}')
        for key in sorted(thr_map.keys()):
            pairs.append((args.rgb_single, thr_map[key], key))
    elif args.pair_by_index:
        print('No name matches found. Pairing by index (sorted filenames).')
        rgb_items = sorted(rgb_map.items())
        thr_items = sorted(thr_map.items())
        n = min(len(rgb_items), len(thr_items))
        if n == 0:
            print('No images to pair.')
            return
        for i in range(n):
            rk, rpath = rgb_items[i]
            tk, tpath = thr_items[i]
            key = tk if tk else rk
            pairs.append((rpath, tpath, key))
    else:
        print('No matching RGB and thermal filenames found. Use --pair_by_index or --rgb_single to proceed.')
        return

    for rgb_path, thr_path, key in pairs:

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB) if rgb is not None else None
        thr = cv2.imread(thr_path, cv2.IMREAD_GRAYSCALE)

        if rgb is None or thr is None:
            print(f'Skipping {key}: failed to read one of the images')
            continue

        detections = infer.predict(rgb, thr)
        out_path = os.path.join(args.output_dir, f'{key}_detections.png')
        infer.visualize_detections(rgb, thr, detections, out_path)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()


