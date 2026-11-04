import os
import time
import cv2
import numpy as np
import streamlit as st
import atexit
import glob

from separate_inference import RGBOnlyInference, ThermalOnlyInference, visualize_detections

# Cleanup function for temporary files
def cleanup_temp_files():
    """Clean up any leftover temporary files on app exit."""
    temp_files = glob.glob("temp_*.mp4") + glob.glob("temp_*.avi") + glob.glob("temp_*.mov")
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except PermissionError:
            pass  # Ignore permission errors during cleanup

# Register cleanup function
atexit.register(cleanup_temp_files)

st.set_page_config(page_title="RGB+Thermal Detection", layout="wide")

st.title("RGB + Thermal Object Detection - Streamlit UI")

with st.sidebar:
    st.header("Configuration")
    default_model = os.path.join("runs", "exp", "best_model.pth")
    model_path = st.text_input("Model path", value=default_model)
    device = st.selectbox("Device", ["auto", "cpu"], index=0)
    input_size = st.number_input("Input size", min_value=160, max_value=1280, value=640, step=32)
    conf_thr = st.slider("Confidence threshold", 0.0, 1.0, 0.9, 0.01)
    nms_thr = st.slider("NMS threshold", 0.0, 1.0, 0.2, 0.01)
    max_detections = st.number_input("Max detections per stream", min_value=1, max_value=50, value=5, step=1)
    frame_skip = st.number_input("Process every N frames", min_value=1, max_value=10, value=2, step=1)

    st.markdown("---")
    st.subheader("Sources")
    
    # Source type selection
    source_type = st.radio("Input Type", ["Images", "Videos", "Camera"], index=1)
    
    if source_type == "Images":
        rgb_source = st.text_input("RGB image path", value=os.path.join("data", "rgb", "sample_000.jpg"))
        thermal_source = st.text_input("Thermal image path", value=os.path.join("data", "thermal", "sample_000.jpg"))
        max_frames = 1  # Single image
    elif source_type == "Videos":
        st.write("**RGB Video:**")
        rgb_video_option = st.radio("RGB Video Source", ["Upload File", "File Path"], index=1)
        if rgb_video_option == "Upload File":
            rgb_uploaded = st.file_uploader("Upload RGB Video", type=['mp4', 'avi', 'mov', 'mkv'])
            rgb_source = rgb_uploaded.name if rgb_uploaded else ""
        else:
            rgb_source = st.text_input("RGB video path", value="sample_videos/rgb_sample.mp4")
        
        st.write("**Thermal Video:**")
        thermal_video_option = st.radio("Thermal Video Source", ["Upload File", "File Path"], index=1)
        if thermal_video_option == "Upload File":
            thermal_uploaded = st.file_uploader("Upload Thermal Video", type=['mp4', 'avi', 'mov', 'mkv'])
            thermal_source = thermal_uploaded.name if thermal_uploaded else ""
        else:
            thermal_source = st.text_input("Thermal video path", value="sample_videos/thermal_sample.mp4")
        
        max_frames = st.number_input("Max frames (0 = unlimited)", min_value=0, max_value=10000, value=0, step=1)
    else:  # Camera
        rgb_source = st.text_input("RGB camera index", value="0")
        thermal_source = st.text_input("Thermal camera index", value="1")
        max_frames = st.number_input("Max frames (0 = unlimited)", min_value=0, max_value=10000, value=0, step=1)

    run_btn = st.button("Start")

status = st.empty()
col1, col2 = st.columns(2)
view = st.empty()

@st.cache_resource(show_spinner=False)
def load_infer(model_path: str, device: str, input_size: int, conf: float, nms: float):
    """Load separate RGB and thermal inference models."""
    rgb_infer = RGBOnlyInference(
        model_path=model_path,
        num_classes=80,
        input_size=input_size,
        conf_threshold=conf,
        nms_threshold=nms,
        device=device,
    )
    
    thermal_infer = ThermalOnlyInference(
        model_path=model_path,
        num_classes=80,
        input_size=input_size,
        conf_threshold=conf,
        nms_threshold=nms,
        device=device,
    )
    
    return rgb_infer, thermal_infer


def parse_source(src: str):
    # If it's an integer, treat as webcam index
    try:
        return int(src)
    except Exception:
        return src


def ensure_open(src):
    cap = cv2.VideoCapture(src)
    return cap if cap.isOpened() else None

def safe_remove_file(file_path, max_retries=3):
    """Safely remove a file with retries for Windows file locking issues."""
    if not file_path or not os.path.exists(file_path):
        return
    
    for attempt in range(max_retries):
        try:
            os.remove(file_path)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.1 * (attempt + 1))  # Increasing delay
            else:
                print(f"Warning: Could not delete temporary file {file_path} after {max_retries} attempts")


def is_image_file(path: str) -> bool:
    return str(path).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))


def is_video_file(path: str) -> bool:
    return str(path).lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv"))


def directory_first_media(path: str) -> str | None:
    try:
        entries = sorted(os.listdir(path))
    except Exception:
        return None
    for f in entries:
        full = os.path.join(path, f)
        if os.path.isfile(full) and (is_video_file(full) or is_image_file(full)):
            return full
    return None


if run_btn:
    if not os.path.isfile(model_path):
        status.error(f"Model not found: {model_path}")
        st.stop()

    rgb_infer, thermal_infer = load_infer(model_path, device, input_size, conf_thr, nms_thr)

    # Handle uploaded files and validate paths
    rgb_temp_path = None
    thermal_temp_path = None
    
    # Initialize session state for cleanup tracking
    if 'temp_files' not in st.session_state:
        st.session_state.temp_files = []
    
    if source_type == "Videos":
        # Save uploaded files to temporary locations
        if rgb_video_option == "Upload File" and rgb_uploaded:
            rgb_temp_path = f"temp_rgb_{rgb_uploaded.name}"
            with open(rgb_temp_path, "wb") as f:
                f.write(rgb_uploaded.getbuffer())
            rgb_source = rgb_temp_path
            st.session_state.temp_files.append(rgb_temp_path)
        
        if thermal_video_option == "Upload File" and thermal_uploaded:
            thermal_temp_path = f"temp_thermal_{thermal_uploaded.name}"
            with open(thermal_temp_path, "wb") as f:
                f.write(thermal_uploaded.getbuffer())
            thermal_source = thermal_temp_path
            st.session_state.temp_files.append(thermal_temp_path)

    # Validate that sources are not empty
    if not rgb_source or not thermal_source:
        status.error("Please provide valid RGB and Thermal sources")
        st.stop()

    # If a directory is provided, auto-pick the first media file inside
    if os.path.isdir(rgb_source):
        picked = directory_first_media(rgb_source)
        if picked:
            rgb_source = picked
        else:
            status.error(f"No media files found in RGB directory: {rgb_source}")
            st.stop()
            
    if os.path.isdir(thermal_source):
        picked = directory_first_media(thermal_source)
        if picked:
            thermal_source = picked
        else:
            status.error(f"No media files found in Thermal directory: {thermal_source}")
            st.stop()

    rgb_src = parse_source(rgb_source)
    thr_src = parse_source(thermal_source)

    cap_rgb = ensure_open(rgb_src)
    cap_thr = ensure_open(thr_src)

    if cap_rgb is None:
        status.error(f"Could not open RGB source: {rgb_source}. Provide a file path, directory with media, or camera index.")
        st.stop()
    if cap_thr is None:
        status.error(f"Could not open Thermal source: {thermal_source}. Provide a file path, directory with media, or camera index.")
        st.stop()

    status.info("Streaming... press Stop in the toolbar or refresh to end.")
    
    # Create side-by-side columns for RGB and Thermal
    rgb_col, thermal_col = st.columns(2)
    
    with rgb_col:
        st.subheader("🔴 RGB Stream")
        rgb_frame_slot = st.empty()
        rgb_info_slot = st.empty()
    
    with thermal_col:
        st.subheader("🌡️ Thermal Stream")
        thermal_frame_slot = st.empty()
        thermal_info_slot = st.empty()
    
    # Overall info slot
    overall_info_slot = st.empty()

    frames_done = 0
    prev = time.time()
    
    # Get video properties for better synchronization
    rgb_fps = cap_rgb.get(cv2.CAP_PROP_FPS)
    thermal_fps = cap_thr.get(cv2.CAP_PROP_FPS)
    rgb_frame_count = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
    thermal_frame_count = int(cap_thr.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Display video info
    overall_info_slot.write(f"RGB: {rgb_fps:.1f} FPS, {rgb_frame_count} frames | Thermal: {thermal_fps:.1f} FPS, {thermal_frame_count} frames")

    while True:
        ok1, f_rgb = cap_rgb.read()
        ok2, f_thr = cap_thr.read()
        if not ok1 or not ok2:
            break

        # Skip frames for better performance
        if frames_done % frame_skip != 0:
            frames_done += 1
            continue

        # Prepare inputs
        rgb_for_model = cv2.cvtColor(f_rgb, cv2.COLOR_BGR2RGB)
        if f_thr.ndim == 3 and f_thr.shape[2] == 3:
            thr_gray = cv2.cvtColor(f_thr, cv2.COLOR_BGR2GRAY)
        else:
            thr_gray = f_thr

        # Run separate inference
        rgb_dets = rgb_infer.predict(rgb_for_model)
        thermal_dets = thermal_infer.predict(thr_gray)
        
        # Visualize detections separately
        rgb_vis = visualize_detections(f_rgb.copy(), rgb_dets, rgb_infer.class_names, "RGB", max_detections)
        thermal_vis = visualize_detections(f_thr.copy(), thermal_dets, thermal_infer.class_names, "Thermal", max_detections)

        # FPS calculation
        now = time.time()
        fps = 1.0 / max(1e-6, (now - prev))
        prev = now
        
        # Add info overlays
        cv2.putText(rgb_vis, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(rgb_vis, f"Frame: {frames_done}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(rgb_vis, f"RGB Detections: {len(rgb_dets)}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(thermal_vis, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(thermal_vis, f"Frame: {frames_done}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(thermal_vis, f"Thermal Detections: {len(thermal_dets)}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Display side by side
        rgb_frame_slot.image(cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB), channels="RGB")
        thermal_frame_slot.image(cv2.cvtColor(thermal_vis, cv2.COLOR_BGR2RGB), channels="RGB")
        
        # Update info
        rgb_info_slot.write(f"RGB: {len(rgb_dets)} total, {min(len(rgb_dets), max_detections)} shown")
        thermal_info_slot.write(f"Thermal: {len(thermal_dets)} total, {min(len(thermal_dets), max_detections)} shown")
        overall_info_slot.write(f"Processing FPS: {fps:.1f} | Frame: {frames_done} | Skip: {frame_skip}")

        frames_done += 1
        if max_frames > 0 and frames_done >= max_frames:
            break

    cap_rgb.release()
    cap_thr.release()
    
    # Clean up temporary files safely
    safe_remove_file(rgb_temp_path)
    safe_remove_file(thermal_temp_path)
    
    # Clean up session state temp files
    for temp_file in st.session_state.temp_files:
        safe_remove_file(temp_file)
    st.session_state.temp_files = []
    
    status.success("Streaming finished.")
