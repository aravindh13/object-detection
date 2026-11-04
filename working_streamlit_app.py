#!/usr/bin/env python3
"""
Complete working video processing system for RGB + Thermal object detection.
"""

import cv2
import numpy as np
import time
import os
import streamlit as st
from separate_inference import RGBOnlyInference, ThermalOnlyInference, visualize_detections

def main():
    st.set_page_config(page_title="RGB+Thermal Detection", layout="wide")
    st.title("RGB + Thermal Object Detection - Working Version")

    with st.sidebar:
        st.header("Configuration")
        
        # Model settings
        model_path = st.text_input("Model path", value="runs/exp/best_model.pth")
        device = st.selectbox("Device", ["cpu", "auto"], index=0)
        conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.95, 0.01)
        nms_threshold = st.slider("NMS threshold", 0.0, 1.0, 0.1, 0.01)
        max_detections = st.number_input("Max detections per stream", min_value=1, max_value=20, value=3, step=1)
        
        st.markdown("---")
        st.subheader("Video Sources")
        
        # Video sources
        rgb_source = st.text_input("RGB video path", value="sample_videos/rgb_sample.mp4")
        thermal_source = st.text_input("Thermal video path", value="sample_videos/thermal_sample.mp4")
        max_frames = st.number_input("Max frames to process (0 = all)", min_value=0, max_value=1000, value=50, step=1)
        
        start_button = st.button("Start Processing", type="primary")

    if start_button:
        # Validate inputs
        if not os.path.exists(model_path):
            st.error(f"Model not found: {model_path}")
            st.stop()
        
        if not os.path.exists(rgb_source):
            st.error(f"RGB video not found: {rgb_source}")
            st.stop()
            
        if not os.path.exists(thermal_source):
            st.error(f"Thermal video not found: {thermal_source}")
            st.stop()

        # Load models
        with st.spinner("Loading models..."):
            try:
                rgb_infer = RGBOnlyInference(
                    model_path=model_path,
                    num_classes=80,
                    input_size=640,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold,
                    device=device
                )
                
                thermal_infer = ThermalOnlyInference(
                    model_path=model_path,
                    num_classes=80,
                    input_size=640,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold,
                    device=device
                )
                st.success("Models loaded successfully!")
            except Exception as e:
                st.error(f"Error loading models: {e}")
                st.stop()

        # Open videos
        with st.spinner("Opening videos..."):
            rgb_cap = cv2.VideoCapture(rgb_source)
            thermal_cap = cv2.VideoCapture(thermal_source)
            
            if not rgb_cap.isOpened():
                st.error(f"Could not open RGB video: {rgb_source}")
                st.stop()
                
            if not thermal_cap.isOpened():
                st.error(f"Could not open Thermal video: {thermal_source}")
                st.stop()
            
            st.success("Videos opened successfully!")

        # Create display areas
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 RGB Stream")
            rgb_placeholder = st.empty()
            rgb_info = st.empty()
        
        with col2:
            st.subheader("🌡️ Thermal Stream")
            thermal_placeholder = st.empty()
            thermal_info = st.empty()
        
        # Processing info
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process videos
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret_rgb, rgb_frame = rgb_cap.read()
                ret_thermal, thermal_frame = thermal_cap.read()
                
                if not ret_rgb or not ret_thermal:
                    break
                
                # Process every 3rd frame for better performance
                if frame_count % 3 != 0:
                    frame_count += 1
                    continue
                
                # Convert thermal to grayscale if needed
                if thermal_frame.ndim == 3:
                    thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
                else:
                    thermal_gray = thermal_frame
                
                # Run inference
                try:
                    rgb_dets = rgb_infer.predict(rgb_frame)
                    thermal_dets = thermal_infer.predict(thermal_gray)
                    
                    # Visualize detections
                    rgb_vis = visualize_detections(rgb_frame.copy(), rgb_dets, rgb_infer.class_names, "RGB", max_detections)
                    thermal_vis = visualize_detections(thermal_frame.copy(), thermal_dets, thermal_infer.class_names, "Thermal", max_detections)
                    
                    # Add frame info
                    cv2.putText(rgb_vis, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(rgb_vis, f"Detections: {len(rgb_dets)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.putText(thermal_vis, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(thermal_vis, f"Detections: {len(thermal_dets)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display frames
                    rgb_placeholder.image(cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB), channels="RGB")
                    thermal_placeholder.image(cv2.cvtColor(thermal_vis, cv2.COLOR_BGR2RGB), channels="RGB")
                    
                    # Update info
                    rgb_info.write(f"RGB: {len(rgb_dets)} total, {min(len(rgb_dets), max_detections)} shown")
                    thermal_info.write(f"Thermal: {len(thermal_dets)} total, {min(len(thermal_dets), max_detections)} shown")
                    
                    # Calculate and display FPS
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    status_text.write(f"Processing FPS: {fps:.1f} | Frame: {frame_count} | Elapsed: {elapsed_time:.1f}s")
                    
                    # Update progress
                    if max_frames > 0:
                        progress = min(frame_count / max_frames, 1.0)
                        progress_bar.progress(progress)
                    
                except Exception as e:
                    st.error(f"Error processing frame {frame_count}: {e}")
                
                frame_count += 1
                
                # Limit processing
                if max_frames > 0 and frame_count >= max_frames:
                    break
                
                # Small delay to prevent overwhelming the UI
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            st.warning("Processing stopped by user")
        
        finally:
            # Cleanup
            rgb_cap.release()
            thermal_cap.release()
            
            # Final stats
            total_time = time.time() - start_time
            avg_fps = frame_count / total_time if total_time > 0 else 0
            
            st.success(f"Processing completed! Processed {frame_count} frames in {total_time:.1f}s (avg {avg_fps:.1f} FPS)")

if __name__ == "__main__":
    main()

