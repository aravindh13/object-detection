#!/usr/bin/env python3
"""
FIXED VIDEO APP - Solves video speed and object detection issues.
"""

import streamlit as st
import cv2
import numpy as np
import time
import os
from separate_inference import RGBOnlyInference, ThermalOnlyInference, visualize_detections

def main():
    st.set_page_config(page_title="Fixed Video Detection", layout="wide")
    st.title("🎥 RGB + Thermal Video Object Detection - FIXED")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model settings
        model_path = st.text_input("Model Path", value="runs/exp/best_model.pth")
        device = st.selectbox("Device", ["cpu", "auto"], index=0)
        
        # FIXED: Much higher confidence to get real detections
        conf_threshold = st.slider("Confidence", 0.0, 1.0, 0.999, 0.001)
        nms_threshold = st.slider("NMS Threshold", 0.0, 1.0, 0.01, 0.01)
        max_detections = st.slider("Max Detections", 1, 20, 5, 1)
        
        st.markdown("---")
        st.subheader("📁 Video Sources")
        
        # Video paths
        rgb_path = st.text_input("RGB Video", value="sample_videos/rgb_sample.mp4")
        thermal_path = st.text_input("Thermal Video", value="sample_videos/thermal_sample.mp4")
        
        # FIXED: Better processing settings
        max_frames = st.number_input("Max Frames", 0, 100, 50, 1)
        frame_skip = st.number_input("Skip Frames", 1, 10, 5, 1)
        
        # FIXED: Add delay control for video speed
        delay_ms = st.slider("Video Delay (ms)", 0, 1000, 200, 10)
        
        # Start button
        start_processing = st.button("🚀 Start Processing", type="primary")
    
    # Main content area
    if start_processing:
        # Validate files
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found: {model_path}")
            st.stop()
        
        if not os.path.exists(rgb_path):
            st.error(f"❌ RGB video not found: {rgb_path}")
            st.stop()
            
        if not os.path.exists(thermal_path):
            st.error(f"❌ Thermal video not found: {thermal_path}")
            st.stop()
        
        # Load models
        with st.spinner("🔄 Loading models..."):
            try:
                rgb_infer = RGBOnlyInference(
                    model_path=model_path,
                    device=device,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
                thermal_infer = ThermalOnlyInference(
                    model_path=model_path,
                    device=device,
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
                st.success(f"✅ Models loaded with confidence {conf_threshold}!")
            except Exception as e:
                st.error(f"❌ Error loading models: {e}")
                st.stop()
        
        # Open videos
        with st.spinner("🔄 Opening videos..."):
            rgb_cap = cv2.VideoCapture(rgb_path)
            thermal_cap = cv2.VideoCapture(thermal_path)
            
            if not rgb_cap.isOpened():
                st.error(f"❌ Could not open RGB video: {rgb_path}")
                st.stop()
                
            if not thermal_cap.isOpened():
                st.error(f"❌ Could not open Thermal video: {thermal_path}")
                st.stop()
            
            st.success("✅ Videos opened successfully!")
            
            # Get video info
            rgb_frames = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            thermal_frames = int(thermal_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            st.info(f"📊 RGB: {rgb_frames} frames | Thermal: {thermal_frames} frames")
        
        # Display areas
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
        
        # Progress and status
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Processing loop
        frame_count = 0
        processed_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret_rgb, rgb_frame = rgb_cap.read()
                ret_thermal, thermal_frame = thermal_cap.read()
                
                if not ret_rgb or not ret_thermal:
                    st.warning("⚠️ End of video reached")
                    break
                
                # Skip frames for performance
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Process frame
                try:
                    # Convert thermal to grayscale if needed
                    if thermal_frame.ndim == 3:
                        thermal_gray = cv2.cvtColor(thermal_frame, cv2.COLOR_BGR2GRAY)
                    else:
                        thermal_gray = thermal_frame
                    
                    # Run detection
                    rgb_dets = rgb_infer.predict(rgb_frame)
                    thermal_dets = thermal_infer.predict(thermal_gray)
                    
                    # FIXED: Only visualize if we have real detections
                    if len(rgb_dets) > 0 or len(thermal_dets) > 0:
                        rgb_vis = visualize_detections(rgb_frame.copy(), rgb_dets, rgb_infer.class_names, "RGB", max_detections)
                        thermal_vis = visualize_detections(thermal_frame.copy(), thermal_dets, thermal_infer.class_names, "Thermal", max_detections)
                    else:
                        # Show original frames if no detections
                        rgb_vis = rgb_frame.copy()
                        thermal_vis = thermal_frame.copy()
                    
                    # Add frame info
                    cv2.putText(rgb_vis, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(rgb_vis, f"Detections: {len(rgb_dets)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(rgb_vis, f"Conf: {conf_threshold:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.putText(thermal_vis, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(thermal_vis, f"Detections: {len(thermal_dets)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(thermal_vis, f"Conf: {conf_threshold:.3f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display
                    rgb_placeholder.image(cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB), channels="RGB")
                    thermal_placeholder.image(cv2.cvtColor(thermal_vis, cv2.COLOR_BGR2RGB), channels="RGB")
                    
                    # Update info
                    rgb_info.write(f"🔴 RGB: {len(rgb_dets)} total, {min(len(rgb_dets), max_detections)} shown")
                    thermal_info.write(f"🌡️ Thermal: {len(thermal_dets)} total, {min(len(thermal_dets), max_detections)} shown")
                    
                    # Calculate FPS
                    elapsed = time.time() - start_time
                    fps = processed_count / elapsed if elapsed > 0 else 0
                    status_text.write(f"📊 Processing: {fps:.1f} FPS | Frame: {frame_count} | Time: {elapsed:.1f}s | Conf: {conf_threshold:.3f}")
                    
                    # Update progress
                    if max_frames > 0:
                        progress = min(processed_count / max_frames, 1.0)
                        progress_bar.progress(progress)
                    
                    processed_count += 1
                    
                except Exception as e:
                    st.error(f"❌ Error processing frame {frame_count}: {e}")
                
                frame_count += 1
                
                # Limit processing
                if max_frames > 0 and processed_count >= max_frames:
                    break
                
                # FIXED: Add proper delay to control video speed
                time.sleep(delay_ms / 1000.0)
        
        except KeyboardInterrupt:
            st.warning("⚠️ Processing stopped by user")
        
        finally:
            # Cleanup
            rgb_cap.release()
            thermal_cap.release()
            
            # Final stats
            total_time = time.time() - start_time
            avg_fps = processed_count / total_time if total_time > 0 else 0
            
            st.success(f"🎉 Processing completed! Processed {processed_count} frames in {total_time:.1f}s (avg {avg_fps:.1f} FPS)")
    
    else:
        # Show instructions when not processing
        st.markdown("""
        ## 🔧 FIXED ISSUES
        
        ### 🎯 Problems Solved:
        1. **Video Too Fast**: Added delay control (200ms default)
        2. **No Object Detection**: Higher confidence threshold (0.999)
        3. **Only Boxes**: Fixed detection logic
        4. **Poor Performance**: Better frame skipping (5 frames)
        
        ### 📋 Instructions:
        1. **Adjust Video Speed**: Use "Video Delay" slider (200ms = normal speed)
        2. **High Confidence**: Use 0.999 for real detections
        3. **Check Paths**: Ensure video files exist
        4. **Click Start**: Press "🚀 Start Processing"
        
        ### 🎥 Video Files:
        - **RGB**: `sample_videos/rgb_sample.mp4`
        - **Thermal**: `sample_videos/thermal_sample.mp4`
        
        ### ⚡ Optimized Settings:
        - **Confidence**: 0.999 (very strict)
        - **NMS**: 0.01 (strong suppression)
        - **Delay**: 200ms (normal speed)
        - **Skip Frames**: 5 (performance)
        """)

if __name__ == "__main__":
    main()

