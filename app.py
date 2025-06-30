import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import threading
from queue import Queue
import shutil
import base64

# Output directory and processed video path
OUTPUT_DIR = Path("output")
# Ensure output directory exists
try:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"✅ Output directory ready: {OUTPUT_DIR}")
except Exception as e:
    print(f"❌ Error creating output directory: {e}")

PROCESSED_VIDEO_PATH = OUTPUT_DIR / "processed_video.mp4"

# Import our r/vision processor
try:
    from r_vision import RVisionProcessor
except ImportError:
    st.error("r_vision module not found. Please ensure r_vision.py is in the same directory.")
    st.stop()

st.set_page_config(
    page_title="r/vision - Live Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #FF6B35, #F7931E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .demo-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    .video-info {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    .progress-info {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    .loader {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'temp_files' not in st.session_state:
    st.session_state.temp_files = []

# Clean up any leftover files from previous sessions
def cleanup_files():
    # Clean up any temporary files tracked in session state
    if 'temp_files' in st.session_state:
        for file_path in st.session_state.temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"✅ Cleaned up file from previous session: {file_path}")
            except Exception as e:
                print(f"❌ Error cleaning up file: {e}")
        st.session_state.temp_files = []
    
    # Always check if the output file exists and clean it up if needed
    if PROCESSED_VIDEO_PATH.exists():
        try:
            PROCESSED_VIDEO_PATH.unlink()
            print(f"✅ Cleaned up output file from previous session: {PROCESSED_VIDEO_PATH}")
        except Exception as e:
            print(f"❌ Error cleaning up output file: {e}")

# Run cleanup on app start
cleanup_files()

# Header
st.markdown('<h1 class="main-header">🎬 r/vision</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Add cyberpunk-style tracking effects to your videos instantly!</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("⚙️ Settings")
st.sidebar.markdown("---")

# Effect settings
st.sidebar.subheader("🎨 Effect Settings")
yolo_confidence = st.sidebar.slider(
    "Object Detection Sensitivity", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05,
    help="Lower = more objects detected (cyberpunk mode)"
)

pose_confidence = st.sidebar.slider(
    "People Tracking Sensitivity", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.05
)

# Feature toggles
st.sidebar.subheader("🔧 Features")
enable_yolo = st.sidebar.checkbox("Object Detection", value=True, help="Track cars, phones, animals, etc.")
enable_pose = st.sidebar.checkbox("People Tracking", value=True, help="Track human poses")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.subheader("📤 Upload Your Video")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Max file size: 200MB for demo"
    )
    
    if uploaded_file is not None:
        # Display file info
        file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # MB
        st.info(f"📁 **{uploaded_file.name}** ({file_size:.1f} MB)")
        
        if file_size > 200:
            st.error("⚠️ File too large for demo. Please use a file smaller than 200MB.")
        else:
            # Show original video
            st.video(uploaded_file)
            
            # Get video info for display
            if not st.session_state.processing:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_path = tmp_file.name
                    # Track the temporary file
                    st.session_state.temp_files.append(temp_path)
                
                try:
                    processor = RVisionProcessor()
                    video_info = processor.get_video_info(temp_path)
                    
                    if video_info:
                        st.markdown(f"""
                        <div class="video-info">
                            <h4>📊 Video Information</h4>
                            <p><strong>Resolution:</strong> {video_info['resolution']}</p>
                            <p><strong>Duration:</strong> {video_info['duration_formatted']}</p>
                            <p><strong>FPS:</strong> {video_info['fps']}</p>
                            <p><strong>Total Frames:</strong> {video_info['total_frames']:,}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.session_state.video_info = video_info
                finally:
                    # Clean up the temporary file
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            print(f"✅ Temporary file deleted after info extraction: {temp_path}")
                            # Remove from tracking list
                            if temp_path in st.session_state.temp_files:
                                st.session_state.temp_files.remove(temp_path)
                    except Exception as e:
                        print(f"❌ Error deleting temporary file: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sample videos section
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.subheader("🎬 Try Sample Videos")
    st.markdown("""
    Don't have a video? Try these examples:
    - **Tech Review**: Phone unboxing or laptop showcase
    - **Travel**: Street scenes with people and cars
    - **Sports**: Action footage with multiple people
    - **Music**: Performance or dance video
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="demo-section">', unsafe_allow_html=True)
    st.subheader("🎯 Processing")
    
    if uploaded_file is not None and file_size <= 200:
        
        # Processing button
        process_button = st.button(
            "🚀 Add Effects", 
            type="primary", 
            disabled=st.session_state.processing
        )
        
        if process_button and not st.session_state.processing:
            st.session_state.processing = True
            st.session_state.processed_video = None
            
            # Clean up old processed video
            if PROCESSED_VIDEO_PATH.exists():
                try:
                    PROCESSED_VIDEO_PATH.unlink()
                except:
                    pass
            
            try:
                # Create temporary files
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
                    tmp_input.write(uploaded_file.getvalue())
                    input_path = tmp_input.name
                    # Track the temporary file
                    st.session_state.temp_files.append(input_path)
                
                output_path = str(PROCESSED_VIDEO_PATH)
                
                # Initialize processor
                with st.spinner("🔄 Initializing r/vision..."):
                    processor = RVisionProcessor(
                        enable_yolo=enable_yolo,
                        enable_pose=enable_pose,
                        confidence_threshold=yolo_confidence,
                        pose_confidence=pose_confidence
                    )
                
                # Progress tracking containers
                progress_container = st.container()
                
                with progress_container:
                    progress_bar = st.progress(0)
                    status_container = st.empty()
                
                def update_progress(progress_info):
                    try:
                        progress_percent = progress_info['progress_percent'] / 100
                        progress_bar.progress(min(progress_percent, 1.0))
                        status_container.markdown(f"""
                        <div class="progress-info">
                            <h4>🔄 Processing Frame {progress_info['frame_count']:,} of {progress_info['total_frames']:,}</h4>
                            <p><strong>Progress:</strong> {progress_info['progress_percent']:.1f}%</p>
                            <p><strong>Processing Speed:</strong> {progress_info['processing_fps']:.1f} FPS</p>
                            <p><strong>Elapsed Time:</strong> {progress_info['elapsed_time']:.1f}s</p>
                            <p><strong>Estimated Remaining:</strong> {progress_info['estimated_remaining']:.1f}s</p>
                        </div>
                        """, unsafe_allow_html=True)
                    except Exception:
                        pass
                
                # Process the video with progress callback
                success = processor.process_video(
                    input_path, 
                    output_path, 
                    progress_callback=update_progress
                )
                
                if success and os.path.exists(output_path):
                    time.sleep(2)
                    file_size = os.path.getsize(output_path)
                    if file_size > 1000:
                        progress_bar.progress(1.0)
                        status_container.success("✅ Processing complete!")
                        st.session_state.processed_video = True  # Just a flag
                        st.success("🎉 Your enhanced video is ready!")
                    else:
                        st.error("❌ Processed video file is too small. Please try again.")
                else:
                    st.error("❌ Processing failed. Please try again with a different video.")
                
                # Cleanup temporary input
                try:
                    if os.path.exists(input_path):
                        os.unlink(input_path)
                        print(f"✅ Temporary input file deleted: {input_path}")
                except Exception as e:
                    print(f"❌ Error deleting temporary input file: {e}")
                    pass
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Try a smaller video file or different format.")
            finally:
                st.session_state.processing = False
        
        # Show download option for processed video
        if st.session_state.processed_video and not st.session_state.processing:
            if PROCESSED_VIDEO_PATH.exists():
                # Read the video file
                with open(PROCESSED_VIDEO_PATH, "rb") as f:
                    video_bytes = f.read()
                
                # Success message with styled container
                st.markdown("""
                <div style="background-color: #e8f5e9; border-left: 6px solid #4caf50; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
                    <h3 style="color: #2e7d32; margin-top: 0;">✅ Video Processing Complete!</h3>
                    <p>Your video has been successfully processed with r/vision effects.</p>
                    <p>Click the download button below to save your enhanced video.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Define these functions at the outer scope to avoid definition issues
                if 'file_downloaded' not in st.session_state:
                    st.session_state.file_downloaded = False
                
                # Enhanced download button
                if not st.session_state.file_downloaded:
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.download_button(
                            label="⬇️ Download Enhanced Video",
                            data=video_bytes,
                            file_name=f"enhanced_{uploaded_file.name}",
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True,
                            on_click=lambda: mark_downloaded()
                        )
                else:
                    # Show message that file was downloaded
                    st.success("✅ Video downloaded successfully! You can now process another video.")
                
                # Process new video button
                st.button("🔄 Process Another Video", type="secondary", on_click=lambda: reset_state())
            else:
                st.error("Processed video not found. Please try again.")
    else:
        st.info("👆 Upload a video file to get started!")

# Functions defined outside of the render flow to avoid redefinition issues
def mark_downloaded():
    """Mark the file as downloaded and schedule it for deletion"""
    st.session_state.file_downloaded = True
    
    # We'll still keep the file until the user resets or processes a new video
    # This prevents the "Processed video not found" error message
    # The file will be cleaned up when the user resets or processes a new video
    print(f"✅ File marked as downloaded, will be deleted on reset: {PROCESSED_VIDEO_PATH}")

def reset_state():
    """Reset the app state completely"""
    st.session_state.processed_video = None
    st.session_state.file_downloaded = False
    # Clean up any files
    try:
        if PROCESSED_VIDEO_PATH.exists():
            PROCESSED_VIDEO_PATH.unlink()
            print(f"✅ File deleted during reset: {PROCESSED_VIDEO_PATH}")
    except Exception as e:
        print(f"❌ Error deleting file during reset: {e}")
    
    # Clear other session states that might need resetting
    cleanup_files()

# Continue with the rest of the UI
st.markdown('</div>', unsafe_allow_html=True)

# Features showcase
st.markdown("---")
st.subheader("✨ What r/vision Does")

feature_cols = st.columns(4)

with feature_cols[0]:
    st.markdown("""
    <div class="feature-box">
        <h4>🎨 Color Coding</h4>
        <p>Different colors for people, cars, electronics, animals</p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols[1]:
    st.markdown("""
    <div class="feature-box">
        <h4>📊 Live Stats</h4>
        <p>FPS and confidence on every tracked object</p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols[2]:
    st.markdown("""
    <div class="feature-box">
        <h4>⚡ Real-time</h4>
        <p>Fast processing optimized for smooth tracking</p>
    </div>
    """, unsafe_allow_html=True)

with feature_cols[3]:
    st.markdown("""
    <div class="feature-box">
        <h4>🎬 Pro Effects</h4>
        <p>Cyberpunk HUD-style overlays</p>
    </div>
    """, unsafe_allow_html=True)

# Footer with GitHub and Instagram links
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("""
    <div style="text-align: center;">
        <h4>🚀 Get the Code & See it in Action!</h4>
        <p>r/vision is open source! Perfect for content creators and developers.</p>
        <div style="margin: 20px 0;">
            <a href="https://github.com/ferrary7/r-vision" target="_blank" style="text-decoration: none; margin: 0 10px;">
                <button style="background: linear-gradient(45deg, #24292e, #586069); color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 14px;">
                    📂 GitHub Repository
                </button>
            </a>
            <a href="https://www.instagram.com/reel/DLhWQweyOcy/" target="_blank" style="text-decoration: none; margin: 0 10px;">
                <button style="background: linear-gradient(45deg, #E4405F, #F77737); color: white; border: none; padding: 12px 24px; border-radius: 6px; cursor: pointer; font-size: 14px;">
                    📱 Instagram Demo
                </button>
            </a>
        </div>
        <p><small>Made with ❤️ for content creators</small></p>
    </div>
    """, unsafe_allow_html=True)

# Tips section
with st.sidebar:
    st.markdown("---")
    st.subheader("💡 Tips")
    st.markdown("""
    **For best results:**
    - Use videos under 30 seconds for demo
    - Clear lighting works better
    - Multiple objects = cooler effects
    - Lower sensitivity = more tracking
    
    **Perfect for:**
    - Social media content
    - Music videos
    - Tech reviews
    - Travel vlogs
    """)
    
    st.markdown("---")
    st.markdown("**🎯 Color Guide:**")
    st.markdown("🟢 People | 🔴 Vehicles | 🔵 Electronics")
    st.markdown("🟣 Animals | 🟠 Food | 🟡 Objects")
    
    # Social links in sidebar too
    st.markdown("---")
    st.markdown("**🔗 Links:**")
    st.markdown("[📂 GitHub Repo](https://github.com/ferrary7/r-vision)")
    st.markdown("[📱 Instagram Demo](https://www.instagram.com/reel/DLhWQweyOcy/)")

# Note: We can't register a shutdown callback directly in Streamlit
# So we'll rely on the cleanup that happens when processing new videos
# and when the user clicks the reset button
