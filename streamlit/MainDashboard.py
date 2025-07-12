import os
from dotenv import load_dotenv
load_dotenv()
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Parklytics with PlatePulse - Main Dashboard",
    page_icon=":parking:",
    layout="wide"
)

# Minimal sidebar: just logo/title and a short description
with st.sidebar:
    st.markdown("""
    # 🚗 Parklytics with PlatePulse
    <small>Edge AI Vehicle & Parking Analytics</small>
    ---
    """, unsafe_allow_html=True)

st.title("🚗 Parklytics with PlatePulse")
st.markdown("""
### Transforming Vehicle and Parking Management with Edge AI

Welcome to **Parklytics with PlatePulse** – a comprehensive edge AI-powered system for real-time vehicle movement analysis, license plate recognition, and parking management. Explore our intelligent automation modules and analytics for modern traffic and parking challenges.
""")

st.markdown("---")

st.header("🔎 Explore Application Modules")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Detection Modules")
    if hasattr(st, "page_link"):
        st.page_link("pages/license_plate_video_inference.py", label="Video Number Plate Detection", icon="🎥")
        st.page_link("pages/license_plate_image.py", label="Image Number Plate Detection", icon="🖼️")
        st.page_link("pages/parking_lot.py", label="Parking Slot Detection", icon="🅿️")
    else:
        st.markdown("- Use the sidebar to navigate to detection modules.")
    st.image("streamlit/demo_/License Plate Detection - Video Analysis demo.png", caption="Video Number Plate Detection")
    st.image("streamlit/demo_/License Plate Image Detection.png", caption="Image Number Plate Detection")
    st.image("streamlit/demo_/Parking Lot Detection.png", caption="Parking Slot Detection")

with col2:
    st.subheader("Analytics Modules")
    if hasattr(st, "page_link"):
        st.page_link("pages/vehicle_movement_patterns.py", label="Vehicle Movement Patterns", icon="📊")
        st.page_link("pages/Parking Data Insights.py", label="Parking Data Insights", icon="📈")
    else:
        st.markdown("- Use the sidebar to navigate to analytics modules.")
    st.image("streamlit/demo_/Vehicle Movement Analysis.png", caption="Vehicle Movement Patterns")
    st.image("streamlit/demo_/Parking-Data-Insights.png", caption="Parking Data Insights")

st.markdown("---")

st.info("Select a module from the sidebar or use the buttons above to get started!")

st.markdown("""
#### About the Project
**Parklytics with PlatePulse** is designed for educational campuses, corporate offices, and urban environments, providing:
- Real-time license plate detection & OCR (YOLOv8 + LPRNet)
- Automated parking space monitoring
- Interactive analytics dashboards
- Edge AI optimization for fast inference
- Web-based interface with Streamlit
""")




