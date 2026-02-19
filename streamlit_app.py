"""
Kalpataru Streamlit UI Entry Point
Interactive web interface for agricultural AI services.
"""

import streamlit as st
from services.translation import translate_text

# Page configuration
st.set_page_config(
    page_title="Kalpataru - Agricultural AI Platform",
    page_icon="🌾",
    layout="wide"
)

def main():
    """Main Streamlit application."""
    st.title("🌾 Kalpataru")
    st.markdown("## Agricultural AI Platform")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["Home", "Disease Detection", "Irrigation", "Weather", "Yield", "Price", "Crop Recommendation"]
    )
    
    if page == "Home":
        st.markdown("""
        Welcome to Kalpataru - Your intelligent agricultural assistant.
        
        ### Features:
        - 🦠 Disease Detection (CNN-based)
        - 💧 Irrigation Optimization
        - 🌤️ Weather Forecasting (LSTM)
        - 📈 Yield Prediction (XGBoost)
        - 💰 Price Forecasting (Prophet)
        - 🌱 Crop Recommendation
        """)
    
    elif page == "Disease Detection":
        st.markdown("### Plant Disease Detection")
        uploaded_file = st.file_uploader("Upload plant image", type=['jpg', 'png', 'jpeg'])
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Image")
            # Placeholder for disease detection
    
    elif page == "Irrigation":
        st.markdown("### Irrigation Optimization")
        st.write("Coming soon...")
    
    elif page == "Weather":
        st.markdown("### Weather Forecasting")
        st.write("Coming soon...")
    
    elif page == "Yield":
        st.markdown("### Yield Prediction")
        st.write("Coming soon...")
    
    elif page == "Price":
        st.markdown("### Price Forecasting")
        st.write("Coming soon...")
    
    elif page == "Crop Recommendation":
        st.markdown("### Crop Recommendation")
        st.write("Coming soon...")

if __name__ == '__main__':
    main()
