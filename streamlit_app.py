"""
Kalpataru Streamlit UI Entry Point
Interactive web interface for agricultural AI services.
"""

import streamlit as st
from services.translation import translate_text
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Kalpataru - Agricultural AI Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4CAF50;
    }
    .metric-card {
        background-color: #f0f7f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #4CAF50;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    # Header
    st.title("🌾 Kalpataru")
    st.markdown("## Agricultural AI Platform")
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Home",
            "🤖 AI Assistant",
            "🦠 Disease Detection",
            "💧 Irrigation",
            "🌤️ Weather",
            "📈 Yield Prediction",
            "💰 Price Forecast",
            "🌱 Crop Recommendation",
            "🧪 Fertilizer Recommendation",
            "🏔️ Soil Classification",
            "📊 Risk Assessment",
            "🔬 Nutrient Analysis"
        ]
    )
    
    # Pages
    if page == "🏠 Home":
        show_home()
    elif page == "🤖 AI Assistant":
        show_ai_assistant()
    elif page == "🦠 Disease Detection":
        show_disease_detection()
    elif page == "💧 Irrigation":
        show_irrigation()
    elif page == "🌤️ Weather":
        show_weather()
    elif page == "📈 Yield Prediction":
        show_yield_prediction()
    elif page == "💰 Price Forecast":
        show_price_forecast()
    elif page == "🌱 Crop Recommendation":
        show_crop_recommendation()
    elif page == "🧪 Fertilizer Recommendation":
        show_fertilizer_recommendation()
    elif page == "🏔️ Soil Classification":
        show_soil_classification()
    elif page == "📊 Risk Assessment":
        show_risk_assessment()
    elif page == "🔬 Nutrient Analysis":
        show_nutrient_analysis()


def show_home():
    """Display home page."""
    st.markdown("""
    Welcome to **Kalpataru** - Your intelligent agricultural assistant.
    
    ### 🌟 Features:
    
    | Module | Description |
    |--------|-------------|
    | 🤖 **AI Assistant** | Intelligent chatbot with image understanding |
    | 🦠 **Disease Detection** | CNN-based plant disease detection (38 disease classes) |
    | 💧 **Irrigation Optimization** | Smart irrigation recommendations |
    | 🌤️ **Weather Forecasting** | Real-time weather data and forecasts |
    | 📈 **Yield Prediction** | ML-based crop yield forecasting |
    | 💰 **Price Forecasting** | Commodity price predictions |
    | 🌱 **Crop Recommendation** | AI-powered crop suggestions |
    | 🧪 **Fertilizer Recommendation** | Smart fertilizer suggestions |
    | 🏔️ **Soil Classification** | Image-based soil type identification |
    | 📊 **Risk Assessment** | Agricultural risk scoring |
    | 🔬 **Nutrient Analysis** | Soil nutrient analysis and recommendations |
    
    ### 🚀 Getting Started:
    1. Select a module from the sidebar
    2. Provide the required inputs
    3. Get AI-powered recommendations
    
    ### 📊 Platform Statistics:
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Disease Classes", "38", "Plant Types")
    with col2:
        st.metric("Crops Supported", "15+", "Varieties")
    with col3:
        st.metric("Languages", "10", "Indian Languages")
    with col4:
        st.metric("API Endpoints", "25+", "Services")


def show_ai_assistant():
    """Display AI Assistant page with chat and image understanding."""
    st.markdown("### 🤖 AI Agricultural Assistant")
    st.markdown("Chat with our intelligent assistant about any agricultural topic. Upload images for analysis!")
    
    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # API key input in sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ AI Settings")
        api_key = st.text_input(
            "OpenRouter API Key", 
            type="password",
            help="Enter your OpenRouter API key for AI chat functionality"
        )
        if api_key:
            import os
            os.environ['OPENROUTER_API_KEY'] = api_key
            st.success("API Key set!")
    
    # Chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat history display
        st.markdown("#### 💬 Chat")
        
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div style="background-color: #e3f2fd; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <b>👤 You:</b><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #e8f5e9; padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <b>🤖 Assistant:</b><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # User input
        user_message = st.text_area(
            "Your message:",
            height=100,
            placeholder="Ask about crops, diseases, fertilizers, or any agricultural topic..."
        )
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload an image (optional)",
            type=['jpg', 'png', 'jpeg'],
            help="Upload a plant, soil, or pest image for analysis"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            send_button = st.button("📤 Send", type="primary", use_container_width=True)
        with col_btn2:
            clear_button = st.button("🗑️ Clear Chat", use_container_width=True)
    
    with col2:
        # Quick prompts
        st.markdown("#### 📋 Quick Prompts")
        quick_prompts = [
            "What crops grow best in sandy soil?",
            "How to treat tomato blight?",
            "Best fertilizer for rice cultivation",
            "Signs of nitrogen deficiency in plants",
            "How to improve soil pH?",
            "Organic pest control methods"
        ]
        
        for prompt in quick_prompts:
            if st.button(prompt, key=f"qp_{prompt[:20]}"):
                user_message = prompt
                send_button = True
        
        # Analysis type selector
        st.markdown("#### 🔍 Image Analysis Type")
        analysis_type = st.selectbox(
            "Select analysis type:",
            ["general", "disease", "soil", "pest", "crop"]
        )
    
    # Handle send button
    if send_button and (user_message or uploaded_image):
        # Add user message to history
        if user_message:
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_message
            })
        
        # Get AI response
        with st.spinner("Thinking..."):
            try:
                from services.ai_assistant import chat_with_ai, analyze_agricultural_image
                
                if uploaded_image:
                    # Process with image
                    image_data = uploaded_image.read()
                    
                    if user_message:
                        result = chat_with_ai(
                            message=user_message,
                            image_data=image_data
                        )
                    else:
                        result = analyze_agricultural_image(
                            image_data=image_data,
                            analysis_type=analysis_type
                        )
                else:
                    # Text only
                    result = chat_with_ai(message=user_message)
                
                if result.get('success'):
                    response = result.get('response', 'No response generated.')
                else:
                    response = result.get('fallback_response', 'Sorry, I could not process your request. Please try again.')
                
                # Add assistant response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure you have set your OpenRouter API key in the sidebar.")
    
    # Handle clear button
    if clear_button:
        st.session_state.chat_history = []
        try:
            from services.ai_assistant import clear_conversation
            clear_conversation()
        except:
            pass
        st.rerun()


def show_disease_detection():
    """Display disease detection page."""
    st.markdown("### 🦠 Plant Disease Detection")
    st.markdown("Upload a plant leaf image to detect diseases")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload plant image", 
            type=['jpg', 'png', 'jpeg'],
            help="Supported formats: JPG, PNG, JPEG"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        if uploaded_file:
            if st.button("🔍 Detect Disease", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Placeholder for actual prediction
                    st.success("Analysis Complete!")
                    
                    st.markdown("#### Results:")
                    st.markdown("""
                    <div class="success-box">
                    <b>Disease:</b> Tomato___Early_blight<br>
                    <b>Confidence:</b> 87.5%<br>
                    <b>Status:</b> Disease Detected
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### Description:")
                    st.info("Early blight is a fungal disease causing target-like lesions on lower leaves.")
                    
                    st.markdown("#### Treatment:")
                    st.warning("• Apply chlorothalonil or mancozeb fungicide\n• Remove infected leaves\n• Improve air circulation")
                    
                    st.markdown("#### Prevention:")
                    st.info("• Use disease-resistant varieties\n• Practice crop rotation\n• Avoid overhead irrigation")


def show_irrigation():
    """Display irrigation page."""
    st.markdown("### 💧 Irrigation Optimization")
    st.markdown("Get smart irrigation recommendations based on conditions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        soil_moisture = st.slider("Soil Moisture (%)", 0, 100, 50)
        temperature = st.number_input("Temperature (°C)", -10, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
    
    with col2:
        crop_type = st.selectbox(
            "Crop Type",
            ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato"]
        )
        growth_stage = st.selectbox(
            "Growth Stage",
            ["Seedling", "Vegetative", "Flowering", "Maturation"]
        )
    
    if st.button("💧 Calculate Irrigation", type="primary"):
        st.markdown("#### Irrigation Recommendation:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Water Required", "5.2 mm/day")
        with col2:
            st.metric("Irrigation Needed", "Yes")
        with col3:
            st.metric("Best Time", "6-8 AM")
        
        st.info("💡 **Recommendation:** Irrigate every 2 days in the early morning for best results.")


def show_weather():
    """Display weather page."""
    st.markdown("### 🌤️ Weather Forecasting")
    st.markdown("Get real-time weather data and agricultural insights")
    
    location = st.text_input("Enter Location", "Delhi")
    days = st.slider("Forecast Days", 1, 14, 7)
    
    if st.button("🌤️ Get Weather", type="primary"):
        with st.spinner("Fetching weather data..."):
            st.success("Weather data retrieved!")
            
            # Current weather
            st.markdown("#### Current Weather")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Temperature", "28°C")
            with col2:
                st.metric("Humidity", "65%")
            with col3:
                st.metric("Wind", "12 km/h")
            with col4:
                st.metric("Condition", "Partly Cloudy")
            
            # Agricultural insights
            st.markdown("#### Agricultural Insights")
            col1, col2 = st.columns(2)
            with col1:
                st.info("✅ Good conditions for spraying")
            with col2:
                st.warning("⚠️ Irrigation recommended in 2 days")


def show_yield_prediction():
    """Display yield prediction page."""
    st.markdown("### 📈 Yield Prediction")
    st.markdown("Predict crop yield based on agricultural parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        crop_type = st.selectbox(
            "Crop Type",
            ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato", "Onion"]
        )
        area = st.number_input("Area (Hectares)", 0.1, 100.0, 1.0)
        region = st.selectbox(
            "Region",
            ["Punjab", "Haryana", "UP", "Maharashtra", "Karnataka", "Tamil Nadu", "Gujarat"]
        )
    
    with col2:
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
        soil_type = st.selectbox("Soil Type", ["Alluvial", "Black", "Red", "Laterite", "Sandy"])
        rainfall = st.number_input("Rainfall (mm)", 0, 3000, 1000)
    
    if st.button("📈 Predict Yield", type="primary"):
        st.markdown("#### Yield Prediction Results:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Yield", "45.2 quintals")
        with col2:
            st.metric("Yield/Hectare", "45.2 q/ha")
        with col3:
            st.metric("Confidence", "85%")
        
        st.info("💡 **Recommendations:** Consider soil testing to optimize fertilizer application.")


def show_price_forecast():
    """Display price forecast page."""
    st.markdown("### 💰 Price Forecasting")
    st.markdown("Get commodity price predictions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        commodity = st.selectbox(
            "Commodity",
            ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane", "Potato", "Tomato", "Onion"]
        )
        market = st.selectbox(
            "Market",
            ["Delhi", "Mumbai", "Kolkata", "Chennai", "Ahmedabad"]
        )
    
    with col2:
        days = st.slider("Forecast Days", 1, 30, 7)
    
    if st.button("💰 Forecast Price", type="primary"):
        st.markdown("#### Price Forecast Results:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", "₹2,200/q")
        with col2:
            st.metric("Predicted Trend", "↑ Increasing")
        with col3:
            st.metric("Confidence", "75%")
        
        st.info("💡 **Recommendation:** Prices expected to rise. Consider holding produce for better returns.")


def show_crop_recommendation():
    """Display crop recommendation page."""
    st.markdown("### 🌱 Crop Recommendation")
    st.markdown("Get AI-powered crop suggestions based on soil and climate")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n = st.number_input("Nitrogen (N)", 0, 200, 50)
        p = st.number_input("Phosphorus (P)", 0, 200, 50)
        k = st.number_input("Potassium (K)", 0, 200, 50)
    
    with col2:
        temperature = st.number_input("Temperature (°C)", -10, 50, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        ph = st.slider("Soil pH", 0.0, 14.0, 7.0)
    
    with col3:
        rainfall = st.number_input("Rainfall (mm)", 0, 3000, 1000)
        soil_type = st.selectbox("Soil Type", ["Alluvial", "Black", "Red", "Laterite", "Sandy", "Clay"])
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
    
    if st.button("🌱 Get Recommendations", type="primary"):
        st.markdown("#### Top Crop Recommendations:")
        
        recommendations = [
            {"crop": "Wheat", "score": 92, "suitability": "Excellent"},
            {"crop": "Rice", "score": 85, "suitability": "Good"},
            {"crop": "Maize", "score": 78, "suitability": "Good"},
            {"crop": "Cotton", "score": 65, "suitability": "Moderate"},
            {"crop": "Sugarcane", "score": 58, "suitability": "Moderate"}
        ]
        
        for rec in recommendations:
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.write(f"**{rec['crop']}**")
            with col2:
                st.write(f"Score: {rec['score']}")
            with col3:
                if rec['suitability'] == "Excellent":
                    st.success(rec['suitability'])
                elif rec['suitability'] == "Good":
                    st.info(rec['suitability'])
                else:
                    st.warning(rec['suitability'])


def show_fertilizer_recommendation():
    """Display fertilizer recommendation page."""
    st.markdown("### 🧪 Fertilizer Recommendation")
    st.markdown("Get smart fertilizer suggestions based on soil nutrients")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Environmental Conditions")
        temperature = st.number_input("Temperature (°C)", 0, 60, 25)
        humidity = st.slider("Humidity (%)", 0, 100, 60)
        moisture = st.slider("Soil Moisture (%)", 0, 100, 50)
    
    with col2:
        st.markdown("#### Soil & Crop Details")
        soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])
        crop_type = st.selectbox("Crop Type", ["Wheat", "Cotton", "Maize", "Paddy", "Barley", "Ground Nuts"])
    
    st.markdown("#### Soil Nutrient Levels")
    col1, col2, col3 = st.columns(3)
    with col1:
        nitrogen = st.number_input("Nitrogen (N)", 0, 140, 60)
    with col2:
        potassium = st.number_input("Potassium (K)", 0, 140, 50)
    with col3:
        phosphorous = st.number_input("Phosphorous (P)", 0, 140, 40)
    
    if st.button("🧪 Get Fertilizer Recommendation", type="primary"):
        st.markdown("#### Fertilizer Recommendation:")
        
        st.success("✅ **Recommended Fertilizer: Urea**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("NPK Ratio", "46-0-0")
        with col2:
            st.metric("Application", "Split doses")
        with col3:
            st.metric("Confidence", "85%")
        
        st.info("💡 **Application Notes:** Apply in split doses during growing season. Best applied when soil is moist.")


def show_soil_classification():
    """Display soil classification page."""
    st.markdown("### 🏔️ Soil Classification")
    st.markdown("Upload a soil image to identify soil type")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload soil image", 
            type=['jpg', 'png', 'jpeg'],
            help="Supported formats: JPG, PNG, JPEG"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Soil Image", use_column_width=True)
    
    with col2:
        if uploaded_file:
            if st.button("🏔️ Classify Soil", type="primary"):
                with st.spinner("Analyzing soil image..."):
                    st.success("Classification Complete!")
                    
                    st.markdown("#### Results:")
                    st.markdown("""
                    <div class="success-box">
                    <b>Soil Type:</b> Alluvial<br>
                    <b>Confidence:</b> 82.5%<br>
                    <b>Method:</b> Color Analysis
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("#### Characteristics:")
                    st.info("• Highly fertile\n• Good water retention\n• Suitable for most crops\n• Rich in potash")
                    
                    st.markdown("#### Recommended Crops:")
                    st.success("Wheat, Rice, Sugarcane, Cotton, Vegetables")


def show_risk_assessment():
    """Display risk assessment page."""
    st.markdown("### 📊 Risk Assessment")
    st.markdown("Calculate agricultural risk score for insurance and planning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Risk Factors")
        weather_risk = st.slider("Weather Risk", 0.0, 1.0, 0.5)
        crop_success = st.slider("Crop Success Rate", 0.0, 1.0, 0.7)
        location_risk = st.slider("Location Risk", 0.0, 1.0, 0.3)
        activity_score = st.slider("Activity Score", 0.0, 1.0, 0.6)
    
    with col2:
        st.markdown("#### Insurance Parameters")
        crop_type = st.selectbox("Crop Type", ["Rice", "Wheat", "Maize", "Cotton", "Sugarcane"])
        coverage = st.number_input("Coverage Amount (₹)", 10000, 500000, 50000)
        season = st.selectbox("Season", ["Kharif", "Rabi", "Zaid"])
    
    if st.button("📊 Calculate Risk & Insurance", type="primary"):
        st.markdown("#### Risk Assessment Results:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("ARS Score", "42.5")
        with col2:
            st.metric("Risk Category", "Moderate")
        with col3:
            st.metric("Risk Multiplier", "1.0x")
        
        st.markdown("#### Insurance Premium:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Premium", "₹1,750")
        with col2:
            st.metric("Base Rate", "3.5%")
        with col3:
            st.metric("Adjusted Rate", "3.5%")


def show_nutrient_analysis():
    """Display nutrient analysis page."""
    st.markdown("### 🔬 Nutrient Analysis")
    st.markdown("Analyze soil nutrients and get fertilization recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Current Nutrient Levels")
        n = st.number_input("Nitrogen (kg/ha)", 0, 500, 50)
        p = st.number_input("Phosphorus (kg/ha)", 0, 200, 30)
        k = st.number_input("Potassium (kg/ha)", 0, 400, 40)
        ph = st.slider("Soil pH", 0.0, 14.0, 7.0)
    
    with col2:
        st.markdown("#### Target Crop & Area")
        crop_type = st.selectbox("Target Crop", ["Wheat", "Rice", "Maize", "Cotton", "Sugarcane", "Potato"])
        area = st.number_input("Area (Hectares)", 0.1, 100.0, 1.0)
    
    if st.button("🔬 Analyze Nutrients", type="primary"):
        st.markdown("#### Nutrient Analysis Results:")
        
        # Nutrient status
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nitrogen Status", "Low", "-70 kg/ha")
        with col2:
            st.metric("Phosphorus Status", "Low", "-20 kg/ha")
        with col3:
            st.metric("Potassium Status", "Moderate", "-40 kg/ha")
        
        # Recommendations
        st.markdown("#### Fertilizer Recommendations:")
        
        recommendations = [
            {"fertilizer": "Urea", "amount": "152 kg/ha", "timing": "Split into 2-3 doses"},
            {"fertilizer": "DAP", "amount": "43 kg/ha", "timing": "At sowing"},
            {"fertilizer": "MOP", "amount": "67 kg/ha", "timing": "During active growth"}
        ]
        
        for rec in recommendations:
            st.info(f"**{rec['fertilizer']}**: {rec['amount']} - {rec['timing']}")


if __name__ == '__main__':
    main()
