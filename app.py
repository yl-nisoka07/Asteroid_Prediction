import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Asteroid Hazard Predictor",
    page_icon="🌌",
    layout="centered"
)

# Title
st.title("🌌 Asteroid Hazard Predictor")
st.write("Enter asteroid parameters to predict if it's hazardous to Earth")

# Initialize session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.model_loaded = False

# Define the expected features
EXPECTED_FEATURES = [
    'Absolute Magnitude',
    'Est Dia in KM(min)',
    'Est Dia in KM(max)',
    'Relative Velocity km per sec',
    'Miss Dist.(Astronomical)',
    'Orbit Uncertainity',
    'Minimum Orbit Intersection',
    'Eccentricity'
]

# Function to create demo model
@st.cache_resource
def create_demo_model():
    """Create a demo model for demonstration purposes"""
    np.random.seed(42)
    
    # Create dummy model with correct number of features
    n_features = len(EXPECTED_FEATURES)
    X_dummy = np.random.randn(1000, n_features)
    y_dummy = np.random.randint(0, 2, 1000)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    scaler = StandardScaler()
    scaler.fit(X_dummy)
    
    return model, scaler

# Load or create model
if not st.session_state.model_loaded:
    try:
        # Try to load saved model
        model = joblib.load('best_asteroid_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.success("✅ Model loaded successfully!")
    except FileNotFoundError:
        # Create demo model if saved model not found
        model, scaler = create_demo_model()
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.warning("⚠️ Using demo model. Run asteroid.py first to train the actual model.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        model, scaler = create_demo_model()
        st.session_state.model = model
        st.session_state.scaler = scaler
    
    st.session_state.model_loaded = True

# Input form
st.header("Enter Asteroid Parameters")

col1, col2 = st.columns(2)

with col1:
    absolute_magnitude = st.number_input(
        "Absolute Magnitude (10-35)", 
        min_value=10.0, 
        max_value=35.0, 
        value=20.0,
        help="Lower values = larger/brighter asteroids"
    )
    
    est_diameter_min = st.number_input(
        "Min Diameter (km)", 
        min_value=0.001, 
        max_value=50.0, 
        value=0.1,
        format="%.3f"
    )
    
    est_diameter_max = st.number_input(
        "Max Diameter (km)", 
        min_value=0.001, 
        max_value=50.0, 
        value=0.3,
        format="%.3f"
    )
    
    relative_velocity = st.number_input(
        "Velocity (km/s)", 
        min_value=1.0, 
        max_value=50.0, 
        value=15.0
    )

with col2:
    miss_distance_au = st.number_input(
        "Miss Distance (AU)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1,
        format="%.4f",
        help="Distance from Earth (1 AU = Earth-Sun distance)"
    )
    
    orbit_uncertainty = st.number_input(
        "Orbit Uncertainty (0-10)", 
        min_value=0, 
        max_value=10, 
        value=2
    )
    
    min_orbit_intersection = st.number_input(
        "Min Orbit Intersection", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.05,
        format="%.4f"
    )
    
    eccentricity = st.number_input(
        "Eccentricity (0-1)", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.3,
        format="%.3f",
        help="0 = circular orbit, 1 = very elongated"
    )

# Validation
def validate_inputs():
    """Validate user inputs"""
    errors = []
    
    if est_diameter_min >= est_diameter_max:
        errors.append("Minimum diameter must be less than maximum diameter")
    
    if miss_distance_au < 0:
        errors.append("Miss distance cannot be negative")
    
    if relative_velocity <= 0:
        errors.append("Relative velocity must be positive")
    
    return errors

# Predict button
if st.button("🚀 PREDICT", type="primary"):
    # Validate inputs
    validation_errors = validate_inputs()
    
    if validation_errors:
        st.error("Please fix the following errors:")
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        # Prepare input data
        input_features = np.array([[
            absolute_magnitude,
            est_diameter_min,
            est_diameter_max,
            relative_velocity,
            miss_distance_au,
            orbit_uncertainty,
            min_orbit_intersection,
            eccentricity
        ]])
        
        try:
            # Scale the input
            input_scaled = st.session_state.scaler.transform(input_features)
            
            # Make prediction
            prediction = st.session_state.model.predict(input_scaled)[0]
            probability = st.session_state.model.predict_proba(input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Result")
            
            hazard_prob = probability[1] * 100
            
            if prediction == 1:
                st.error(f"⚠️ **POTENTIALLY HAZARDOUS ASTEROID**")
                st.error(f"**Hazard Probability: {hazard_prob:.1f}%**")
                
                if hazard_prob > 70:
                    st.error("🚨 **HIGH RISK** - Requires immediate attention")
                elif hazard_prob > 30:
                    st.warning("⚠️ **MODERATE RISK** - Requires monitoring")
                else:
                    st.info("ℹ️ **LOW-MODERATE RISK** - Continue observation")
                    
            else:
                st.success(f"✅ **NON-HAZARDOUS ASTEROID**")
                st.success(f"**Hazard Probability: {hazard_prob:.1f}%**")
                st.success("🌍 **SAFE** - No immediate threat to Earth")
            
            # Show key parameters
            st.subheader("Key Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Size Range", f"{est_diameter_min:.3f} - {est_diameter_max:.3f} km")
                st.metric("Velocity", f"{relative_velocity} km/s")
                st.metric("Miss Distance", f"{miss_distance_au:.4f} AU")
                st.metric("Brightness", f"{absolute_magnitude}")
            
            with col2:
                st.metric("Orbit Uncertainty", f"{orbit_uncertainty}")
                st.metric("Min Orbit Intersection", f"{min_orbit_intersection:.4f}")
                st.metric("Eccentricity", f"{eccentricity:.3f}")
                
                # Risk interpretation
                if miss_distance_au < 0.05:
                    st.warning("⚠️ Very close approach")
                elif miss_distance_au < 0.1:
                    st.info("ℹ️ Close approach")
                else:
                    st.success("✅ Safe distance")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

# Help section
with st.expander("ℹ️ Help - Understanding the Parameters"):
    st.markdown("""
    **Absolute Magnitude**: Brightness measure (10-35, lower = larger/brighter)
    
    **Diameter**: Physical size of the asteroid in kilometers
    
    **Velocity**: Speed relative to Earth in km/s
    
    **Miss Distance**: Closest approach distance in Astronomical Units (AU)
    - 1 AU = Earth-Sun distance (~150 million km)
    - Values < 0.05 AU are considered "close approaches"
    
    **Orbit Uncertainty**: How well we know the orbit (0-10, higher = more uncertain)
    
    **Min Orbit Intersection**: Minimum possible distance between orbits
    
    **Eccentricity**: How elliptical the orbit is (0 = circular, 1 = very elongated)
    """)

# Add some example scenarios in the help section
with st.expander("📊 Example Scenarios"):
    st.markdown("""
    **Safe Asteroid Example:**
    - Absolute Magnitude: 28.0 (small, dim)
    - Diameter: 0.01-0.05 km (very small)
    - Velocity: 8.0 km/s (slow)
    - Miss Distance: 0.2 AU (far)
    - Low orbit uncertainty, high min intersection distance
    
    **Potentially Hazardous Example:**
    - Absolute Magnitude: 15.0 (large, bright)
    - Diameter: 2.0-3.5 km (large)
    - Velocity: 35.0 km/s (fast)
    - Miss Distance: 0.005 AU (very close)
    - High orbit uncertainty, low min intersection distance
    
    *Try entering these values to see how the model responds!*
    """)

# Footer
st.markdown("---")
st.markdown("🌌 **Asteroid Hazard Predictor** | Built with Streamlit")
st.markdown("*Adjust the parameters above and click PREDICT to analyze asteroid risk*")