import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. PAGE CONFIGURATION & THEMING
st.set_page_config(
    page_title="Shopper Spectrum Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: 700; color: #1E3A8A; margin-bottom: 20px; }
    .section-desc { font-size: 16px; color: #4B5563; margin-bottom: 30px; }
    .card { padding: 20px; border-radius: 10px; background-color: #F3F4F6; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# 2. CACHED RESOURCE LOADING
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler_transformer.pkl")
        similarity = joblib.load("product_similarity_matrix.pkl")
        return model, scaler, similarity
    except FileNotFoundError as e:
        st.error(f"❌ Missing serialized asset files. Ensure they are in the same folder.")
        return None, None, None

kmeans, scaler, similarity_df = load_assets()

# 3. SIDEBAR NAVIGATION
st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate Engine Modules:", ["🎯 Product Recommendations", "👥 Customer Segmentation"])

# MODULE 1: PRODUCT RECOMMENDATION ENGINE
if page == "🎯 Product Recommendations":
    st.markdown('<div class="main-title">🎯 Product Recommendation Engine</div>', unsafe_allow_html=True)
    
    if similarity_df is not None:
        search_query = st.selectbox("Select or Type a Product Name:", options=list(similarity_df.index), index=0)
        
        if st.button("Get Recommendations", type="primary"):
            st.markdown("### Top 5 Recommended Items")
            scores = similarity_df[search_query].sort_values(ascending=False)
            recommendations = scores.iloc[1:6].reset_index()
            recommendations.columns = ['Product Description', 'Similarity Score']
            
            for idx, row in recommendations.iterrows():
                st.markdown(f"""
                <div class="card">
                    <span style="font-weight:600; font-size:16px;">🔹 {row['Product Description']}</span><br>
                    <span style="font-size:14px; color:#6B7280;">Behavioral Match Score: </span>
                    <span style="font-weight:600; color:#0D9488;">{row['Similarity Score']:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

# MODULE 2: CUSTOMER LIFECYCLE SEGMENTATION
elif page == "👥 Customer Segmentation":
    st.markdown('<div class="main-title">👥 Customer Lifecycle Segmentation</div>', unsafe_allow_html=True)
    
    if kmeans is not None and scaler is not None:
        col1, col2, col3 = st.columns(3)
        with col1: recency = st.number_input("Recency (Days since last order):", min_value=1, max_value=365, value=30)
        with col2: frequency = st.number_input("Frequency (Total orders placed):", min_value=1, max_value=500, value=5)
        with col3: monetary = st.number_input("Monetary Value (Total dollar spend):", min_value=0.0, max_value=500000.0, value=500.0)
            
        if st.button("Predict Lifecycle Segment", type="primary"):
            raw_features = np.array([[recency, frequency, monetary]])
            scaled_features = scaler.transform(raw_features)
            cluster_id = kmeans.predict(scaled_features)[0]
            
            cluster_mapping = {
                0: {"label": "At-Risk", "color": "#EF4444", "desc": "Inactive accounts with low frequency. Needs active win-back promotions."},
                1: {"label": "Regular", "color": "#3B82F6", "desc": "Steady shoppers with moderate spend. Great fit for standard engagement programs."},
                2: {"label": "High-Value", "color": "#10B981", "desc": "Top-tier spenders (Whales). Priority support and early access programs recommended."},
                3: {"label": "Champions / Power Shoppers", "color": "#F59E0B", "desc": "Hyper-frequent power buyers. Excellent targets for loyalty rewards."}
            }
            target = cluster_mapping.get(cluster_id, {"label": "Unknown", "color": "#6B7280", "desc": "Undefined vector profile."})
            
            st.markdown("---")
            st.markdown(f"""
            <div style="padding: 25px; border-left: 8px solid {target['color']}; background-color: #F9FAFB; border-radius: 4px;">
                <span style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #6B7280; font-weight: 600;">Predicted Category</span><br>
                <span style="font-size: 32px; font-weight: 700; color: {target['color']};">{target['label']}</span><br><br>
                <p style="font-size: 16px; color: #374151; margin: 0;"><strong>Strategic Playbook:</strong> {target['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
