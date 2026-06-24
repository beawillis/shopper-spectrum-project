import streamlit as st
import pandas as pd
import joblib
import urllib.request

# ==========================================
# 1. PAGE CONFIGURATION & THEMING
# ==========================================
st.set_page_config(
    page_title="Shopper Spectrum Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI styling
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: 700; color: #1E3A8A; margin-bottom: 20px; }
    .section-desc { font-size: 16px; color: #4B5563; margin-bottom: 30px; }
    .card { padding: 20px; border-radius: 10px; background-color: #F3F4F6; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); margin-bottom: 15px; }
    .metric-val { font-size: 24px; font-weight: 600; color: #0D9488; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CLOUD RESOURCE LOADING
# ==========================================
@st.cache_resource
def load_assets_via_cloud():
    import os # Moved import os here
    pkl_filename = "product_similarity_matrix.pkl"

    google_drive_id = "1mjmluvsQu0AbPql2YeWJ8ybxqxPZZ4WH"
    download_url = f"https://docs.google.com/uc?export=download&id={google_drive_id}"

    try:
        # Download the file if it does not exist locally in the Streamlit cloud container instance
        if not os.path.exists(pkl_filename):
            with st.spinner("Downloading heavy recommendation matrix from secure remote storage (This happens once on startup)..."):
                # Configure custom user agent to avoid potential remote server request blocks
                opener = urllib.request.build_opener()
                opener.addheaders = [('User-agent', 'Mozilla/5.0')]
                urllib.request.install_opener(opener)

                urllib.request.urlretrieve(download_url, pkl_filename)

        # Load assets into current deployment workspace memory map
        model = joblib.load("kmeans_model.pkl")
        scaler = joblib.load("scaler_transformer.pkl")
        similarity_df = joblib.load(pkl_filename)

        return model, scaler, similarity_df
    except Exception as e:
        st.error(f"❌ Asset recovery lifecycle initialization crash: {str(e)}")
        return None, None, None

kmeans, scaler, similarity_df = load_assets_via_cloud()

# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🛒 Shopper Spectrum")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate Engine Modules:",
    ["🎯 Product Recommendations", "👥 Customer Segmentation"]
)
st.sidebar.markdown("---")
st.sidebar.info("Designed as an integrated business intelligence & personalization platform.")

# ==========================================
# MODULE 1: PRODUCT RECOMMENDATION ENGINE
# ==========================================
if page == "🎯 Product Recommendations":
    st.markdown('<div class="main-title">🎯 Product Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Enter an item name below to find the top 5 highly associated products based on collaborative purchasing behavior patterns.</p>', unsafe_allow_html=True)

    if similarity_df is not None:
        search_query = st.selectbox(
            "Select or Type a Product Name:",
            options=list(similarity_df.index),
            index=0
        )

        if st.button("Get Recommendations", type="primary"):
            st.markdown("### Top 5 Recommended Items")

            # Extract recommendation matrix vectors
            scores = similarity_df[search_query].sort_values(ascending=False)
            recommendations = scores.iloc[1:6].reset_index()
            recommendations.columns = ['Product Description', 'Similarity Score']

            # Render visually striking recommendation cards
            for idx, row in recommendations.iterrows():
                st.markdown(f"""
                <div class="card">
                    <span style="font-weight:600; font-size:16px;">🔹 {row['Product Description']}</span><br>
                    <span style="font-size:14px; color:#6B7280;">Behavioral Match Score: </span>
                    <span style="font-weight:600; color:#0D9488;">{row['Similarity Score']:.4f}</span>
                </div>
                """, unsafe_allow_html=True)

# ==========================================
# MODULE 2: CUSTOMER LIFECYCLE SEGMENTATION
# ==========================================
elif page == "👥 Customer Segmentation":
    st.markdown('<div class="main-title">👥 Customer Lifecycle Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Input a customer\'s raw operational features to instantly classify their lifecycle value tier.</p>', unsafe_allow_html=True)

    if kmeans is not None and scaler is not None:
        col1, col2, col3 = st.columns(3)

        with col1:
            recency = st.number_input("Recency (Days since last order):", min_value=1, max_value=365, value=30)
        with col2:
            frequency = st.number_input("Frequency (Total orders placed):", min_value=1, max_value=500, value=5)
        with col3:
            monetary = st.number_input("Monetary (Total spend last year):", min_value=0.0, max_value=200000.0, value=100.0)

        if st.button("Classify Customer Segment", type="primary"):
            input_data = pd.DataFrame([[recency, frequency, monetary]], columns=['Recency', 'Frequency', 'Monetary'])
            scaled_data = scaler.transform(input_data)
            segment_id = kmeans.predict(scaled_data)[0]

            # Mapping numerical cluster ID to semantic segment names
            cluster_mapping = {
                0: "At-Risk",
                1: "Regular",
                2: "High-Value",
                3: "Occasional"
            }
            segment_name = cluster_mapping.get(segment_id, "Unknown")

            st.markdown(f"<div class='card' style='background-color: #E0F2F1;'>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:18px; font-weight:600;'>Identified Segment: <span class='metric-val'>{segment_name}</span></p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Segmentation model assets not loaded. Please ensure `kmeans_model.pkl` and `scaler_transformer.pkl` exist.")
