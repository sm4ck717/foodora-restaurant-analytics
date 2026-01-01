import streamlit as st
import pandas as pd
import pickle
from catboost import CatBoostRegressor
import google.generativeai as genai

# Configuration
st.set_page_config(page_title="Foodora Eats Analytics Suite", layout="wide", initial_sidebar_state="expanded")

# --- Load Assets ---
@st.cache_resource
def load_assets():
    # Task 1
    model_t1 = CatBoostRegressor()
    model_t1.load_model("catboost_model.cbm")
    with open("task1_options.pkl", "rb") as f:
        t1_data = pickle.load(f)

    # Task 3
    with open("recommendations.pkl", "rb") as f:
        recs_dict = pickle.load(f)
    df_recs = pd.read_csv("restaurants_for_rec.csv")

    # Task 4
    df_rankings = pd.read_csv("ranking_results.csv")

    with open("lightgbm_ranker.pkl", "rb") as f:
        ranker_model = pickle.load(f)

    with open("task4_features.pkl", "rb") as f:
        ranker_features = pickle.load(f)

    return (
        model_t1,
        t1_data,
        recs_dict,
        df_recs,
        df_rankings,
        ranker_model,
        ranker_features
    )

try:
    (
    model_t1,
    t1_data,
    recs_dict,
    df_recs,
    df_rankings,
    ranker_model,
    ranker_features
) = load_assets()

except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.title("Restaurant Analytics")
st.sidebar.info("Select a module below:")
page = st.sidebar.radio("Go to", [
    "1. Predict Quality (Regression)", 
    "2. Cuisine Strategy (Clustering)", 
    "3. Recommender System", 
    "4. Partner Ranking"
])

# ==========================================
# PAGE 1: PREDICT QUALITY
# ==========================================
if page == "1. Predict Quality (Regression)":
    st.title("🍽️ Predict New Restaurant Quality")
    st.markdown("Use the **CatBoost Regression** model to estimate the aggregate rating for a new restaurant.")

    # --- API KEY SETUP ---
    api_key = st.sidebar.text_input("🔑 Enter Gemini API Key", type="password")

    # --- DATA LOADING (Force Load for Cuisines) ---
    if 'df' not in locals() or 'df' not in st.session_state:
        try:
            # Attempt to load the main dataset for cuisine list
            df = pd.read_csv("Foodora Data.csv", encoding='latin-1')
            st.session_state['df'] = df
        except Exception:
            df = pd.DataFrame() # Fallback

    if 'df' in st.session_state:
        df = st.session_state['df']

    # Initialize Session State
    if 'lat_val' not in st.session_state: st.session_state.lat_val = 0.0
    if 'lon_val' not in st.session_state: st.session_state.lon_val = 0.0
    if 'prediction_result' not in st.session_state: st.session_state.prediction_result = None
    if 'input_context' not in st.session_state: st.session_state.input_context = {}

    geo_map = t1_data["geo_map"]

    # --- EXTRACT CUISINES ---
    try:
        if not df.empty and "Cuisines" in df.columns:
            raw_cuisines = df["Cuisines"].dropna().astype(str)
            unique_cuisines = set()
            for c in raw_cuisines:
                parts = [x.strip() for x in c.replace("|", ",").split(",")] 
                unique_cuisines.update(parts)
            cuisine_options = sorted(list(unique_cuisines))
        else:
            cuisine_options = ["North Indian", "Chinese", "Italian", "Continental"] 
    except:
        cuisine_options = ["North Indian", "Chinese"]

    # --- INPUT FORM ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Location Details")
        country = st.selectbox("Country", sorted(geo_map.keys()))
        cities = sorted(geo_map[country].keys())
        city = st.selectbox("City", cities)
        localities = sorted(geo_map[country][city])
        locality = st.selectbox("Locality", localities)
        
        def fetch_coords():
            from geopy.geocoders import Nominatim
            geolocator = Nominatim(user_agent="restaurant_predictor_app")
            try:
                loc_string = f"{locality}, {city}, {country}"
                location = geolocator.geocode(loc_string)
                if location:
                    st.session_state.lat_val = location.latitude
                    st.session_state.lon_val = location.longitude
                    st.toast(f"📍 Found: {location.latitude:.4f}, {location.longitude:.4f}")
                else:
                    st.toast("⚠️ Location not found.")
            except:
                st.error("Error fetching location.")

        st.button("📍 Auto-Fetch Coordinates", on_click=fetch_coords)
        lat = st.number_input("Latitude", format="%.6f", key="lat_val")
        lon = st.number_input("Longitude", format="%.6f", key="lon_val")

    with col2:
        st.subheader("🍕 Restaurant Features")
        selected_cuisine = st.selectbox(
            "Select Cuisine", 
            options=cuisine_options,
            index=cuisine_options.index("North Indian") if "North Indian" in cuisine_options else 0
        )
        price_range = st.selectbox("Price Range", t1_data["Price range"])
        votes = st.number_input("Projected Votes", min_value=0, value=50)
        
        c1, c2 = st.columns(2)
        has_table = c1.selectbox("Table Booking", ["Yes", "No"])
        has_online = c2.selectbox("Online Delivery", ["Yes", "No"])
        is_delivering = c1.selectbox("Delivering Now", ["Yes", "No"])
        switch_menu = c2.selectbox("Switch to Menu", ["Yes", "No"])

    # --- PREDICTION LOGIC ---
    if st.button("Predict Rating", type="primary"):
        input_data = pd.DataFrame({
            "Country": [country],
            "City": [city],
            "Locality": [locality],
            "Cuisines": [selected_cuisine],
            "Price range": [str(price_range)],
            "Has Table booking": [has_table],
            "Has Online delivery": [has_online],
            "Is delivering now": [is_delivering],
            "Switch to order menu": [switch_menu],
            "Votes": [votes],
            "Longitude": [lon],
            "Latitude": [lat]
        })
        
        st.session_state.input_context = input_data.iloc[0].to_dict()
        
        # Predict
        model_input = input_data[t1_data["features"]]
        pred = model_t1.predict(model_input)[0]
        st.session_state.prediction_result = pred

    # --- DISPLAY RESULTS & GRAPHS ---
    if st.session_state.prediction_result is not None:
        pred = st.session_state.prediction_result
        
        st.divider()
        c_res1, c_res2 = st.columns([1, 3])
        c_res1.metric("Predicted Rating", f"{pred:.2f} / 5.0")
        
        if pred > 4.0:
            c_res2.success("🌟 This looks like a potential hit!")
        elif pred > 2.5:
            c_res2.warning("😐 Average performance expected.")
        else:
            c_res2.error("⚠️ Risk of low rating.")
            
        # --- NEW: FEATURE IMPORTANCE GRAPH ---
        st.subheader("📊 What Influenced This Prediction?")
        with st.expander("Show Feature Importance Graph", expanded=True):
            try:
                # Get importance from the model
                feature_importance = model_t1.get_feature_importance()
                feature_names = model_t1.feature_names_
                
                # Create DataFrame
                fi_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": feature_importance
                }).sort_values(by="Importance", ascending=False)
                
                # Plot using Streamlit's native bar chart
                st.bar_chart(fi_df.set_index("Feature"), color="#FF4B4B")
                st.caption("Higher bars mean this feature had a stronger impact on the model's decision.")
            except Exception as e:
                st.warning("Could not load feature importance.")
        
        st.markdown("---")
        
        # --- GEN AI SECTION ---
        st.subheader("🤖 AI Consultant")
        if st.button("✨ Get Gen AI Insights & Recommendations"):
            if not api_key:
                st.error("Please enter your Gemini API Key in the sidebar first!")
            else:
                try:
                    with st.spinner("Analyzing restaurant potential..."):
                        genai.configure(api_key=api_key)
                        
                        try:
                             model = genai.GenerativeModel('gemini-2.5-flash-lite')
                        except:
                             model = genai.GenerativeModel('gemini-pro')

                        ctx = st.session_state.input_context
                        prompt = f"""
                        I am planning to open a restaurant with the following details:
                        - **Location**: {ctx['Locality']}, {ctx['City']}
                        - **Cuisine**: {ctx['Cuisines']}
                        - **Price Range**: {ctx['Price range']} (1=Cheap, 4=Expensive)
                        - **Features**: Table Booking: {ctx['Has Table booking']}, Online Delivery: {ctx['Has Online delivery']}
                        - **Predicted Rating**: {pred:.2f} / 5.0

                        Act as a restaurant consultant. Based on the predicted rating and features:
                        1. Explain WHY the rating might be this way based on the mix of location, cuisine, and price.
                        2. Suggest 3 concrete actionable steps to improve the rating or ensure success.
                        3. Recommend a marketing tagline for this specific setup.
                        """
                        response = model.generate_content(prompt)
                        st.markdown(response.text)
                except Exception as e:
                    st.error(f"AI Error: {e}")

# ==========================================
# PAGE 2: CUISINE STRATEGY (CLUSTERING)
# ==========================================
if page == "2. Cuisine Strategy (Clustering)":
    st.title("🍜 Cuisine Cluster Analysis")
    st.markdown("Analyze cuisine performance using **K-Means Clustering** to find market gaps.")

    if "api_key" not in locals() or not api_key:
        api_key = st.sidebar.text_input("🔑 Enter Gemini API Key", type="password", key="api_key_p2")

    # --- DATA LOADING ---
    if 'df_clusters' not in st.session_state:
        try:
            st.session_state['df_clusters'] = pd.read_csv("cuisine_clusters.csv")
        except:
            st.warning("⚠️ 'cuisine_clusters.csv' not found. Using dummy data.")
            st.session_state['df_clusters'] = pd.DataFrame({
                "Cuisines": ["North Indian", "Chinese", "Italian", "Fast Food"],
                "cluster": [0, 1, 2, 1],
                "Aggregate rating": [4.2, 3.5, 4.0, 3.2],
                "Price range": [2.5, 1.5, 3.8, 1.2], 
                "Votes": [100, 50, 200, 150]
            })

    df_clusters = st.session_state['df_clusters']

    # --- SMART COLUMN MAPPING (FIXED PRICE RANGE) ---
    cols = df_clusters.columns.tolist()
    
    # Map Rating
    if "Aggregate rating" in cols: rate_col = "Aggregate rating"
    elif "avg_rating" in cols: rate_col = "avg_rating"
    else: rate_col = cols[2] 

    # Map Price Range (We look for Range, not Cost)
    if "avg_price_range" in cols: price_col = "avg_price_range"
    elif "Price range" in cols: price_col = "Price range"
    elif "avg_price" in cols: price_col = "avg_price" 
    else: price_col = cols[3]

    # Map Votes
    if "Votes" in cols: vote_col = "Votes"
    elif "avg_votes" in cols: vote_col = "avg_votes"
    else: vote_col = cols[4]

    # --- SUMMARY METRICS ---
    # Group by cluster 
    if len(df_clusters) > 10: 
        cluster_summary = df_clusters.groupby("cluster")[[rate_col, price_col, vote_col]].mean().reset_index()
    else:
        cluster_summary = df_clusters

    # --- UI DISPLAY ---
    tab1, tab2 = st.tabs(["📊 Cluster Insights", "🔍 Check Cuisine"])

    with tab1:
        st.subheader("Cluster Performance Groups")
        # Format price as a float score (e.g. 2.5) not currency
        st.dataframe(cluster_summary.style.format({price_col: "{:.2f}"}).highlight_max(axis=0, color='lightgreen'))
        st.caption(f"Note: '{price_col}' represents Price Range (1 = Pocket Friendly, 4 = Expensive).")

    with tab2:
        st.subheader("Identify Cuisine Strategy")
        
        try:
            raw_list = df_clusters["Cuisines"].dropna().astype(str).unique()
            all_cuisines = sorted(raw_list)
        except:
            all_cuisines = ["North Indian", "Chinese"]

        selected_cuisine = st.selectbox("Select a Cuisine to Analyze", all_cuisines)
        
        if 'cluster_context' not in st.session_state: st.session_state.cluster_context = None

        if st.button("Analyze Cuisine", type="primary"):
            subset = df_clusters[df_clusters["Cuisines"] == selected_cuisine]
            
            if not subset.empty:
                cid = subset.iloc[0]["cluster"]
                c_stats = cluster_summary[cluster_summary["cluster"] == cid].iloc[0]
                
                # Store context
                st.session_state.cluster_context = {
                    "cuisine": selected_cuisine,
                    "cluster_id": int(cid),
                    "avg_rating": c_stats[rate_col],
                    "avg_price_range": c_stats[price_col],
                    "avg_votes": c_stats[vote_col]
                }
                
                st.success(f"**{selected_cuisine}** belongs to **Cluster {cid}**")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Rating", f"{c_stats[rate_col]:.2f} / 5")
                c2.metric("Avg Price Range", f"{c_stats[price_col]:.1f} / 4.0") 
                c3.metric("Avg Votes", f"{c_stats[vote_col]:.0f}")
                
            else:
                st.error("Data not found for this cuisine.")
                st.session_state.cluster_context = None

        # --- GEN AI SECTION ---
        if st.session_state.cluster_context:
            st.markdown("---")
            st.subheader("🤖 AI Strategic Advisor")
            
            if st.button("✨ Get Strategy for this Cuisine"):
                if not api_key:
                    st.error("Please enter your Gemini API Key in the sidebar.")
                else:
                    try:
                        with st.spinner("Generating market strategy..."):
                            genai.configure(api_key=api_key)
                            try:
                                model = genai.GenerativeModel('gemini-2.5-flash-lite') 
                            except:
                                model = genai.GenerativeModel('gemini-pro')

                            ctx = st.session_state.cluster_context
                            
                            # UPDATED PROMPT FOR PRICE RANGE (1-4)
                            prompt = f"""
                            I am analyzing the cuisine '{ctx['cuisine']}' for a restaurant business.
                            It falls into Cluster {ctx['cluster_id']} which has the following average metrics:
                            - Average Rating: {ctx['avg_rating']:.2f} / 5.0
                            - Average Price Range: {ctx['avg_price_range']:.2f} (Scale: 1=Pocket Friendly, 4=Luxury)
                            - Average Votes (Popularity): {ctx['avg_votes']:.0f}

                            As a business strategist, provide a brief analysis:
                            1. **Verdict**: Is this a "Safe Bet" or "Risky" category? Explain why based on the rating vs popularity.
                            2. **Pricing Insight**: The cluster average is {ctx['avg_price_range']:.2f} on a scale of 4. Does this suggest a budget, mid-range, or premium market expectation?
                            3. **Competition**: What is the likely competition intensity for this cluster?
                            """
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"AI Error: {e}")

# ==========================================
# PAGE 3: RECOMMENDER
# ==========================================
elif page == "3. Recommender System":
    st.title("🔍 Restaurant Recommender")
    st.markdown("Content-based recommendation using **TF-IDF (Cuisines)** and **Nearest Neighbors**.")
    
    all_restaurants = sorted(df_recs["Restaurant Name"].unique())
    selected_rest = st.selectbox("Select a Restaurant you like:", all_restaurants)
    
    if st.button("Find Similar Restaurants"):
        if selected_rest in recs_dict:
            recommendations = recs_dict[selected_rest]
            st.subheader(f"Because you like '{selected_rest}':")
            results = df_recs[df_recs["Restaurant Name"].isin(recommendations)].copy()
            results = results.set_index("Restaurant Name").loc[recommendations].reset_index()
            st.dataframe(results[["Restaurant Name", "Cuisines", "Aggregate rating", "City"]], hide_index=True, use_container_width=True)
        else:
            st.warning("No recommendations found for this restaurant.")

# ==========================================
# PAGE 4: PARTNER RANKING
# ==========================================
elif page == "4. Partner Ranking":
    st.title("🏆 Partner Ranking Leaderboard")
    st.markdown("Ranking restaurants within a locality using **LightGBM LambdaRank**.")

    # -----------------------------
    # Filters
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        city = st.selectbox(
            "Select City",
            sorted(df_rankings["City"].unique())
        )

    with col2:
        locality = st.selectbox(
            "Select Locality",
            sorted(df_rankings[df_rankings["City"] == city]["Locality"].unique())
        )

    # -----------------------------
    # Ranking Table
    # -----------------------------
    subset = df_rankings[df_rankings["Locality"] == locality] \
        .sort_values("rank_in_locality")

    if not subset.empty:
        top = subset.iloc[0]
        st.success(
            f"🥇 **#{int(top['rank_in_locality'])}: {top['Restaurant Name']}** "
            f"({top['Cuisines']})"
        )

        st.dataframe(
            subset[
                ["rank_in_locality", "Restaurant Name",
                 "pred_score", "Aggregate rating",
                 "Votes", "Cuisines"]
            ],
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No ranking data available.")

    st.markdown("---")

    # -----------------------------
    # Feature Importance
    # -----------------------------
    st.subheader("📊 What Drives These Rankings?")

    with st.expander("Show Feature Importance", expanded=False):
        fi_df = pd.DataFrame({
            "Feature": ranker_features,
            "Importance": ranker_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.bar_chart(fi_df.head(10).set_index("Feature"))
        st.caption(
            "Higher importance indicates stronger influence on restaurant rank within a locality."
        )
