import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from streamlit_option_menu import option_menu

# config
st.set_page_config(
    page_title="Social Media Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ui theme
st.markdown("""
<style>
    /* IMPORT FONT - Inter */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* A. MAIN BACKGROUND */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Text Colors - Dark Slate for better readability than pure black */
    h1, h2, h3, h4, h5, h6, .stMarkdown, p, div, span {
        color: #1e293b !important; /* Slate 800 */
    }
    
    /* B. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Sidebar Text */
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stMarkdown {
        color: #0f172a !important; /* Slate 900 */
    }

    /* C. METRIC CARDS */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.875rem;
        font-weight: 500;
        color: #64748b; /* Slate 500 */
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.875rem;
        font-weight: 700;
        color: #0f172a; /* Slate 900 */
    }
    
    /* D. BUTTONS (Sidebar) */
    button[kind="secondary"] {
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        color: #334155 !important;
        border-radius: 8px;
        transition: all 0.2s;
    }
    button[kind="secondary"]:hover {
        border-color: #94a3b8;
        background-color: #f1f5f9;
        color: #0f172a !important;
    }
    button[kind="primary"] {
        background-color: #0f766e !important; /* Teal 700 */
        border: none;
        color: #ffffff !important;
        border-radius: 8px;
        box-shadow: 0 4px 6px -1px rgba(15, 118, 110, 0.3);
    }
    button[kind="primary"]:hover {
        background-color: #115e59 !important; /* Teal 800 */
        color: #ffffff !important;
    }
    
    /* F. CHARTS */
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        padding: 16px;
    }
    
    /* G. DATAFRAME */
    .stDataFrame {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }

</style>
""", unsafe_allow_html=True)


# --- 1. SIDEBAR NAVIGATION & SETTINGS ---
with st.sidebar:
    # Navigation Menu
    selected = option_menu(
        menu_title="Main Menu",
        options=["Dashboard", "Campaigns", "Platforms", "Geography", "Data Explorer", "AI Assistant"],
        icons=["house", "graph-up-arrow", "cast", "globe", "table", "robot"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#64748b", "font-size": "16px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#f1f5f9", "color": "#334155"},
            "nav-link-selected": {"background-color": "#0f766e", "color": "white", "font-weight": "500"},
        }
    )
    
    st.divider()

    # --- DATA SOURCE SETTINGS ---
    with st.expander("📂 Data Source Settings", expanded=False):
        st.caption("Manage your dataset")
        data_source = st.radio("Source", ["Default Dataset", "Upload CSV"], horizontal=True)
        
        if data_source == "Upload CSV":
            st.info("Download a template if needed.")
            # Mock Data Template
            sample_data = {
                'Platform': ['Instagram', 'TikTok', 'YouTube', 'Facebook'],
                'Region': ['USA', 'Brazil', 'UK', 'India'],
                'Post_Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
                'Content_Type': ['Reel', 'Video', 'Short', 'Post'],
                'Hashtag': ['#viral', '#trend', '#tech', '#news'],
                'Views': [1200, 5000, 3000, 1500],
                'Likes': [120, 500, 250, 100],
                'Comments': [15, 60, 30, 10],
                'Shares': [5, 50, 20, 5],
                'Ad_Spend': [50.0, 150.0, 80.0, 40.0],
                'Revenue_Generated': [250.0, 300.0, 200.0, 100.0]
            }
            sample_df = pd.DataFrame(sample_data)
            csv_sample = sample_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Template",
                data=csv_sample,
                file_name="social_media_sample_template.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            uploaded_file = st.file_uploader("Upload CSV", type=['csv'])

# --- 2. DATA LOADING LOGIC ---
# Helper function to load and clean data
@st.cache_data
def load_and_clean_data(file_input):
    try:
        # Read Data (handle both file path string and uploaded file object)
        if isinstance(file_input, str):
            df = pd.read_csv(file_input)
        else:
            df = pd.read_csv(file_input)
        
        # 1. Normalize Columns (strip spaces)
        df.columns = df.columns.str.strip()
        
        # 2. Ensure Date Format
        if 'Post_Date' in df.columns:
            df['Post_Date'] = pd.to_datetime(df['Post_Date'], errors='coerce')

        # 3. Ensure Numeric Metrics & Auto-Clean
        numeric_cols = ['Likes', 'Shares', 'Comments', 'Views']
        for col in numeric_cols:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        # --- DATA ENRICHMENT ---
        if 'Total_Interactions' not in df.columns:
            df['Total_Interactions'] = df['Likes'] + df['Shares'] + df['Comments']

        if 'Engagement_Rate' not in df.columns:
            df['Engagement_Rate'] = df.apply(
                lambda x: (x['Total_Interactions'] / x['Views'] * 100) if x['Views'] > 0 else 0, axis=1
            )

        if 'Ad_Spend' not in df.columns:
            df['Ad_Spend'] = (df['Views'] / 1000) * 5.00  
        
        if 'Revenue_Generated' not in df.columns:
            df['Revenue_Generated'] = (df['Total_Interactions'] * 0.50)

        if 'ROI' not in df.columns:
            df['ROI'] = df.apply(
                lambda x: ((x['Revenue_Generated'] - x['Ad_Spend']) / x['Ad_Spend'] * 100) if x['Ad_Spend'] > 0 else 0, axis=1
            )
        
        # 4. Handle Missing Categoricals
        if 'Region' not in df.columns: df['Region'] = 'Unknown'
        if 'Platform' not in df.columns: df['Platform'] = 'Unknown'
        if 'Content_Type' not in df.columns: df['Content_Type'] = 'Unknown'
        if 'Hashtag' not in df.columns: df['Hashtag'] = 'None'

        return df

    except Exception as e:
        return None

# Load Data based on selection
df = None
if data_source == "Upload CSV":
    if uploaded_file:
        df = load_and_clean_data(uploaded_file)
        if df is None:
            st.error("Error reading file.")
            st.stop()
    else:
        st.warning("Please upload a CSV file inside 'Data Source Settings' to proceed.")
        st.stop()
else:
    try:
        file_path = os.path.join(os.path.dirname(__file__), 'Cleaned_Viral_Social_Media_Trends.csv')
        df = load_and_clean_data(file_path)
    except FileNotFoundError:
        st.error("Default data file not found.")
        st.stop()

if df is None:
    st.stop()

# --- 3. FILTER SETTINGS (SIDEBAR) ---
with st.sidebar:
    with st.expander("⚙️ Filter Settings", expanded=False):
        st.caption("Apply filters to the dashboard")
        
        # Platform Filter
        selected_platform = st.multiselect(
            "Filter by Platform",
            options=df['Platform'].unique(),
            default=[] 
        )
        
        st.divider()
        st.subheader("Select Region")
        
        # Region Selection using st.pills (Streamlit 1.35+)
        unique_regions = sorted(list(df['Region'].unique()))
        
        # Default selection logic needs to be handled carefully with pills
        # If nothing is selected in pills, it usually implies "All" or "None". 
        # We'll treat None/Empty as "All Regions".
        
        selected_regions_pills = st.pills(
            "Regions",
            options=unique_regions,
            selection_mode="multi",
            default=None # Defaults to empty selection
        )
        
        # Map pills selection to logic
        # If selection is empty, we treat it as ALL.
        # But for 'st.pills' multi, distinct visual feedback is good.
        
        if not selected_regions_pills:
             st.caption("All Regions Selected")
             selected_regions = []
        else:
             selected_regions = selected_regions_pills


# --- 4. APPLY FILTER LOGIC ---
df_filtered = df.copy()

if selected_platform:
    df_filtered = df_filtered[df_filtered['Platform'].isin(selected_platform)]

if selected_regions:
    df_filtered = df_filtered[df_filtered['Region'].isin(selected_regions)]


# --- 5. CHART UTILS ---
# Update Plotly Commons
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#0f766e", "#3b82f6", "#f59e0b", "#ec4899", "#8b5cf6", "#10b981"]

def update_chart_layout(fig):
    fig.update_layout(
        font_family="Inter",
        font_color="#334155",
        title_font_size=18,
        title_font_family="Inter",
        title_font_color="#0f172a",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, showline=True, linecolor='#cbd5e1', tickfont=dict(color='#64748b')),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', showline=False, tickfont=dict(color='#64748b'))
    )
    return fig


# --- 6. RENDER PAGE CONTENT BASED ON SELECTION ---

st.title("Social Media Intelligence Dashboard")

# === DASHBOARD (Executive Overview) ===
if selected == "Dashboard":
    st.header("Executive Overview")
    st.markdown("High-level performance metrics and key performance indicators (KPIs).")
    
    total_views = df_filtered['Views'].sum()
    avg_engagement_rate = df_filtered['Engagement_Rate'].mean()
    total_spend = df_filtered['Ad_Spend'].sum()
    avg_roi = df_filtered['ROI'].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Impressions", f"{total_views:,.0f}")
    col2.metric("Avg Engagement Rate", f"{avg_engagement_rate:.2f}%")
    col3.metric("Total Ad Spend (Est.)", f"${total_spend:,.2f}")
    col4.metric("Average ROI", f"{avg_roi:.2f}%")
    
    st.markdown("---")
    
    st.subheader("Performance Trends Over Time")
    if 'Post_Date' in df_filtered.columns:
        daily_trends = df_filtered.groupby('Post_Date')[['Views', 'Total_Interactions']].sum().reset_index()
        fig_line = px.line(daily_trends, x='Post_Date', y=['Views', 'Total_Interactions'], 
                           markers=True, 
                           title="Impressions vs. Interactions Timeline")
        fig_line = update_chart_layout(fig_line)
        fig_line.update_xaxes(showgrid=False)
        fig_line.update_yaxes(showgrid=True, gridcolor='#ECEFF1')
        st.plotly_chart(fig_line, use_container_width=True)

# === CAMPAIGNS ===
elif selected == "Campaigns":
    st.header("Campaign Effectiveness & ROI")
    st.markdown("Analysis of campaign performance and return on investment.")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("ROI by Content Category")
        roi_by_cat = df_filtered.groupby('Content_Type')[['ROI', 'Ad_Spend']].mean().reset_index()
        fig_bar_roi = px.bar(roi_by_cat, x='Content_Type', y='ROI', 
                             color='Content_Type', 
                             title="Average Return on Investment (ROI) by Category")
        fig_bar_roi = update_chart_layout(fig_bar_roi)
        st.plotly_chart(fig_bar_roi, use_container_width=True)

    with c2:
        st.subheader("Cost Efficiency Analysis")
        # Handle empty df case for quantiles
        if not df_filtered.empty:
            upper_limit = df_filtered['ROI'].quantile(0.95)
            lower_limit = df_filtered['ROI'].quantile(0.05)
            df_chart = df_filtered[(df_filtered['ROI'] < upper_limit) & (df_filtered['ROI'] > lower_limit)]
        else:
            df_chart = df_filtered

        fig_bubble = px.scatter(df_chart, x='Ad_Spend', y='ROI', size='Views', color='Content_Type',
                                hover_name='Hashtag', title=f"Ad Spend vs. ROI (Outliers Removed)", opacity=0.8)
        fig_bubble.add_hline(y=0, line_dash="dash", line_color="#B0BEC5")
        fig_bubble = update_chart_layout(fig_bubble)
        st.plotly_chart(fig_bubble, use_container_width=True)
        st.caption(f"Note: Extreme outliers (Top/Bottom 5%) removed.")

    st.subheader("Top Performing Hashtags (Campaigns)")
    top_campaigns = df_filtered.groupby('Hashtag')[['Views', 'Engagement_Rate', 'ROI']].mean().reset_index()
    top_campaigns = top_campaigns.sort_values(by='ROI', ascending=False).head(10)
    st.dataframe(top_campaigns.style.format({"Views": "{:,.0f}", "Engagement_Rate": "{:.2f}", "ROI": "{:.2f}"}), use_container_width=True)

# === PLATFORMS ===
elif selected == "Platforms":
    st.header("Platform Analytics")
    st.markdown("Comparative analysis of channel performance.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Share of Voice")
        fig_pie = px.pie(df_filtered, names='Platform', values='Views', hole=0.5, title="Distribution of Views by Platform")
        fig_pie = update_chart_layout(fig_pie)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("Engagement Quality")
        fig_scatter = px.scatter(df_filtered, x='Views', y='Total_Interactions', color='Platform', title="Views vs. Total Interactions")
        fig_scatter = update_chart_layout(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

# === GEOGRAPHY ===
elif selected == "Geography":
    st.header("Geographic Distribution")
    st.subheader("Regional Performance Hierarchy")
    fig_tree = px.treemap(df_filtered, path=['Region', 'Platform'], values='Views',
                          color='Engagement_Rate', color_continuous_scale='Viridis',
                          title="Views by Region (Color = Engagement Rate)")
    fig_tree = update_chart_layout(fig_tree)
    fig_tree.update_layout(height=600)
    fig_tree.update_traces(textfont=dict(color='#ffffff', size=14), marker=dict(line=dict(color='#FFFFFF', width=1)))
    st.plotly_chart(fig_tree, use_container_width=True)

# === DATA EXPLORER ===
elif selected == "Data Explorer":
    st.header("Dataset Explorer")
    st.markdown("Detailed view of the filtered dataset.")
    st.dataframe(df_filtered, use_container_width=True)
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download Filtered CSV", data=csv, file_name='filtered_social_media_data.csv', mime='text/csv')

# === AI ASSISTANT ===
elif selected == "AI Assistant":
    st.header("AI Data Assistant")
    st.markdown("Ask questions about your social media data and get instant insights.")

    # --- SETUP API KEY ---
    try:
        api_key = st.secrets.get("GOOGLE_API_KEY")
    except FileNotFoundError:
        api_key = None
    
    if not api_key:
        api_key = st.sidebar.text_input("Google API Key", type="password", help="Enter your Gemini API Key")
    
    if not api_key:
        st.warning("Please enter your Google API Key in the sidebar to use the AI Assistant.")
    else:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        st.markdown("#### Suggested Questions")
        sq_cols = st.columns(3)
        suggestions = ["Which region has the highest ROI?", "Compare engagement between Instagram and TikTok", "What content type drives the most revenue?"]
        
        # Helper to set question (can be removed if not used elsewhere, but keeping logic local)
        # We will check buttons directly.

        buttons_prompt = None
        for i, question in enumerate(suggestions):
            if sq_cols[i % 3].button(question, use_container_width=True):
                 buttons_prompt = question

        # Always show the chat input
        user_prompt = st.chat_input("Ask a question about your data...")

        # Determine prompt
        prompt = buttons_prompt if buttons_prompt else user_prompt

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate Summary Context
            def summarize_dataframe_detailed(df):
                info = []
                info.append(f"### Dataset High-Level Overview")
                info.append(f"Total Records: {len(df)}")
                info.append(f"Columns: {', '.join(df.columns)}")
                
                if 'Ad_Spend' in df.columns and 'Revenue_Generated' in df.columns:
                    total_spend = df['Ad_Spend'].sum()
                    total_rev = df['Revenue_Generated'].sum()
                    global_roi = ((total_rev - total_spend) / total_spend) * 100 if total_spend > 0 else 0
                    info.append(f"Total Ad Spend: ${total_spend:,.2f}")
                    info.append(f"Total Revenue: ${total_rev:,.2f}")
                    info.append(f"Global ROI: {global_roi:.2f}%")

                def add_groupby_summary(name, group_col, metric_cols=['ROI', 'Engagement_Rate', 'Views', 'Revenue_Generated']):
                    if group_col in df.columns:
                        info.append(f"\\n### Aggregation by {name}")
                        try:
                            # Calculate Group Means
                            grp_mean = df.groupby(group_col)[metric_cols].mean()
                            # Calculate Group Counts
                            grp_count = df.groupby(group_col).size().rename('Count')
                            
                            # Combine
                            grp = pd.concat([grp_mean, grp_count], axis=1).reset_index()
                            grp = grp.sort_values(by='Count', ascending=False)
                            
                            info.append(grp.to_markdown(index=False, floatfmt=".2f"))
                        except Exception as e:
                            pass

                add_groupby_summary("Region", "Region")
                add_groupby_summary("Platform", "Platform")
                add_groupby_summary("Content Type", "Content_Type")

                # 2. Raw Data Sample (New)
                # Take up to 50 rows for context to allow the AI to see the raw structure
                sample_size = min(50, len(df))
                if sample_size > 0:
                    # Use random sample
                    msg_sample = df.sample(n=sample_size).to_csv(index=False)
                    info.append(f"\\n### Raw Data Sample ({sample_size} random rows)")
                    info.append("The following is a sample of the raw data to understand the structure and values. Use this to infer patterns or answers for specific examples:")
                    info.append(f"```csv\\n{msg_sample}\\n```")
                
                return "\\n".join(info)

            data_context = summarize_dataframe_detailed(df_filtered)
            system_prompt = f"""
            You are an expert data analyst assistant. 
            You are analyzing a social media dataset.
            Here is a detailed summary of the current filtered data, including aggregations:
            
            {data_context}
            
            Answer the user's question based on this data. 
            CRITICAL INSTRUCTIONS:
            1. Answer DIRECTLY. Do not start with "Based on the data..." or "Here is the analysis...".
            2. Be EXTREMELY CONCISE. Use bullet points for lists.
            3. Prioritize numbers, metrics, and facts.
            4. Do not offer unsolicited advice or "hope this helps" messages.
            5. If the answer is a single number or name, just provide that.
            """

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                try:
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    model_name = "models/gemini-1.5-flash"
                    if model_name not in available_models:
                         model_name = available_models[0] if available_models else "gemini-pro"

                    model = genai.GenerativeModel(model_name)
                    full_prompt = system_prompt + "\n\nUser Question: " + prompt
                    response = model.generate_content(full_prompt)
                    
                    full_response = response.text
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    error_msg = str(e)
                    if "429" in error_msg or "quota" in error_msg.lower():
                        st.error("⚠️ **Traffic Limit Reached**\n\nThe free usage quota for the AI Assistant has been exceeded safely. Please wait approx 60 seconds and try again!")
                    else:
                        st.error(f"An error occurred: {error_msg}")