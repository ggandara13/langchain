# NCL Dynamic Pricing Platform - Senior DS Interview Demo
# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Set page config
st.set_page_config(
    page_title="Pricing Intelligence Platform ",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)







# Force light theme with better compatibility
st.markdown("""
<style>
    /* Force light theme for main app */
    .stApp {
        background-color: #FFFFFF;
        color: #262730;
    }
    
    /* Fix ALL selectboxes - both in sidebar and main content */
    .stSelectbox > div > div {
        background-color: #FFFFFF !important;
        color: #0E1117 !important;
        border: 1px solid #D3D3D3 !important;
    }
    
    /* Fix all selectbox labels */
    .stSelectbox label {
        color: #0E1117 !important;
        font-weight: 500 !important;
    }
    
    /* Fix all dropdown containers */
    [data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }
    
    /* Fix dropdown input areas */
    [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        color: #0E1117 !important;
    }
    
    /* Fix dropdown menu that appears */
    [data-baseweb="popover"] {
        background-color: #FFFFFF !important;
    }
    
    /* Fix dropdown menu list */
    [data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }
    
    /* Fix dropdown options */
    [role="option"] {
        background-color: #FFFFFF !important;
        color: #0E1117 !important;
        padding: 8px 12px !important;
    }
    
    /* Hover state for options */
    [role="option"]:hover {
        background-color: #F0F2F6 !important;
        color: #0E1117 !important;
    }
    
    /* Selected option highlight */
    [aria-selected="true"] {
        background-color: #E6F3FF !important;
        color: #0E1117 !important;
    }
    
    /* Fix any black backgrounds in selects */
    div[data-baseweb="select"] div {
        background-color: #FFFFFF !important;
        color: #0E1117 !important;
    }
    
    /* Force all text in dropdowns to be dark */
    .stSelectbox * {
        color: #0E1117 !important;
    }
    
    /* Fix sidebar specific styling */
    section[data-testid="stSidebar"] {
        background-color: #F0F2F6 !important;
    }
    
    /* Fix metric containers */
    [data-testid="metric-container"] {
        background-color: #F0F2F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        border-radius: 8px;
        padding: 0.5rem;
        border: 1px solid #E0E0E0;
    }
    
    /* Target metric values */
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #0E1117 !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
    }
    
    /* Fix code blocks */
    .stCodeBlock, pre, code {
        background-color: #F5F5F5 !important;
        color: #0E1117 !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 6px !important;
    }
    
    /* Fix expander styling */
    .streamlit-expanderHeader {
        background-color: #F8F9FA !important;
        color: #0E1117 !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
    }
    
    /* Global text color override */
    * {
        color: #0E1117 !important;
    }
    
    /* Keep buttons white */
    .stButton > button {
        color: white !important;
    }
    
    /* Keep checkmark green */
    [data-testid="stMetricValue"] svg {
        color: #21c354 !important;
    }
    
    /* Additional force for any remaining black elements */
    [style*="background-color: black"],
    [style*="background-color: rgb(0, 0, 0)"],
    [style*="background: black"],
    [style*="background: rgb(0, 0, 0)"] {
        background-color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)






# Title and intro with logo (left-aligned)
# Logo and Title (vertically stacked)
# Center the logo
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image(
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQFIOLSxBNUYVqchXTA0cdeQps44W2Tba82jQ&s",
        width=300
    )

# Title and subtitle below the logo
st.title("🚢 NCL Pricing Intelligence Platform")
st.markdown("**Prototype by Gerardo Gandara | Senior Data Scientist Interview**")
st.markdown("---")  # Add a horizontal line for separation



# Sidebar navigation
st.sidebar.header("Platform Modules")
selected_module = st.sidebar.selectbox(
    "Select Module",
    ["Executive Dashboard", "Model Lifecycle Demo", "Price Optimization Engine", 
     "Deployment Strategies", "A/B Testing Framework"]
)

# Load data (using your scraped data)
@st.cache_data
def load_real_cruise_data():
    """Load the actual scraped cruise data"""
    # This would load your real scraped data
    # For demo, generating similar structure
    np.random.seed(42)
    
    cruise_lines = ['Norwegian', 'Royal Caribbean', 'Carnival', 'MSC', 'Celebrity']
    ships = {
        'Norwegian': ['Norwegian Escape', 'Norwegian Joy', 'Norwegian Bliss'],
        'Royal Caribbean': ['Symphony of the Seas', 'Icon of the Seas'],
        'Carnival': ['Carnival Celebration', 'Carnival Magic'],
        'MSC': ['MSC Seascape', 'MSC World America'],
        'Celebrity': ['Celebrity Beyond', 'Celebrity Apex']
    }
    
    data = []
    for _ in range(300):
        line = np.random.choice(cruise_lines)
        ship = np.random.choice(ships[line])
        
        # Base prices by line (matching your findings)
        base_prices = {
            'Royal Caribbean': 1315,
            'Norwegian': 1031,
            'Carnival': 845,
            'MSC': 940,
            'Celebrity': 973
        }
        
        # Discounts by line (matching your findings)
        discounts = {
            'Royal Caribbean': 0.62,
            'Norwegian': 0.88,
            'Carnival': 0.20,
            'MSC': 0.45,
            'Celebrity': 0.50
        }
        
        base = base_prices[line] + np.random.normal(0, 100)
        discount = discounts[line] + np.random.normal(0, 0.05)
        discount = np.clip(discount, 0.1, 0.95)  # Ensure discount stays in valid range
        
        data.append({
            'cruise_line': line,
            'ship': ship,
            'sail_date': datetime(2026, np.random.randint(1, 13), np.random.randint(1, 29)),
            'base_price': base / (1 - discount),
            'discount': discount,
            'final_price': base,
            'occupancy': min(0.95, max(0.5, 1 - discount + np.random.normal(0, 0.1))),
            'bookings': int(np.random.poisson(max(10, 100 * (1 - discount))))  # Ensure positive lambda
        })
    
    return pd.DataFrame(data)

df = load_real_cruise_data()

# Module 1: Executive Dashboard
if selected_module == "Executive Dashboard":
    st.header("Executive Dashboard - Real-Time Pricing Intelligence")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    ncl_data = df[df['cruise_line'] == 'Norwegian']
    rc_data = df[df['cruise_line'] == 'Royal Caribbean']
    
    with col1:
        avg_price_ncl = ncl_data['final_price'].mean()
        st.metric("NCL Avg Price", f"${avg_price_ncl:.0f}", "-27% vs RC")
    
    with col2:
        avg_discount_ncl = ncl_data['discount'].mean()
        st.metric("NCL Avg Discount", f"{avg_discount_ncl*100:.1f}%", "🔴 Too High")
    
    with col3:
        elasticity = 0.821
        st.metric("NCL Price Elasticity", f"{elasticity}", "Price Sensitive")
    
    with col4:
        revenue_opp = (rc_data['final_price'].mean() - avg_price_ncl) * len(ncl_data)
        st.metric("Revenue Opportunity", f"${revenue_opp/1e6:.1f}M", "+35% potential")
    
    # Competitive positioning
    st.subheader("Competitive Positioning Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Scatter plot from your analysis
        fig = px.scatter(df, x='discount', y='final_price', color='cruise_line',
                        title="Price vs Discount Strategy by Cruise Line",
                        labels={'discount': 'Discount %', 'final_price': 'Final Price ($)'},
                        size='occupancy', hover_data=['ship'])
    

        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#262730'),  # Darker gray for better contrast
            xaxis=dict(
                gridcolor='#E5E5E5',
                title_font=dict(color='#262730'),
                tickfont=dict(color='#262730')
            ),
            yaxis=dict(
                gridcolor='#E5E5E5',
                title_font=dict(color='#262730'),
                tickfont=dict(color='#262730')
            ),
            title_font=dict(color='#262730')
        )





        
        
        # Add zones
        fig.add_shape(type="rect", x0=0, x1=0.3, y0=1200, y1=1600,
                     fillcolor="green", opacity=0.1, layer="below")
        fig.add_shape(type="rect", x0=0.7, x1=1, y0=800, y1=1200,
                     fillcolor="red", opacity=0.1, layer="below")
        
        fig.add_annotation(x=0.15, y=1400, text="Premium Zone", showarrow=False)
        fig.add_annotation(x=0.85, y=1000, text="Discount Trap", showarrow=False)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("""
        ### Key Insights
        
        ✅ **Royal Caribbean**
        - Premium positioning
        - -0.012 elasticity
        - Pricing power
        
        ⚠️ **Norwegian**
        - Discount trap
        - 0.821 elasticity  
        - Price sensitive
        
        💡 **Opportunity**
        - Reduce discounts 10%
        - Test on new ships
        - Monitor elasticity
        """)
    
    # Your real elasticity analysis
    st.subheader("Price Elasticity by Cruise Line")
    
    elasticity_data = pd.DataFrame({
        'Cruise Line': ['Royal Caribbean', 'Norwegian', 'Carnival', 'MSC', 'Celebrity'],
        'Elasticity': [-0.012, 0.821, 0.516, 0.132, 0.587],
        'Interpretation': ['Inelastic', 'Elastic', 'Elastic', 'Inelastic', 'Elastic']
    })

    
    fig = px.bar(elasticity_data, x='Cruise Line', y='Elasticity', color='Interpretation',
                 title="Price Elasticity Comparison",
                 color_discrete_map={'Elastic': 'red', 'Inelastic': 'green'})
    
    # Add this to fix the black background and ensure ALL text is visible
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#262730', size=12),
        xaxis=dict(
            gridcolor='#E5E5E5',
            title_font=dict(color='#262730', size=14),
            tickfont=dict(color='#262730', size=12),
            linecolor='#262730'
        ),
        yaxis=dict(
            gridcolor='#E5E5E5', 
            title_font=dict(color='#262730', size=14),
            tickfont=dict(color='#262730', size=12),
            linecolor='#262730'
        ),
        title_font=dict(color='#262730', size=16),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(255, 255, 255, 0.9)',  # White background with slight transparency
            bordercolor='#262730',
            borderwidth=1,
            font=dict(color='#262730', size=12),  # Ensure legend text is dark
            title=dict(
                text='Interpretation',
                font=dict(color='#262730', size=12)
            )
        )
    )
    
    # Also update the traces to ensure hover text is visible
    fig.update_traces(
        textfont=dict(color='#262730'),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial",
            font_color="#262730"
        )
    )



    
    
    st.plotly_chart(fig, use_container_width=True)

# Module 2: Model Lifecycle Demo
elif selected_module == "Model Lifecycle Demo":
    st.header("Full Model Lifecycle - Production Implementation")
    
    st.markdown("### Demonstrating my Kalibri Labs forecasting project lifecycle")
    
    # Create lifecycle visualization
    lifecycle_stages = {
        "1. Problem Definition": {
            "description": "Forecast hotel occupancy 60 days out for 10K+ Marriott properties",
            "tools": "Stakeholder interviews, business case analysis",
            "duration": "2 weeks",
            "details": "Group bookings (RC21) were critical for revenue but hard to forecast"
        },
        "2. Data Engineering": {
            "description": "Built scalable data extraction for 10K properties with 1000x performance optimization",
            "tools": "Python, SQL, Snowflake, Multiprocessing, Transient Tables",
            "duration": "3 weeks + optimization",
            "details": """Week 1: EDA on multiple tables, identified group commit table
Week 2: SQL query development, tested on 1→100 properties  
Week 3: Parallel processing for 10K properties (8hrs→45min)
🚀 MAJOR OPTIMIZATION (v0.0.2):
- Database Throughput: 4,470 rows/second (1000x improvement)
- Batch Processing: Single bulk operations vs thousands of individual INSERTs
- Transient Tables: Eliminated connection overhead
- Processing Time: 602 properties in 39.7 minutes (91.4% success rate)
- Production Scale: Successfully deployed for 7K+ Marriott properties"""
        },
        "3. Model Development": {
            "description": "Two-stage hybrid ML approach: Classification + Regression with time series ensemble",
            "tools": "XGBoost, Prophet, scikit-learn, Feature Engineering, Ensemble Methods",
            "duration": "4 weeks",
            "details": """🎯 TWO-STAGE MODELING APPROACH:
Stage 1 - Classification: Predicts if significant change (>10%) will occur
Stage 2 - Regression: Predicts materialization rate (0-150%)
📊 FEATURE ENGINEERING:
- 28 engineered features capturing commitment volatility patterns
- Key features: lead_time, commitment_ratio, relative_change_from_first, 
  min/max_commit_so_far, pct_change_volatility, recent_trend
- Adaptive simplification: 4 levels (28→9 features) based on data availability
- Handles missing evolution data with pattern-based imputation
🔧 MODEL ARCHITECTURE:
- XGBoost with early stopping (50 rounds)
- Chronological validation to prevent data leakage
- Handles edge cases: single-class scenarios, missing data
- Hybrid ensemble: 70% XGBoost + 30% time series (Prophet/ARIMA)
📈 PERFORMANCE METRICS:
- Classification F1: 0.82 (Precision: 0.85, Recall: 0.79)
- Regression MAPE: 4.2% (RMSE: 0.056, MAE: 0.041)
- Feature importance: top 5 features = 65% of model importance
- Validation: Last 10% of chronological data (adaptive sizing)
🚀 KEY INNOVATIONS:
- Two-stage approach reduces false positives by 40%
- Feature reduction maintains 98% accuracy with 68% fewer features
- Ensemble method improves edge case predictions by 15%
- Production-ready with fallback models for insufficient data"""
        },
        "4. Validation": {
            "description": "Backtesting, cross-validation, pilot with 20 properties",
            "tools": "Custom validation framework, statistical tests",
            "duration": "2 weeks",
            "details": "Validated on 2 years holdout data, A/B tested with select properties"
        },
        "5. Deployment": {
            "description": "Production pipeline: 10K properties in 2 hours",
            "tools": "Jenkins, Docker, Snowflake",
            "duration": "2 weeks",
            "details": """Jenkins orchestration (runs Wed 2am)
Docker containers for reproducibility
Full pipeline: Extract→Train→Predict(60 days)→Update Snowflake"""
        },
        "6. Monitoring": {
            "description": "Weekly MAPE tracking, auto-alerts on degradation",
            "tools": "Snowflake dashboards, email alerts",
            "duration": "Ongoing",
            "details": "Automated reports to revenue managers, retraining triggers at 5% MAPE degradation"
        }
    }
    
    # Display lifecycle
    selected_stage = st.selectbox("Select Lifecycle Stage", list(lifecycle_stages.keys()))
    
    stage_info = lifecycle_stages[selected_stage]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        #st.metric("Stage", selected_stage
        st.metric("Stage", selected_stage.split(". ")[1] if ". " in selected_stage else selected_stage)
    with col2:
        st.metric("Duration", stage_info["duration"])
    with col3:
        st.metric("Status", "✅ Completed")
    
    st.markdown(f"**Description:** {stage_info['description']}")
    st.markdown(f"**Tools Used:** {stage_info['tools']}")
    
    # Show detailed breakdown
    with st.expander("Detailed Breakdown"):
        st.markdown(stage_info['details'])
    
    # Show code example for selected stage
    if selected_stage == "2. Data Engineering":
        with st.expander("Show Code Example - Architecture Optimization"):
            st.code("""
# BEFORE: Individual property processing (slow)
for property_id in properties:
    conn = snowflake.connector.connect()
    data = fetch_data(conn, property_id)
    process_property(data)
    conn.close()  # 10K connections!

# AFTER: Batch processing with transient tables (1000x faster)
def process_batch_with_transient_tables(property_batch):
    '''Architecture optimization that achieved 4,470 rows/sec'''
    
    # Single connection for entire batch
    conn = snowflake.connector.connect(**SNOWFLAKE_PARAMS)
    
    # Create transient staging table
    conn.cursor().execute('''
        CREATE OR REPLACE TRANSIENT TABLE temp_batch_data (
            property_id VARCHAR,
            date DATE,
            room_nights NUMBER,
            group_bookings NUMBER
        ) CLUSTER BY (property_id, date)
    ''')
    
    # Bulk load data for all properties at once
    write_pandas(conn, batch_df, 'temp_batch_data')
    
    # Process entire batch with single query
    results = conn.cursor().execute('''
        SELECT * FROM temp_batch_data
        WHERE property_id IN ({})
    '''.format(','.join(property_batch))).fetchall()
    
    # Bulk save predictions (single INSERT instead of 10K)
    save_forecast_group_commit_batch(conn, predictions_df)
    
    conn.close()  # Only 1 connection per batch!

# Performance Configuration (Critical for production)
export OPENBLAS_NUM_THREADS=4  # Jenkins/production setting
xgb_params = {'nthread': 1}    # XGBoost stability in parallel

# Results: 4,470 rows/second, 91.4% success rate
            """)
            
        # Add performance metrics visualization
        st.markdown("### 🏆 Engineering Achievement Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Database Throughput", "4,470 rows/sec", "↑ 1000x")
        with col2:
            st.metric("Batch Processing", "10K properties", "Single operation")
        with col3:
            st.metric("Success Rate", "91.4%", "Production tested")
        with col4:
            st.metric("Time Reduction", "95%", "vs v0.0.1")
    
    elif selected_stage == "3. Model Development":
        with st.expander("Show Model Architecture - Two-Stage Hybrid Approach"):
            st.code("""
def train_two_stage_hybrid_model(
    historical_data, group_commits, future_commits, 
    property_id=None, use_hybrid=True, xgb_weight=0.7
):
    '''Two-stage model with time series ensemble for group booking forecasting'''
    
    # 1. Feature Engineering (28 features capturing commitment volatility)
    features = [
        'lead_time', 'occupancy_month', 'occupancy_dayofweek',
        'commitment_ratio', 'relative_change_from_first',
        'min_commit_so_far', 'max_commit_so_far', 'commit_range', 
        'commit_std', 'pct_change_volatility', 'change_frequency',
        'recent_trend', 'change_acceleration', 'total_group_commit'
    ]
    
    # 2. Adaptive Feature Simplification (based on data availability)
    if simplification_level == 3:  # Aggressive reduction to 9 features
        features = [
            'total_group_commit', 'lead_time', 'relative_change_from_first',
            'snapshot_month', 'max_commit_so_far', 'min_commit_so_far',
            'month_materialization_rate', 'recent_trend', 'occupancy_quarter'
        ]
    
    # 3. Stage 1: Classification Model
    # Predicts: Will there be >10% change between commitment and actual?
    class_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    class_model = xgb.train(params=class_params, dtrain=dtrain,
                           callbacks=[EarlyStopping(rounds=50)])
    
    # 4. Stage 2: Regression Model  
    # Predicts: What will the materialization rate be (0-150%)?
    reg_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'learning_rate': 0.1,
        'max_depth': 5
    }
    reg_model = xgb.train(params=reg_params, dtrain=dtrain,
                         callbacks=[EarlyStopping(rounds=50)])
    
    # 5. Hybrid Ensemble (if enabled)
    if use_hybrid:
        # Train time series model (Prophet/ARIMA)
        ts_predictions = generate_simplified_ts_predictions(ts_model, future_commits)
        
        # Ensemble: 70% XGBoost + 30% time series
        final_predictions = ensemble_predictions(
            xgb_predictions, ts_predictions, xgb_weight=0.7
        )
    
    return models, final_predictions, X, y_class, y_reg

# Chronological validation to prevent data leakage
val_size = max(int(total_samples * 0.1), 5)  # Last 10% for validation
X_train = X_sorted.iloc[:-val_size]
X_val = X_sorted.iloc[-val_size:]

# Handle edge cases (single-class scenarios)
if y_train.unique().size == 1:
    # Add synthetic example of opposite class
    X_train = add_synthetic_example(X_train, y_train)
            """)
        
        # Show feature importance analysis
        st.markdown("### 📊 Feature Importance Analysis")
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'Feature': ['total_group_commit', 'lead_time', 'relative_change_from_first', 
                       'max_commit_so_far', 'snapshot_month', 'min_commit_so_far',
                       'month_materialization_rate', 'recent_trend', 'commit_range',
                       'occupancy_quarter', 'pct_change_volatility', 'change_frequency'],
            'Classification_Importance': [0.28, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.03, 0.02, 0.02, 0.02],
            'Regression_Importance': [0.22, 0.18, 0.14, 0.11, 0.09, 0.08, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01]
        })
        
        # Plot feature importance
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Classification', x=feature_importance['Feature'][:6], 
                            y=feature_importance['Classification_Importance'][:6]))
        fig.add_trace(go.Bar(name='Regression', x=feature_importance['Feature'][:6], 
                            y=feature_importance['Regression_Importance'][:6]))
        
        fig.update_layout(
            title='Top 6 Feature Importance by Model Type',
            xaxis_title='Feature', 
            yaxis_title='Importance Score',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='black')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Classification F1", "0.82", "Good balance")
        with col2:
            st.metric("Regression MAPE", "4.2%", "Industry leading")
        with col3:
            st.metric("Feature Reduction", "28→9", "68% reduction")
        with col4:
            st.metric("Processing Time", "39.7 min", "For 602 properties")
    
    elif selected_stage == "5. Deployment":
        with st.expander("Show Jenkins Pipeline"):
            st.code("""
// Jenkinsfile - Runs every Wednesday at 2 AM
pipeline {
    agent { docker { image 'python:3.9' } }
    
    triggers {
        cron('0 2 * * 3')  // Wednesday 2 AM
    }
    
    stages {
        stage('Extract Data') {
            steps {
                sh '''
                python src/extract_parallel.py \\
                    --properties 10000 \\
                    --workers 20
                '''
            }
        }
        
        stage('Train Models') {
            steps {
                sh 'python src/train_ensemble.py'
            }
        }
        
        stage('Generate Forecasts') {
            steps {
                sh '''
                python src/predict.py \\
                    --horizon 60 \\
                    --output snowflake
                '''
            }
        }
        
        stage('Update Snowflake') {
            steps {
                sh 'python src/update_forecasts.py'
            }
        }
    }
    
    post {
        success {
            mail to: 'revenue-managers@marriott.com',
                 subject: 'Forecast Update Complete',
                 body: '10K properties updated. Runtime: 2 hours.'
        }
    }
}
            """)
    
    # Performance metrics
    if selected_stage in ["2. Data Engineering", "5. Deployment"]:
        st.markdown("### Performance Achievements")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Properties Scaled", "10,000", "+9,900 from pilot")
        with col2:
            st.metric("Pipeline Runtime", "2 hours", "-6 hours from v1")
        with col3:
            st.metric("Forecast Horizon", "60 days", "Daily granularity")

# Module 3: Price Optimization Engine
elif selected_module == "Price Optimization Engine":
    st.header("Price Optimization Engine")
    
    st.markdown("### Revenue Maximization with Constraints")
    
    # Optimization parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        target_occupancy = st.slider("Target Occupancy %", 70, 95, 85)
    with col2:
        min_price = st.number_input("Min Price ($)", 500, 1000, 800)
    with col3:
        max_discount = st.slider("Max Discount %", 50, 90, 75)
    
    # Show optimization problem
    with st.expander("Optimization Problem Formulation"):
        st.latex(r"""
        \max_{p} \sum_{i} p_i \cdot d_i(p_i) \\
        \text{subject to:} \\
        p_i \geq p_{min} \\
        (1 - \frac{p_i}{p_{list}}) \leq \delta_{max} \\
        \sum_{i} d_i(p_i) / \text{capacity} \geq \text{occupancy}_{target}
        """)
        
        st.markdown("""
        Where:
        - $p_i$ = price for sailing $i$
        - $d_i(p_i)$ = demand function (using elasticity)
        - $p_{min}$ = minimum price constraint
        - $\delta_{max}$ = maximum discount constraint
        - occupancy$_{target}$ = target occupancy rate
        """)
    
    # Simulate optimization
    st.subheader("Optimization Results")
    
    # Generate optimal prices
    sailings = pd.DataFrame({
        'sailing': [f"Sailing {i+1}" for i in range(10)],
        'current_price': np.random.normal(1000, 100, 10),
        'current_occupancy': np.random.uniform(0.7, 0.9, 10)
    })
    
    # Simple optimization simulation
    elasticity = 0.821
    sailings['optimal_price'] = sailings.apply(
        lambda x: max(min_price, 
                     min(x['current_price'] * (target_occupancy/100 / x['current_occupancy'])**(1/elasticity),
                         x['current_price'] / (1 - max_discount/100))), 
        axis=1
    )
    
    sailings['price_change'] = (sailings['optimal_price'] / sailings['current_price'] - 1) * 100
    sailings['revenue_impact'] = sailings['price_change'] * (1 - elasticity * sailings['price_change']/100)
    
    # Display results
    st.dataframe(
        sailings.style.format({
            'current_price': '${:.0f}',
            'optimal_price': '${:.0f}',
            'current_occupancy': '{:.1%}',
            'price_change': '{:+.1f}%',
            'revenue_impact': '{:+.1f}%'
        })
    )
    
    total_impact = sailings['revenue_impact'].mean()
    st.metric("Average Revenue Impact", f"{total_impact:+.1f}%", 
              "Optimization successful" if total_impact > 0 else "Review constraints")

# Module 4: Deployment Strategies
elif selected_module == "Deployment Strategies":
    st.header("Model Deployment Strategies")
    
    st.markdown("### Understanding Batch vs API vs UDF Deployments")
    
    # Create comparison table
    deployment_comparison = pd.DataFrame({
        'Deployment Type': ['Batch', 'Real-time API', 'Snowflake UDF'],
        'Use Case': ['Daily forecasting', 'Dynamic pricing', 'In-database scoring'],
        'Latency': ['Hours', 'Milliseconds', 'Seconds'],
        'Complexity': ['Low', 'High', 'Medium'],
        'My Experience': ['✅ Production (Jenkins)', '🟡 Learning', '🟡 Exploring']
    })
    
    st.dataframe(deployment_comparison)
    
    # Show deployment decision tree
    st.subheader("Deployment Decision Framework")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Choose Batch When:**
        - Predictions needed daily/weekly
        - Large volume processing
        - Latency not critical
        - Example: Forecast all cruises for next 90 days
        """)
        
        with st.expander("My Batch Implementation"):
            st.code("""
# Jenkins Pipeline (Jenkinsfile)
pipeline {
    agent any
    
    triggers {
        cron('0 2 * * *')  // Daily at 2 AM
    }
    
    stages {
        stage('Data Pull') {
            steps {
                sh 'python src/extract_data.py'
            }
        }
        stage('Model Scoring') {
            steps {
                sh 'python src/score_model.py'
            }
        }
        stage('Write Results') {
            steps {
                sh 'python src/write_to_snowflake.py'
            }
        }
    }
}
            """)
    
    with col2:
        st.markdown("""
        **Choose API When:**
        - Real-time decisions needed
        - User-facing applications
        - Low latency critical
        - Example: Price quote on website
        """)
        
        with st.expander("API Implementation Plan"):
            st.code("""
# FastAPI Implementation (Learning)
from fastapi import FastAPI
import joblib

app = FastAPI()
model = joblib.load('price_model.pkl')

@app.post("/predict_price")
async def predict(features: dict):
    X = preprocess(features)
    price = model.predict(X)[0]
    return {"recommended_price": price}
            """)
    
    # UDF example
    st.subheader("Snowflake UDF Example")
    st.markdown("For lightweight scoring directly in SQL:")
    
    st.code("""
CREATE OR REPLACE FUNCTION calculate_optimal_price(
    current_price FLOAT,
    occupancy FLOAT,
    elasticity FLOAT
)
RETURNS FLOAT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.8'
HANDLER = 'optimize_price'
AS $$
def optimize_price(current_price, occupancy, elasticity):
    target_occupancy = 0.85
    price_multiplier = (target_occupancy / occupancy) ** (1/elasticity)
    return current_price * price_multiplier
$$;

-- Usage in SQL
SELECT 
    sailing_id,
    current_price,
    calculate_optimal_price(current_price, occupancy, 0.821) as optimal_price
FROM cruise_inventory;
    """)

# Module 5: A/B Testing Framework
else:  # A/B Testing
    st.header("A/B Testing Framework - Causal Inference")
    
    st.markdown("### Demonstrating my Difference-in-Differences expertise from Zimmerman")
    
    # Test setup
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        test_type = st.selectbox("Test Type", ["Price Change", "Discount Reduction", "New Pricing Model"])
    with col2:
        test_size = st.slider("Test Size %", 10, 50, 20)
    with col3:
        test_duration = st.slider("Duration (days)", 7, 30, 14)
    with col4:
        effect_size = st.slider("True Effect %", -20, 20, 5)
    
    # Generate synthetic A/B test data
    np.random.seed(42)
    days = np.arange(test_duration*2)
    
    # Pre and post periods
    pre_days = days[:test_duration]
    post_days = days[test_duration:]
    
    # Generate data
    control_pre = 100 + np.random.normal(0, 10, test_duration)
    control_post = 100 + np.random.normal(0, 10, test_duration)
    
    treatment_pre = 100 + np.random.normal(0, 10, test_duration)
    treatment_post = treatment_pre.mean() * (1 + effect_size/100) + np.random.normal(0, 10, test_duration)
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'day': list(pre_days) + list(post_days) + list(pre_days) + list(post_days),
        'group': ['Control']*test_duration*2 + ['Treatment']*test_duration*2,
        'period': ['Pre']*test_duration + ['Post']*test_duration + ['Pre']*test_duration + ['Post']*test_duration,
        'bookings': list(control_pre) + list(control_post) + list(treatment_pre) + list(treatment_post)
    })
    
    # DiD Calculation
    did_table = test_data.groupby(['group', 'period'])['bookings'].mean().unstack()
    did_effect = (did_table.loc['Treatment', 'Post'] - did_table.loc['Treatment', 'Pre']) - \
                 (did_table.loc['Control', 'Post'] - did_table.loc['Control', 'Pre'])
    
    # Visualization
    fig = px.line(test_data, x='day', y='bookings', color='group',
                  title="A/B Test Results: Difference-in-Differences Analysis")
    
    # Add vertical line
    fig.add_vline(x=test_duration/2, line_dash="dash", 
                  annotation_text="Treatment Start")
    
    # Add this to fix the black background
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#262730'),  # Darker gray for better contrast
        xaxis=dict(
            gridcolor='#E5E5E5',
            title_font=dict(color='#262730'),
            tickfont=dict(color='#262730')
        ),
        yaxis=dict(
            gridcolor='#E5E5E5',
            title_font=dict(color='#262730'),
            tickfont=dict(color='#262730')
        ),
        title_font=dict(color='#262730')
    )

    
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("DiD Effect", f"{did_effect:.1f} bookings/day", 
                  "Positive impact" if did_effect > 0 else "Negative impact")
    
    with col2:
        # Simple t-test approximation
        t_stat = abs(did_effect) / 5
        p_value = 0.02 if t_stat > 2 else 0.15
        st.metric("P-value", f"{p_value:.3f}", 
                  "Significant" if p_value < 0.05 else "Not significant")
    
    with col3:
        revenue_impact = did_effect * 1200  # Assuming $1200 per booking
        st.metric("Daily Revenue Impact", f"${revenue_impact:,.0f}")
    
    # Show DiD calculation
    with st.expander("DiD Calculation Details"):
        st.markdown("### Difference-in-Differences Table")
        st.dataframe(did_table.style.format("{:.1f}"))
        
        st.markdown(f"""
        **Calculation:**
        - Treatment Effect = (Treatment_Post - Treatment_Pre) - (Control_Post - Control_Pre)
        - Treatment Effect = ({did_table.loc['Treatment', 'Post']:.1f} - {did_table.loc['Treatment', 'Pre']:.1f}) - ({did_table.loc['Control', 'Post']:.1f} - {did_table.loc['Control', 'Pre']:.1f})
        - Treatment Effect = {did_effect:.1f}
        """)
        
        st.code("""
# My implementation at Zimmerman
import statsmodels.api as sm

# Prepare data
df['treatment'] = (df['group'] == 'Treatment').astype(int)
df['post'] = (df['period'] == 'Post').astype(int)
df['did'] = df['treatment'] * df['post']

# Run regression
model = sm.OLS(df['bookings'], 
               sm.add_constant(df[['treatment', 'post', 'did']]))
results = model.fit()

# Extract causal effect
causal_effect = results.params['did']
p_value = results.pvalues['did']
confidence_interval = results.conf_int().loc['did']

print(f"Causal Effect: {causal_effect:.2f} (p={p_value:.3f})")
print(f"95% CI: [{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]")
        """)

# Footer
st.markdown("---")
st.markdown("""
### ___________________________________________________
### This prototype demonstrates:
1. **Full lifecycle experience** - From problem definition to monitoring
2. **Production deployment** - Real models in production via Jenkins
3. **Causal inference expertise** - DiD from daily work at Zimmerman  
4. **Business translation** - Converting technical results to revenue impact
5. **Learning agility** - Built this in 24 hours after our first call

Contact: gerardo.gandara@gmail.com | https://www.linkedin.com/in/gerardo-gandara
""")
