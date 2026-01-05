# Streamlit app to read v_stock_features view from postgres and display with filters
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import asyncio
import util

# Import the AI search utility from util.py
from util import ai_search_tab, create_agent_aws, run_async

# Ensure ChatMessage is imported
from langchain_core.messages import ChatMessage

# Page configuration
st.set_page_config(page_title="Stock Features Report", layout="wide")

if os.environ.get('RUNNING_IN_DOCKER') == 'true':
    CONNECTION_URI = st.secrets["connections"]["postgres"]["url_docker"]
else:
    CONNECTION_URI = st.secrets["connections"]["postgres"]["url"]


@st.cache_data(ttl=30000)  # Cache for 5 minutes
def load_data():
    """Load data from v_stock_features view with caching"""
    engine = create_engine(CONNECTION_URI, poolclass=NullPool)
    query = "SELECT * FROM alpha.v_stock_features"
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    # Clean data: set alpha = 10000 to null
    if 'alpha' in df.columns:
        df.loc[df['alpha'] == 10000, 'alpha'] = pd.NA
    
    # Clean data: set atr > 20 to null
    if 'atr' in df.columns:
        df.loc[df['atr'] > 20, 'atr'] = pd.NA
    
    return df

@st.cache_data(ttl=30000)  # Cache for 5 minutes
def load_distribution_data():
    """Load data from a_stock_features table for distribution analysis"""
    engine = create_engine(CONNECTION_URI, poolclass=NullPool)
    query = "SELECT * FROM alpha.a_stock_features"
    df = pd.read_sql(query, engine)
    engine.dispose()
    
    # Clean data: set alpha = 10000 to null
    if 'alpha' in df.columns:
        df.loc[df['alpha'] == 10000, 'alpha'] = pd.NA
    
    # Clean data: set atr > 20 to null
    if 'atr' in df.columns:
        df.loc[df['atr'] > 20, 'atr'] = pd.NA
    
    return df

@st.cache_data
def get_column_info(df):
    """Identify low cardinality columns (dimensions) for dropdown filters"""
    dimension_cols = []
    numeric_cols = []
    
    for col in df.columns:
        if df[col].dtype in ['object', 'category', 'bool']:
            dimension_cols.append(col)
        elif df[col].dtype in ['int64', 'float64']:
            unique_count = df[col].nunique()
            # Consider as dimension if less than 50 unique values
            if unique_count < 50:
                dimension_cols.append(col)
            else:
                numeric_cols.append(col)
    
    return dimension_cols, numeric_cols

# Main app
# Load data first to get max date
with st.spinner("Loading data..."):
    df = load_data()

# Initialize session state for AI agent
if "loop" not in st.session_state:
    st.session_state.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(st.session_state.loop)

if "agent" not in st.session_state:
    st.session_state.agent = run_async(create_agent_aws(), st.session_state.loop)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="I am a stock market assistant. I can help with questions about stocks, market trends, and analyst reviews.")
    ]

# Get max date1 from the data
max_date = ""
if 'date1' in df.columns:
    max_date_val = pd.to_datetime(df['date1']).max()
    if pd.notna(max_date_val):
        max_date = f" - As of {max_date_val.strftime('%Y-%m-%d')}"

st.title(f"📊 Unified Dashboard{max_date}")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Stock Features", 
    "🏭 Industry Metrics", 
    "💡 Stock Insights", 
    "📊 Distribution Analysis",
    "🤖 AI Search"
])

# Tab 1: Stock Features (existing content)
with tab1:
    #st.markdown("---")
        
    # Exclude specified columns from the table
    EXCLUDE_COLUMNS = ['date1', 'latest', 'wave', 'gcpc']
    df = df.drop(columns=[col for col in EXCLUDE_COLUMNS if col in df.columns])
    
    # Reorder columns to move buy_score and sell_score right after company
    cols = df.columns.tolist()
    
    if 'buy_score' in cols and 'company' in cols:
        cols.remove('buy_score')
        company_index = cols.index('company')
        cols.insert(company_index + 1, 'buy_score')
    
    if 'sell_score' in cols and 'company' in cols:
        cols.remove('sell_score')
        company_index = cols.index('company')
        # Insert after buy_score if it exists, otherwise after company
        if 'buy_score' in cols:
            buy_score_index = cols.index('buy_score')
            cols.insert(buy_score_index + 1, 'sell_score')
        else:
            cols.insert(company_index + 1, 'sell_score')
    
    # Reorder columns to move date1, cur_hl_pl, prev_hl_pl, and kish_score before beta
    # Remove columns that we want to reposition
    cols_to_move = []
    if 'date1' in cols:
        cols.remove('date1')
        cols_to_move.append('date1')
    if 'cur_hl_pl' in cols:
        cols.remove('cur_hl_pl')
        cols_to_move.append('cur_hl_pl')
    if 'prev_hl_pl' in cols:
        cols.remove('prev_hl_pl')
        cols_to_move.append('prev_hl_pl')
    kish_score_col = None
    if 'kish_score' in cols:
        cols.remove('kish_score')
        kish_score_col = 'kish_score'

    # Find the position of 'beta' column
    beta_index = cols.index('beta') if 'beta' in cols else len(cols)

    # Insert the columns before 'beta'
    for i, col in enumerate(cols_to_move):
        cols.insert(beta_index + i, col)
    if kish_score_col:
        cols.insert(beta_index + len(cols_to_move), kish_score_col)

    df = df[cols]
    
    dimension_cols, numeric_cols = get_column_info(df)

#st.markdown("---")

# Filters section
st.sidebar.header("🔍 Filters")

# Initialize session state for clearing filters
if 'clear_filters' not in st.session_state:
    st.session_state.clear_filters = False

# Clear filters button
if st.sidebar.button("🔄 Clear All Filters", width='stretch'):
    # Clear all filter-related keys from session state
    keys_to_delete = [key for key in st.session_state.keys() if key.startswith('filter_') or key.startswith('range_') or key.startswith('bullish_') or key.startswith('watch_') or key.startswith('latest_') or key.startswith('signal_')]
    for key in keys_to_delete:
        del st.session_state[key]
    st.session_state.clear_filters = True
    st.rerun()

# Buy/Sell Signal Filter (at the top)
if 'buy_score' in df.columns and 'sell_score' in df.columns:
    st.sidebar.subheader("📊 Buy/Sell Signal")
    signal_options = ["All", "Buy", "Sell", "Q Buy", "Q Sell"]
    selected_signal = st.sidebar.selectbox(
        "Filter by Signal:",
        options=signal_options,
        index=0,
        key="signal_filter"
    )

# Columns to exclude from filters
EXCLUDE_FROM_FILTERS = ['date1', 'latest', 'wave']

# Create filters for dimension columns
filtered_df = df.copy()

# Apply Buy/Sell signal filter if columns exist
if 'buy_score' in filtered_df.columns and 'sell_score' in filtered_df.columns and 'qscore' in filtered_df.columns:
    if selected_signal == "Buy":
        # qscore > 1 and buy_score > sell_score
        filtered_df = filtered_df[(filtered_df['qscore'] > 1) & (filtered_df['buy_score'] > filtered_df['sell_score'])]
    elif selected_signal == "Sell":
        # qscore > 1 and sell_score >= buy_score
        filtered_df = filtered_df[(filtered_df['qscore'] > 1) & (filtered_df['sell_score'] >= filtered_df['buy_score'])]
    elif selected_signal == "Q Buy":
        # qscore <= 1 and buy_score > sell_score
        filtered_df = filtered_df[(filtered_df['qscore'] <= 1) & (filtered_df['buy_score'] > filtered_df['sell_score'])]
    elif selected_signal == "Q Sell":
        # qscore <= 1 and sell_score >= buy_score
        filtered_df = filtered_df[(filtered_df['qscore'] <= 1) & (filtered_df['sell_score'] >= filtered_df['buy_score'])]
    # "All" option requires no filtering

for col in dimension_cols:
    # Skip excluded columns
    if col in EXCLUDE_FROM_FILTERS:
        continue
    
    unique_values = df[col].dropna().unique()
    if len(unique_values) > 0 and len(unique_values) < 100:  # Only show dropdown if reasonable number
        selected = st.sidebar.multiselect(
            f"Filter by {col}",
            options=sorted(unique_values.tolist()),
            default=None,
            key=f"filter_{col}"
        )
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

# Numeric range filters for selected numeric columns
if numeric_cols:
    st.sidebar.subheader("Numeric Filters")
    show_numeric = st.sidebar.checkbox("Show numeric range filters")
    
    if show_numeric:
        # Prioritize buy_score, sell_score, kish_score, beta, alpha, and industry_rank columns
        priority_cols = ['buy_score', 'sell_score', 'kish_score', 'beta', 'alpha', 'industry_rank']
        display_cols = []
        
        # Add priority columns first if they exist
        for col in priority_cols:
            if col in numeric_cols:
                display_cols.append(col)
        
        # Add remaining numeric columns up to a limit
        for col in numeric_cols:
            if col not in display_cols and len(display_cols) < 7:
                display_cols.append(col)
        
        for col in display_cols:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            
            if min_val != max_val:  # Only show if there's a range
                range_values = st.sidebar.slider(
                    f"{col}",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"range_{col}"
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= range_values[0]) & 
                    (filtered_df[col] <= range_values[1])
                ]

# Search box
st.sidebar.subheader("Search")
search_term = st.sidebar.text_input("Search in all columns")
if search_term:
    mask = filtered_df.astype(str).apply(
        lambda x: x.str.contains(search_term, case=False, na=False)
    ).any(axis=1)
    filtered_df = filtered_df[mask]

with tab1:

    # Display data info - calculated from filtered_df
    # Row 1 - All 7 metrics
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        # Count breakout buy stocks (breakout_signal='BREAKOUT_BUY' and qscore>1)
        breakout_buy_count = 0
        if 'breakout_signal' in filtered_df.columns and 'qscore' in filtered_df.columns:
            breakout_buy_count = len(filtered_df[(filtered_df['breakout_signal'] == 'BREAKOUT_BUY') & (filtered_df['qscore'] > 1)])
        st.metric("Breakout Buy", breakout_buy_count)
        show_breakout_buy = st.checkbox("Show Breakout Buy", key="breakout_buy_filter")
    with col2:
        # Count breakout sell stocks (breakout_signal='BREAKOUT_SELL' and qscore>1)
        breakout_sell_count = 0
        if 'breakout_signal' in filtered_df.columns and 'qscore' in filtered_df.columns:
            breakout_sell_count = len(filtered_df[(filtered_df['breakout_signal'] == 'BREAKOUT_SELL') & (filtered_df['qscore'] > 1)])
        st.metric("Breakout Sell", breakout_sell_count)
        show_breakout_sell = st.checkbox("Show Breakout Sell", key="breakout_sell_filter")
    with col3:
        # Count watch indicator stocks (where watch_ind == 'Y')
        watch_count = len(filtered_df[filtered_df['watch_ind'] == 'Y']) if 'watch_ind' in filtered_df.columns else 0
        st.metric("Watch Indicator", watch_count)
        show_watch_only = st.checkbox("Show Watched", key="watch_filter")
    with col4:
        # Count latest winners (cur_hl_pl > 10 and sincedays < 10)
        latest_winners_count = 0
        if 'cur_hl_pl' in filtered_df.columns and 'sincedays' in filtered_df.columns:
            latest_winners_count = len(filtered_df[(filtered_df['cur_hl_pl'] > 10) & (filtered_df['sincedays'] < 10)])
        st.metric("Latest Winners", latest_winners_count)
        show_latest_winners_only = st.checkbox("Filter Latest Winners", key="latest_winners_filter")
    with col5:
        # Count latest losers (cur_hl_pl < -10 and sincedays < 10)
        latest_losers_count = 0
        if 'cur_hl_pl' in filtered_df.columns and 'sincedays' in filtered_df.columns:
            latest_losers_count = len(filtered_df[(filtered_df['cur_hl_pl'] < -10) & (filtered_df['sincedays'] < 10)])
        st.metric("Latest Losers", latest_losers_count)
        show_latest_losers_only = st.checkbox("Filter Latest Losers", key="latest_losers_filter")
    with col6:
        # Count longest winners (cur_hl_pl > 10 and sincedays > 65)
        longest_winners_count = 0
        if 'cur_hl_pl' in filtered_df.columns and 'sincedays' in filtered_df.columns:
            longest_winners_count = len(filtered_df[(filtered_df['cur_hl_pl'] > 10) & (filtered_df['sincedays'] > 65)])
        st.metric("Longest Winners", longest_winners_count)
        show_longest_winners_only = st.checkbox("Filter Longest Winners", key="longest_winners_filter")
    with col7:
        # Count longest losers (cur_hl_pl < -10 and sincedays > 65)
        longest_losers_count = 0
        if 'cur_hl_pl' in filtered_df.columns and 'sincedays' in filtered_df.columns:
            longest_losers_count = len(filtered_df[(filtered_df['cur_hl_pl'] < -10) & (filtered_df['sincedays'] > 65)])
        st.metric("Longest Losers", longest_losers_count)
        show_longest_losers_only = st.checkbox("Filter Longest Losers", key="longest_losers_filter")

    # Apply checkbox filters after sidebar filters
    if show_breakout_buy and 'breakout_signal' in filtered_df.columns and 'qscore' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['breakout_signal'] == 'BREAKOUT_BUY') & (filtered_df['qscore'] > 1)]

    if show_breakout_sell and 'breakout_signal' in filtered_df.columns and 'qscore' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['breakout_signal'] == 'BREAKOUT_SELL') & (filtered_df['qscore'] > 1)]

    if show_watch_only and 'watch_ind' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['watch_ind'] == 'Y']

    if show_latest_winners_only and 'cur_hl_pl' in filtered_df.columns and 'sincedays' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['cur_hl_pl'] > 10) & (filtered_df['sincedays'] < 10)]

    if show_latest_losers_only and 'cur_hl_pl' in filtered_df.columns and 'sincedays' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['cur_hl_pl'] < -10) & (filtered_df['sincedays'] < 10)]

    if show_longest_winners_only and 'cur_hl_pl' in filtered_df.columns and 'sincedays' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['cur_hl_pl'] > 10) & (filtered_df['sincedays'] > 65)]

    if show_longest_losers_only and 'cur_hl_pl' in filtered_df.columns and 'sincedays' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['cur_hl_pl'] < -10) & (filtered_df['sincedays'] > 65)]

    #st.markdown("---")


    # Display filtered results with clickable symbol field
    st.subheader(f"📋 Detailed Summary ({len(filtered_df)} stocks)")

    # Ensure sell_score is positioned right after buy_score in the detailed summary
    display_cols = filtered_df.columns.tolist()
    if 'sell_score' in display_cols and 'buy_score' in display_cols:
        display_cols.remove('sell_score')
        buy_score_idx = display_cols.index('buy_score')
        display_cols.insert(buy_score_idx + 1, 'sell_score')
        filtered_df = filtered_df[display_cols]

    # Add a checkbox column for selecting rows to copy prompt
    styled_df = filtered_df.copy()
    
    # Initialize selected row in session state
    if 'selected_row_index' not in st.session_state:
        st.session_state.selected_row_index = None
    
    # Validate that the selected index is still valid for the current filtered dataframe
    if st.session_state.selected_row_index is not None and st.session_state.selected_row_index >= len(filtered_df):
        st.session_state.selected_row_index = None
    
    # Set Copy column based on current selection
    styled_df.insert(0, 'Copy', False)
    if st.session_state.selected_row_index is not None and st.session_state.selected_row_index < len(styled_df):
        styled_df.loc[st.session_state.selected_row_index, 'Copy'] = True
    
    # Use st.data_editor with checkbox column
    edited_df = st.data_editor(
        styled_df,
        column_config={
            "Copy": st.column_config.CheckboxColumn(
                "Copy",
                help="Select to generate AI prompt",
                default=False,
            )
        },
        disabled=[col for col in styled_df.columns if col != "Copy"],
        hide_index=True,
        width='stretch',
        key="detailed_summary_table"
    )
    
    # Detect selection changes (enforce single selection)
    selected_indices = edited_df.index[edited_df['Copy'] == True].tolist()
    
    if len(selected_indices) > 0:
        # Get the most recently selected
        new_selection = selected_indices[-1]
        
        # If this is a different selection or multiple are selected, update and rerun
        if new_selection != st.session_state.selected_row_index or len(selected_indices) > 1:
            st.session_state.selected_row_index = new_selection
            st.rerun()
    elif st.session_state.selected_row_index is not None:
        # User unchecked the current selection
        st.session_state.selected_row_index = None
        st.rerun()
    
    # Display prompt for selected row (with bounds check)
    if st.session_state.selected_row_index is not None and st.session_state.selected_row_index < len(filtered_df):
        row = filtered_df.iloc[st.session_state.selected_row_index]
        ai_prompt = f"Explain current PL of {row['cur_hl_pl']}% for Ticker {row['symbol']} since {row['sincedays']} days after prior movement of {row['prev_hl_pl']}%. What is analysts review on this stock?"
        
        st.markdown(f"**📋 AI Prompt for {row['symbol']}:**")
        st.code(ai_prompt, language=None)
        st.caption("💡 Hover over the text above and click the copy icon in the top-right corner")

    # Download button
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name="stock_features_filtered.csv",
        mime="text/csv"
    )

    # Add interactivity to filter by stock symbol using distribution data
    drill_col1, drill_col2 = st.columns([3, 1])
    with drill_col1:
        st.subheader("Drill Down by Stock")
    with drill_col2:
        selected_symbol = st.selectbox("Select a Stock Symbol", options=filtered_df['symbol'].unique(), label_visibility="collapsed")

    if selected_symbol:
        # Load distribution data
        distribution_data = load_distribution_data()

        # Filter the distribution data based on the selected symbol
        drilled_down_df = distribution_data[distribution_data['symbol'] == selected_symbol]
        # Sort the distribution data by 'date1' in descending order
        drilled_down_df = drilled_down_df.sort_values(by='date1', ascending=False)
        # Reorder columns to move date1, cur_hl_pl, prev_hl_pl, and kish_score before beta in the drill-down table
        cols = drilled_down_df.columns.tolist()

        # Remove columns that we want to reposition
        cols_to_move = []
        if 'date1' in cols:
            cols.remove('date1')
            cols_to_move.append('date1')
        if 'cur_hl_pl' in cols:
            cols.remove('cur_hl_pl')
            cols_to_move.append('cur_hl_pl')
        if 'prev_hl_pl' in cols:
            cols.remove('prev_hl_pl')
            cols_to_move.append('prev_hl_pl')
        kish_score_col = None
        if 'kish_score' in cols:
            cols.remove('kish_score')
            kish_score_col = 'kish_score'

        # Find the position of 'beta' column
        beta_index = cols.index('beta') if 'beta' in cols else len(cols)

        # Insert the columns before 'beta'
        for i, col in enumerate(cols_to_move):
            cols.insert(beta_index + i, col)
        if kish_score_col:
            cols.insert(beta_index + len(cols_to_move), kish_score_col)

        drilled_down_df = drilled_down_df[cols]
        
        # Filter the drill-down table to only show the latest record for each wave BEFORE transforming date1
        # This preserves the original date1 sort order (most recent data first)
        if 'wave' in drilled_down_df.columns:
            drilled_down_df = drilled_down_df.drop_duplicates(subset=['wave'], keep='first')
        
        # Calculate new date1 column as date1 - sincedays and rename it to 'start_date'
        # sincedays is index diff (trading days), multiply by 7/5 to get approximate calendar days
        if 'date1' in drilled_down_df.columns and 'sincedays' in drilled_down_df.columns:
            drilled_down_df['start_date'] = pd.to_datetime(drilled_down_df['date1']) - pd.to_timedelta(drilled_down_df['sincedays'] * 7 / 5, unit='D')
            drilled_down_df = drilled_down_df.drop(columns=['date1'])

        # Reorder columns to place 'start_date' in the desired position
        cols = drilled_down_df.columns.tolist()
        if 'start_date' in cols:
            cols.remove('start_date')
            cols.insert(0, 'start_date')
        drilled_down_df = drilled_down_df[cols]
        
        # Breakout Signals Table - show latest record for each wave where breakout_signal='BREAKOUT_BUY'
        st.subheader(f"🚀 Breakout Signals for {selected_symbol}")
        
        if 'breakout_signal' in distribution_data.columns:
            # Filter for the selected symbol and breakout signals
            breakout_df = distribution_data[
                (distribution_data['symbol'] == selected_symbol) & 
                (distribution_data['breakout_signal'] == 'BREAKOUT_BUY')
            ].copy()
            
            if len(breakout_df) > 0:
                # Sort by date1 descending to get latest first
                breakout_df = breakout_df.sort_values(by='date1', ascending=False)
                
                # Keep only the latest record for each wave
                if 'wave' in breakout_df.columns:
                    breakout_df = breakout_df.drop_duplicates(subset=['wave'], keep='first')
                
                # Calculate start_date
                if 'sincedays' in breakout_df.columns:
                    breakout_df['start_date'] = pd.to_datetime(breakout_df['date1']) - pd.to_timedelta(breakout_df['sincedays'] * 7 / 5, unit='D')
                
                # Select relevant columns for display
                breakout_display_cols = ['start_date', 'date1', 'wave', 'close', 'cur_hl_pl', 'sincedays', 
                                        'buy_score', 'sell_score', 'stage_cycle', 'zigzag', 'macd', 'bollband', 'rsi', 'adx']
                # Only include columns that exist
                breakout_display_cols = [col for col in breakout_display_cols if col in breakout_df.columns]
                
                breakout_display = breakout_df[breakout_display_cols]
                
                st.dataframe(breakout_display, hide_index=True)
                
                # Download button for breakout signals
                breakout_csv = breakout_display.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Breakout Signals as CSV",
                    data=breakout_csv,
                    file_name=f"breakout_signals_{selected_symbol}.csv",
                    mime="text/csv",
                    key=f"download_breakout_{selected_symbol}"
                )
            else:
                st.info(f"No breakout signals found for {selected_symbol}")
        else:
            st.warning("breakout_signal column not found in the dataset")
        
        st.subheader(f"Stage Cycles Distribution for {selected_symbol}")
        
        st.dataframe(drilled_down_df)
        
        # Load and display prediction data
        st.subheader(f"Predictions for {selected_symbol}")
        
        engine = create_engine(CONNECTION_URI, poolclass=NullPool)
        prediction_query = f"SELECT * FROM alpha.a_predict_stg_cycle WHERE symbol = '{selected_symbol}' ORDER BY date1 DESC"
        prediction_df = pd.read_sql(prediction_query, engine)
        engine.dispose()
        
        if len(prediction_df) > 0:
            # Round score values if score column exists (already in 0-100 range)
            if 'score' in prediction_df.columns:
                prediction_df['score'] = prediction_df['score'].round(2)
            
            st.dataframe(prediction_df, hide_index=True)
            
            # Download button for predictions
            prediction_csv = prediction_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions as CSV",
                data=prediction_csv,
                file_name=f"predictions_{selected_symbol}.csv",
                mime="text/csv",
                key=f"download_predictions_{selected_symbol}"
            )
        else:
            st.info(f"No prediction data available for {selected_symbol}")

# Tab 2: Industry Metrics
with tab2:
    #st.subheader("🏭 Industry Performance Metrics")
    
    # Load data for industry metrics
    with st.spinner("Loading industry data..."):
        if 'df' not in locals():
            df = load_data()
        
        # Check if gicssubindustry column exists
        if 'gicssubindustry' in df.columns:
            # Group by gicssubindustry and calculate overall metrics first
            industry_metrics = df.groupby('gicssubindustry').agg({
                'cur_hl_pl': ['mean', 'count', lambda x: (x > 0).sum(), lambda x: (x < 0).sum()],
                'rsi': 'median',
                'alpha': 'median'
            }).round(2)
            
            # Flatten column names
            industry_metrics.columns = ['Avg P/L', 'Stock Count', 'Bullish Count', 'Bearish Count', 'Avg RSI', 'Avg Alpha']
            industry_metrics = industry_metrics.reset_index()
            
            # Filter for latest stocks (sincedays < 10) for latest metrics
            df_latest = df[df['sincedays'] < 10] if 'sincedays' in df.columns else df
            
            # Filter for longest stocks (sincedays > 65) for longest metrics
            df_longest = df[df['sincedays'] > 65] if 'sincedays' in df.columns else df
            
            # Calculate latest winner/loser counts per industry
            latest_industry_metrics = df_latest.groupby('gicssubindustry').agg({
                'cur_hl_pl': [lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
            })
            latest_industry_metrics.columns = ['Latest Bullish Count', 'Latest Bearish Count']
            latest_industry_metrics = latest_industry_metrics.reset_index()
            
            # Calculate longest winner/loser counts per industry
            longest_industry_metrics = df_longest.groupby('gicssubindustry').agg({
                'cur_hl_pl': [lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
            })
            longest_industry_metrics.columns = ['Longest Bullish Count', 'Longest Bearish Count']
            longest_industry_metrics = longest_industry_metrics.reset_index()
            
            # Merge with latest metrics
            industry_metrics = industry_metrics.merge(latest_industry_metrics[['gicssubindustry', 'Latest Bullish Count', 'Latest Bearish Count']], on='gicssubindustry', how='left')
            
            # Merge with longest metrics
            industry_metrics = industry_metrics.merge(longest_industry_metrics[['gicssubindustry', 'Longest Bullish Count', 'Longest Bearish Count']], on='gicssubindustry', how='left')
            
            # Fill NaN values with 0 for industries that don't have latest or longest stocks
            industry_metrics[['Latest Bullish Count', 'Latest Bearish Count', 'Longest Bullish Count', 'Longest Bearish Count']] = industry_metrics[['Latest Bullish Count', 'Latest Bearish Count', 'Longest Bullish Count', 'Longest Bearish Count']].fillna(0)
            
            # Calculate percentages based on TOTAL industry stock count
            industry_metrics['Latest Bullish %'] = ((industry_metrics['Latest Bullish Count'] / industry_metrics['Stock Count']) * 100).round(2)
            industry_metrics['Latest Bearish %'] = ((industry_metrics['Latest Bearish Count'] / industry_metrics['Stock Count']) * 100).round(2)
            industry_metrics['Longest Bullish %'] = ((industry_metrics['Longest Bullish Count'] / industry_metrics['Stock Count']) * 100).round(2)
            industry_metrics['Longest Bearish %'] = ((industry_metrics['Longest Bearish Count'] / industry_metrics['Stock Count']) * 100).round(2)
            
            # Calculate Bullish % and Bearish %
            industry_metrics['Bullish %'] = ((industry_metrics['Bullish Count'] / industry_metrics['Stock Count']) * 100).round(2)
            industry_metrics['Bearish %'] = ((industry_metrics['Bearish Count'] / industry_metrics['Stock Count']) * 100).round(2)
            
            # Reorder columns
            industry_metrics = industry_metrics[['gicssubindustry', 'Avg P/L', 'Stock Count', 'Bullish %', 'Bearish %', 'Latest Bullish %', 'Latest Bearish %', 'Longest Bullish %', 'Longest Bearish %', 'Avg RSI', 'Avg Alpha']]
            industry_metrics = industry_metrics.sort_values('Avg P/L', ascending=False)
            
            # Store original for filtering
            industry_metrics_original = industry_metrics.copy()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Count latest winner industries (Latest Bullish % > 33% and Latest Bearish % <= 33%)
                winner_industries_count = len(industry_metrics[
                    (industry_metrics['Latest Bullish %'] > 33) & 
                    (industry_metrics['Latest Bullish %'].notna()) &
                    ~((industry_metrics['Latest Bearish %'] > 33) & (industry_metrics['Latest Bearish %'].notna()))
                ])
                st.metric("Latest Winner Industries", winner_industries_count)
                show_winner_industries = st.checkbox("Filter Winner Industries", key="winner_industries_filter")
            with col2:
                # Count latest loser industries (Latest Bearish % > 33% and Latest Bullish % <= 33%)
                loser_industries_count = len(industry_metrics[
                    (industry_metrics['Latest Bearish %'] > 33) & 
                    (industry_metrics['Latest Bearish %'].notna()) &
                    ~((industry_metrics['Latest Bullish %'] > 33) & (industry_metrics['Latest Bullish %'].notna()))
                ])
                st.metric("Latest Loser Industries", loser_industries_count)
                show_loser_industries = st.checkbox("Filter Loser Industries", key="loser_industries_filter")
            with col3:
                # Count longest winner industries (Longest Bullish % > 51% and Longest Bearish % <= 51%)
                longest_winner_industries_count = len(industry_metrics[
                    (industry_metrics['Longest Bullish %'] > 33) & 
                    (industry_metrics['Longest Bullish %'].notna()) &
                    ~((industry_metrics['Longest Bearish %'] > 33) & (industry_metrics['Longest Bearish %'].notna()))
                ])
                st.metric("Longest Winner Industries", longest_winner_industries_count)
                show_longest_winner_industries = st.checkbox("Filter Longest Winner Industries", key="longest_winner_industries_filter")
            with col4:
                # Count longest loser industries (Longest Bearish % > 51% and Longest Bullish % <= 51%)
                longest_loser_industries_count = len(industry_metrics[
                    (industry_metrics['Longest Bearish %'] > 33) & 
                    (industry_metrics['Longest Bearish %'].notna()) &
                    ~((industry_metrics['Longest Bullish %'] > 33) & (industry_metrics['Longest Bullish %'].notna()))
                ])
                st.metric("Longest Loser Industries", longest_loser_industries_count)
                show_longest_loser_industries = st.checkbox("Filter Longest Loser Industries", key="longest_loser_industries_filter")
            
            # Apply filters based on checkbox selections (with exclusivity)
            if show_winner_industries and 'Latest Bullish %' in industry_metrics.columns:
                industry_metrics = industry_metrics[
                    (industry_metrics['Latest Bullish %'] > 33) & 
                    (industry_metrics['Latest Bullish %'].notna()) &
                    ~((industry_metrics['Latest Bearish %'] > 33) & (industry_metrics['Latest Bearish %'].notna()))
                ]
            
            if show_loser_industries and 'Latest Bearish %' in industry_metrics.columns:
                industry_metrics = industry_metrics[
                    (industry_metrics['Latest Bearish %'] > 33) & 
                    (industry_metrics['Latest Bearish %'].notna()) &
                    ~((industry_metrics['Latest Bullish %'] > 33) & (industry_metrics['Latest Bullish %'].notna()))
                ]
            
            if show_longest_winner_industries and 'Longest Bullish %' in industry_metrics.columns:
                industry_metrics = industry_metrics[
                    (industry_metrics['Longest Bullish %'] > 33) & 
                    (industry_metrics['Longest Bullish %'].notna()) &
                    ~((industry_metrics['Longest Bearish %'] > 33) & (industry_metrics['Longest Bearish %'].notna()))
                ]
            
            if show_longest_loser_industries and 'Longest Bearish %' in industry_metrics.columns:
                industry_metrics = industry_metrics[
                    (industry_metrics['Longest Bearish %'] > 33) & 
                    (industry_metrics['Longest Bearish %'].notna()) &
                    ~((industry_metrics['Longest Bullish %'] > 33) & (industry_metrics['Longest Bullish %'].notna()))
                ]
            
            st.markdown("---")
            
            # Display industry table
            st.dataframe(
                industry_metrics,
                height=500
            )
            
            # Download button for industry metrics
            industry_csv = industry_metrics.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download industry metrics as CSV",
                data=industry_csv,
                file_name="industry_metrics.csv",
                mime="text/csv"
            )
        else:
            st.warning("gicssubindustry column not found in the dataset.")

# Tab 3: Stock Insights
with tab3:
    st.subheader("💡 Stock Insights & Analytics")
    
    # Load data for insights
    with st.spinner("Loading stock insights..."):
        if 'df' not in locals():
            df = load_data()
    
    # Create insights sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Top Performers")
        
        # Top gainers by current P/L
        if 'cur_hl_pl' in df.columns and 'symbol' in df.columns and 'qscore' in df.columns:
            top_gainers = df[df['qscore'] > 1].nlargest(10, 'cur_hl_pl')[['symbol', 'company', 'cur_hl_pl', 'sincedays', 'rsi', 'stage_cycle']]
            st.dataframe(
                top_gainers,
                height=350,
                hide_index=True
            )
        
        st.markdown("### 📉 Top Decliners")
        
        # Top losers by current P/L
        if 'cur_hl_pl' in df.columns and 'symbol' in df.columns and 'qscore' in df.columns:
            top_losers = df[df['qscore'] > 1].nsmallest(10, 'cur_hl_pl')[['symbol', 'company', 'cur_hl_pl', 'sincedays', 'rsi', 'stage_cycle']]
            st.dataframe(
                top_losers,
                height=350,
                hide_index=True
            )
    
    with col2:
        st.markdown("### ⚡ High Momentum Stocks")
        
        # Stocks with high RSI and positive P/L
        if 'rsi' in df.columns and 'cur_hl_pl' in df.columns and 'qscore' in df.columns:
            high_momentum = df[(df['rsi'] > 70) & (df['cur_hl_pl'] > 0) & (df['qscore'] > 1)].nlargest(10, 'cur_hl_pl')[
                ['symbol', 'company', 'cur_hl_pl', 'rsi', 'sincedays', 'alpha']
            ]
            st.dataframe(
                high_momentum,
                height=350,
                hide_index=True
            )
        
        st.markdown("### 🔍 Oversold Opportunities")
        
        # Stocks with low RSI (potential buying opportunities)
        if 'rsi' in df.columns and 'qscore' in df.columns:
            oversold = df[(df['rsi'] < 30) & (df['qscore'] > 1)].nsmallest(10, 'rsi')[
                ['symbol', 'company', 'rsi', 'cur_hl_pl', 'sincedays', 'stage_cycle']
            ]
            st.dataframe(
                oversold,
                height=350,
                hide_index=True
            )
    
    st.markdown("---")
    
    # Additional insights row
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 🌟 High Alpha Stocks")
        
        # Stocks with highest alpha (outperforming the market)
        if 'alpha' in df.columns and 'qscore' in df.columns:
            high_alpha = df[(df['alpha'].notna()) & (df['alpha'] < 100) & (df['qscore'] > 1)].nlargest(10, 'alpha')[
                ['symbol', 'company', 'alpha', 'beta', 'cur_hl_pl', 'industry_rank']
            ]
            st.dataframe(
                high_alpha,
                height=350,
                hide_index=True
            )
        
        st.markdown("### 👁️ Watch List Stocks")
        
        # Stocks marked for watching with positive performance
        if 'watch_ind' in df.columns and 'qscore' in df.columns:
            watch_stocks = df[(df['watch_ind'] == 'Y') & (df['cur_hl_pl'] > 0) & (df['qscore'] > 1)].nlargest(10, 'cur_hl_pl')[
                ['symbol', 'company', 'cur_hl_pl', 'kish_score', 'rsi', 'sincedays']
            ]
            st.dataframe(
                watch_stocks,
                height=350,
                hide_index=True
            )
    
    with col4:
        st.markdown("### 🚀 Bullish Signals")
        
        # Stocks with bullish indicators (MACD and Bullish flags)
        if 'macd' in df.columns and 'bullish' in df.columns and 'qscore' in df.columns:
            bullish_stocks = df[
                (df['macd'] == "True") & 
                (df['bullish'] == "True") & 
                (df['cur_hl_pl'] > 0) & 
                (df['qscore'] > 1)
            ].nlargest(10, 'cur_hl_pl')[
                ['symbol', 'company', 'cur_hl_pl', 'rsi', 'kish_score', 'sincedays']
            ]
            st.dataframe(
                bullish_stocks,
                height=350,
                hide_index=True
            )
        
        st.markdown("### 🏆 Top Industry Performers")
        
        # Top stock in each industry by rank
        if 'industry_rank' in df.columns and 'gicssubindustry' in df.columns and 'qscore' in df.columns:
            top_by_industry = df[(df['industry_rank'] >90) & (df['qscore'] > 1)].nlargest(10, 'cur_hl_pl')[
                ['symbol', 'company', 'gicssubindustry', 'cur_hl_pl', 'industry_rank', 'kish_score']
            ]
            st.dataframe(
                top_by_industry,
                height=350,
                hide_index=True
            )
        
    # Summary statistics
    st.markdown("### 📊 Market Summary")
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    
    with summary_col1:
        avg_pl = df['cur_hl_pl'].mean() if 'cur_hl_pl' in df.columns else 0
        st.metric("Average P/L", f"{avg_pl:.2f}%")
        
        median_rsi = df['rsi'].median() if 'rsi' in df.columns else 0
        st.metric("Median RSI", f"{median_rsi:.2f}")
    
    with summary_col2:
        bullish_pct = (df['cur_hl_pl'] > 0).sum() / len(df) * 100 if 'cur_hl_pl' in df.columns else 0
        st.metric("Bullish Stocks %", f"{bullish_pct:.1f}%")
        
        avg_alpha = df['alpha'].mean() if 'alpha' in df.columns else 0
        st.metric("Average Alpha", f"{avg_alpha:.4f}")
    
    with summary_col3:
        high_momentum_count = len(df[(df['rsi'] > 70) & (df['cur_hl_pl'] > 0)]) if 'rsi' in df.columns and 'cur_hl_pl' in df.columns else 0
        st.metric("High Momentum Stocks", high_momentum_count)
        
        oversold_count = len(df[df['rsi'] < 30]) if 'rsi' in df.columns else 0
        st.metric("Oversold Stocks", oversold_count)
    
    with summary_col4:
        watch_count = len(df[df['watch_ind'] == 'Y']) if 'watch_ind' in df.columns else 0
        st.metric("Watch List Stocks", watch_count)
        
        # Top 100 Value Stocks metric
        if 'growth_days' in df.columns and 'fall_days' in df.columns:
            value_stocks_count = len(df[df['growth_days'] < 15])
            st.metric("Top 100 Value Stocks", min(value_stocks_count, 100))
        else:
            st.metric("Top 100 Value Stocks", "N/A")
    
    
    # Top 100 Value Stocks Section
    st.markdown("### 💎 Top 100 Value Stocks")
    st.info("📌 Stocks with growth_days < 15, ordered by fall_days (descending)")
    
    # Filter checkbox for Top 100 Value Stocks
    show_value_stocks = st.checkbox("Filter Top 100 Value Stocks Table", value=False, key="value_stocks_filter")
    
    if show_value_stocks:
        if 'growth_days' in df.columns and 'fall_days' in df.columns and 'qscore' in df.columns:
            # Filter and sort for top 100 value stocks
            value_stocks_df = df[(df['growth_days'] < 15) & (df['qscore'] > 1)].copy()
            value_stocks_df = value_stocks_df.sort_values('fall_days', ascending=False).head(100)
            
            # Select relevant columns for display
            display_columns = ['symbol', 'company', 'growth_days', 'fall_days', 'cur_hl_pl', 'rsi', 'kish_score', 'sincedays', 'stage_cycle']
            # Only include columns that exist in the dataframe
            display_columns = [col for col in display_columns if col in value_stocks_df.columns]
            
            value_stocks_display = value_stocks_df[display_columns]
            
            st.dataframe(
                value_stocks_display,
                height=500,
                hide_index=True
            )
            
            # Download button for value stocks
            value_stocks_csv = value_stocks_display.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Top 100 Value Stocks as CSV",
                data=value_stocks_csv,
                file_name="top_100_value_stocks.csv",
                mime="text/csv"
            )
        else:
            st.warning("Required columns 'growth_days' and/or 'fall_days' not found in the dataset.")

# Tab 4: Distribution Analysis
with tab4:
    
    # Load data for distribution analysis from a_stock_features table
    with st.spinner("Loading data for distribution analysis..."):
        df_dist = load_distribution_data()
    
    # Filter out ETFs
    if 'gicssubindustry' in df_dist.columns:
        df_dist = df_dist[df_dist['gicssubindustry'] != 'ETF'].copy()
    
    # Filter out specific symbols
    if 'symbol' in df_dist.columns:
        exclude_symbols = ['^GSPC', '^IXIC', 'BKNG', 'NVR']
        df_dist = df_dist[~df_dist['symbol'].isin(exclude_symbols)].copy()
    
    # Create columns for filters in one row
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    # Date filter
    with filter_col1:
        if 'date1' in df_dist.columns:
            st.markdown("### 📅 Date")
            available_dates = sorted(df_dist['date1'].dropna().unique(), reverse=True)
            
            if len(available_dates) > 0:
                # Add "All Dates" option at the beginning
                date_options = ["All Dates"] + list(available_dates)
                
                selected_date = st.selectbox(
                    "Select Date:",
                    options=date_options,
                    index=0,
                    key="dist_date_filter"
                )
                
                # Filter data by selected date (if not "All Dates")
                if selected_date != "All Dates":
                    df_dist = df_dist[df_dist['date1'] == selected_date].copy()
    
    # Get numerical columns for metric selection (before filters)
    numerical_cols_temp = df_dist.select_dtypes(include=['int64', 'float64']).columns.tolist()
    # Exclude specific columns from metric selection
    exclude_metrics = ['cur_hl_pl', 'close', 'industry_rank', 'volatality_signals', 'volatility_signals']
    available_cols_temp = [col for col in numerical_cols_temp if col not in exclude_metrics]
    
    # Store original df_dist for prev_stage_cycle options (before stage_cycle filter)
    df_dist_before_stage = df_dist.copy()
    
    # Metric selection
    with filter_col2:
        if available_cols_temp:
            # Column selection for distribution analysis
            st.markdown("### 🎯 Metric")
            
            selected_col = st.selectbox(
                "Select metric:",
                options=available_cols_temp,
                index=0,
                key="primary_metric"
            )
    
    # Stage Cycle filter
    with filter_col3:
        if 'stage_cycle' in df_dist.columns:
            st.markdown("### 🔄 Stage Cycle")
            available_stages = sorted(df_dist['stage_cycle'].dropna().unique())
            
            if len(available_stages) > 0:
                selected_stages = st.multiselect(
                    "Select Stage(s):",
                    options=available_stages,
                    default=[],
                    key="dist_stage_filter"
                )
                
                # Filter data by selected stages (if any selected)
                if selected_stages:
                    df_dist = df_dist[df_dist['stage_cycle'].isin(selected_stages)].copy()
    
    # Bollinger Band filter
    with filter_col4:
        if 'bollband' in df_dist_before_stage.columns:
            st.markdown("### 📊 Bollinger Band")
            # Use original data before stage_cycle filtering for available options
            available_bollbands = sorted(df_dist_before_stage['bollband'].dropna().unique())
            
            if len(available_bollbands) > 0:
                bollband_options = ["All"] + list(available_bollbands)
                
                selected_bollband = st.selectbox(
                    "Select Bollinger Band:",
                    options=bollband_options,
                    index=0,
                    key="dist_bollband_filter"
                )
                
                # Filter data by selected bollinger band (if not "All")
                if selected_bollband != "All":
                    df_dist = df_dist[df_dist['bollband'] == selected_bollband].copy()
    
    # Get numerical columns (after all filters applied)
    numerical_cols = df_dist.select_dtypes(include=['int64', 'float64']).columns.tolist()
    available_cols = [col for col in numerical_cols if col != 'cur_hl_pl']
    
    if available_cols:
        st.markdown("---")
        
        # Distribution visualization for primary metric
        st.markdown(f"### 📈 Distribution: {selected_col}")
        
        # Histogram with individual bar coloring based on cur_hl_pl
        if 'cur_hl_pl' in df_dist.columns:
            # Create bins and calculate median cur_hl_pl for each bin
            import numpy as np
            
            # Remove NaN values for binning
            valid_data = df_dist[[selected_col, 'cur_hl_pl']].dropna()
            
            if len(valid_data) > 0:
                min_val = valid_data[selected_col].min()
                max_val = valid_data[selected_col].max()
                
                # Check if there's variation in the data
                if min_val == max_val or pd.isna(min_val) or pd.isna(max_val):
                    # Fallback for constant values
                    fig_hist = px.histogram(
                        valid_data,
                        x=selected_col,
                        nbins=50,
                        title=f"Histogram - {selected_col}",
                        labels={selected_col: selected_col}
                    )
                    fig_hist.update_layout(showlegend=False, height=500)
                    st.plotly_chart(fig_hist, width='stretch')
                    st.caption("⚠️ All values are identical - no distribution to show")
                else:
                    # Create 50 bins
                    nbins = 50
                    bin_edges = np.linspace(min_val, max_val, nbins + 1)
                    valid_data['bin'] = pd.cut(valid_data[selected_col], bins=bin_edges, include_lowest=True)
                    
                    # Calculate count and median cur_hl_pl for each bin
                    bin_stats = valid_data.groupby('bin', observed=True).agg({
                        selected_col: 'count',
                        'cur_hl_pl': 'median'
                    }).reset_index()
                    
                    # Get bin centers for x-axis
                    bin_stats['bin_center'] = bin_stats['bin'].apply(lambda x: x.mid)
                    
                    # Assign colors based on median cur_hl_pl
                    bin_stats['color'] = bin_stats['cur_hl_pl'].apply(
                        lambda x: 'rgba(239, 85, 59, 0.7)' if x < 0 else 'rgba(0, 204, 150, 0.7)'
                    )
                    
                    # Create histogram using go.Bar
                    fig_hist = go.Figure(data=[go.Bar(
                        x=bin_stats['bin_center'],
                        y=bin_stats[selected_col],
                        marker=dict(
                            color=bin_stats['color'],
                            line=dict(color='white', width=0.5)
                        ),
                        hovertemplate='<b>%{x:.2f}</b><br>Count: %{y}<br>Median P/L: %{customdata:.2f}%<extra></extra>',
                        customdata=bin_stats['cur_hl_pl']
                    )])
                    
                    fig_hist.update_layout(
                        title=f"Histogram - {selected_col} (colored by median cur_hl_pl per bin)",
                        xaxis_title=selected_col,
                        yaxis_title="Count",
                        showlegend=False,
                        height=500
                    )
                    st.plotly_chart(fig_hist, width='stretch')
                    st.caption("🔴 Red bars: median cur_hl_pl < 0 | 🟢 Green bars: median cur_hl_pl ≥ 0")
            else:
                st.warning("No valid data available for histogram")
        else:
            # Fallback if cur_hl_pl not available
            fig_hist = px.histogram(
                df_dist,
                x=selected_col,
                nbins=50,
                title=f"Histogram - {selected_col}",
                labels={selected_col: selected_col}
            )
            fig_hist.update_layout(showlegend=False, height=500)
            st.plotly_chart(fig_hist, width='stretch')
        
        st.markdown("---")
        
        # Stage Transition Sankey Diagram
        if 'stage_cycle' in df_dist.columns and 'prior_stage_cycle' in df_dist.columns and 'cur_hl_pl' in df_dist.columns:
            st.markdown("### 🔄 Stage Cycle Transitions (Sankey Diagram)")
            
            # Prepare data for Sankey diagram
            transition_data = df_dist[['prior_stage_cycle', 'stage_cycle', 'symbol', 'cur_hl_pl']].dropna()
            
            if len(transition_data) > 0:
                # Count transitions with distinct symbols
                transition_counts = transition_data.groupby(['prior_stage_cycle', 'stage_cycle']).agg({
                    'symbol': 'nunique',
                    'cur_hl_pl': 'median'
                }).reset_index()
                transition_counts.columns = ['source', 'target', 'count', 'median_pl']
                
                # Create unique labels for nodes
                all_stages = list(set(transition_counts['source'].unique()) | set(transition_counts['target'].unique()))
                all_stages_sorted = sorted(all_stages)
                
                # Create node labels with "Prior: " prefix for sources and "Current: " for targets
                node_labels = []
                node_dict = {}
                
                # Add prior stages
                for stage in all_stages_sorted:
                    label = f"Prior: {stage}"
                    node_dict[('prior', stage)] = len(node_labels)
                    node_labels.append(label)
                
                # Add current stages
                for stage in all_stages_sorted:
                    label = f"Current: {stage}"
                    node_dict[('current', stage)] = len(node_labels)
                    node_labels.append(label)
                
                # Prepare source, target, and value lists
                sources = []
                targets = []
                values = []
                link_colors = []
                
                for _, row in transition_counts.iterrows():
                    source_idx = node_dict[('prior', row['source'])]
                    target_idx = node_dict[('current', row['target'])]
                    sources.append(source_idx)
                    targets.append(target_idx)
                    values.append(row['count'])
                    
                    # Color based on median P/L
                    if row['median_pl'] < 0:
                        link_colors.append('rgba(239, 85, 59, 0.3)')  # Red with transparency
                    else:
                        link_colors.append('rgba(0, 204, 150, 0.3)')  # Green with transparency
                
                # Create Sankey diagram
                fig_sankey = go.Figure(data=[go.Sankey(
                    node=dict(
                        pad=15,
                        thickness=20,
                        line=dict(color="black", width=0.5),
                        label=node_labels,
                        color='lightblue'
                    ),
                    link=dict(
                        source=sources,
                        target=targets,
                        value=values,
                        color=link_colors
                    )
                )])
                
                fig_sankey.update_layout(
                    title="Stock Stage Cycle Transitions (colored by median cur_hl_pl)",
                    font=dict(size=10),
                    height=600
                )
                
                st.plotly_chart(fig_sankey, width='stretch')
                st.caption("🔴 Red flows: median cur_hl_pl < 0 | 🟢 Green flows: median cur_hl_pl ≥ 0")
                
                # Show transition summary table
                with st.expander("📊 View Transition Details"):
                    transition_summary = transition_counts.copy()
                    transition_summary['median_pl'] = transition_summary['median_pl'].round(2)
                    transition_summary = transition_summary.sort_values('count', ascending=False)
                    st.dataframe(
                        transition_summary,
                        height=400,
                        hide_index=True
                    )
                
                # View stocks in specific transition
                st.markdown("#### 🔍 View Stocks in Specific Transition")
                
                # Create transition options
                transition_options = [f"{row['source']} → {row['target']} ({int(row['count'])} stocks)" 
                                     for _, row in transition_counts.sort_values('count', ascending=False).iterrows()]
                
                selected_transition = st.selectbox(
                    "Select a transition to view stocks:",
                    options=["Select a transition..."] + transition_options,
                    key="transition_selector"
                )
                
                if selected_transition != "Select a transition...":
                    # Parse the selected transition
                    parts = selected_transition.split(" → ")
                    prior_stage = parts[0]
                    current_stage = parts[1].split(" (")[0]
                    
                    # Filter stocks for this transition
                    transition_stocks = df_dist[
                        (df_dist['prior_stage_cycle'] == prior_stage) & 
                        (df_dist['stage_cycle'] == current_stage)
                    ].copy()
                    
                    # Display columns to show
                    display_cols = ['symbol', 'company', 'cur_hl_pl', 'prior_stage_cycle', 'stage_cycle']
                    
                    # Add optional columns if they exist
                    optional_cols = ['rsi', 'sincedays', 'kish_score', 'date1']
                    for col in optional_cols:
                        if col in transition_stocks.columns:
                            display_cols.append(col)
                    
                    # Filter to only existing columns
                    display_cols = [col for col in display_cols if col in transition_stocks.columns]
                    
                    # Sort by cur_hl_pl descending
                    transition_stocks = transition_stocks[display_cols].sort_values('cur_hl_pl', ascending=False)
                    
                    # Count distinct symbols
                    distinct_symbol_count = transition_stocks['symbol'].nunique() if 'symbol' in transition_stocks.columns else len(transition_stocks)
                    st.write(f"**{distinct_symbol_count} stocks** transitioned from **{prior_stage}** to **{current_stage}**")
                    
                    # Show statistics
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        avg_pl = transition_stocks['cur_hl_pl'].mean()
                        st.metric("Average P/L", f"{avg_pl:.2f}%")
                    with stats_col2:
                        positive_count = (transition_stocks['cur_hl_pl'] > 0).sum()
                        positive_pct = (positive_count / distinct_symbol_count) * 100 if distinct_symbol_count > 0 else 0
                        st.metric("Bullish Stocks", f"{positive_count} ({positive_pct:.1f}%)")
                    with stats_col3:
                        negative_count = (transition_stocks['cur_hl_pl'] < 0).sum()
                        negative_pct = (negative_count / distinct_symbol_count) * 100 if distinct_symbol_count > 0 else 0
                        st.metric("Bearish Stocks", f"{negative_count} ({negative_pct:.1f}%)")
                    
                    # Display the stocks table
                    st.dataframe(
                        transition_stocks,
                        height=400,
                        hide_index=True
                    )
                    
                    # Download button
                    transition_csv = transition_stocks.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Stocks as CSV",
                        data=transition_csv,
                        file_name=f"transition_{prior_stage}_to_{current_stage}.csv",
                        mime="text/csv"
                    )
            else:
                st.info("No transition data available for the selected filters.")
        
        st.markdown("---")
        
        # Summary statistics table for all numerical columns
        st.markdown("### 📋 Complete Statistical Summary")
        
        show_full_stats = st.checkbox("Filter numerical columns", value=False)
        
        if show_full_stats:
            stats_df = df_dist[numerical_cols].describe().T
            stats_df['range'] = stats_df['max'] - stats_df['min']
            stats_df['variance'] = df_dist[numerical_cols].var()
            stats_df['skewness'] = df_dist[numerical_cols].skew()
            stats_df['kurtosis'] = df_dist[numerical_cols].kurtosis()
            
            st.dataframe(
                stats_df.style.format("{:.4f}"),
                height=600
            )
            
            # Download button for statistics
            stats_csv = stats_df.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Download Statistics as CSV",
                data=stats_csv,
                file_name="distribution_statistics.csv",
                mime="text/csv"
            )
    else:
        st.warning("No numerical columns found in the dataset.")

# Tab 5: AI-Powered Search
with tab5:
    # Ensure session state is initialized for messages
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            ChatMessage(role="assistant", content="I am an Stock SME, I can help with Stock questions with the help of tools.")
        ]
    
    ai_search_tab(st, st.session_state)
