import streamlit as st
import pandas as pd 
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from predictor import SportsPredictor

# Page configuration
st.set_page_config(
    page_title="Cointoss",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

TTL_CACHE = 3600



# ============================================================================
# SESSION STATE
# ============================================================================

if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_league' not in st.session_state:
    st.session_state.selected_league = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = None
if 'team_stats' not in st.session_state:
    st.session_state.team_stats = None
if 'prediction-history' not in st.session_state:
    st.session_state.prediction_history = []
if 'last_data_fetch' not in st.session_state:
    st.session_state.last_data_fetch = None


# ============================================================================
# HOME PAGE
# ============================================================================

def show_home_page():
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            font-size: 4rem;
            font-weight: bold;
            margin-bottom: 1rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            text-align: center;
            font-size: 1.5rem;
            color: #666;
            margin-bottom: 3rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-title">🪙 Cointoss</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Predict game outcomes using machine learning</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Select League")
    st.markdown("")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("🏈")
        st.markdown("#### **NFL**")
        st.markdown("National Football League")
        st.markdown("✅ **Available**")
        if st.button("Launch NFL Predictor", key="launch_nfl", type="primary", use_container_width=True):
            st.session_state.selected_league = "NFL"
            st.session_state.page = "predictor"
            st.rerun()
    
    with col2:
        st.markdown("🏀")
        st.markdown("#### **NBA**")
        st.markdown("National Basketball Association")
        st.markdown("✅ **Available**")
        if st.button("Launch NBA Predictor", key="launch_nba", type="primary", use_container_width=True):
            st.session_state.selected_league = "NBA"
            st.session_state.page = "predictor"
            st.rerun()
    
    with col3:
        st.markdown("⚾")
        st.markdown("#### **MLB**")
        st.markdown("Major League Baseball")
        st.markdown("✅ **Available**")
        if st.button("Launch MLB Predictor", key="launch_mlb", type="primary", use_container_width=True):
            st.session_state.selected_league = "MLB"
            st.session_state.page = "predictor"
            st.rerun()
    
    with col4:
        st.markdown("🏒")
        st.markdown("#### **NHL**")
        st.markdown("National Hockey League")
        st.markdown("✅ **Available**")
        if st.button("Launch NHL Predictor", key="launch_nhl", type="primary", use_container_width=True):
            st.session_state.selected_league = "NHL"
            st.session_state.page = "predictor"
            st.rerun()
    
    st.markdown("---")

    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            Built with Streamlit • Data from ESPN API
        </div>
    """, unsafe_allow_html=True)


# ============================================================================
# PREDICTOR PAGE (UNIVERSAL FOR ALL LEAGUES)
# ============================================================================

def show_predictor_page():
    """Display the predictor page for selected league"""
    
    league = st.session_state.selected_league
    
    # Header with back button
    col1, col2 = st.columns([1, 6])
    with col1:
        if st.button("← Back", key="back_home"):
            st.session_state.page = 'home'
            st.session_state.model_trained = False
            st.session_state.predictor = None
            st.rerun()
    
    with col2:
        st.title(f"Predict {league} Games")
    
    st.markdown("**Predict game outcomes using real data and machine learning**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Model Settings")

        if st.session_state.last_data_fetch:
            time_diff = datetime.now() - st.session_state.last_data_fetch
            minutes_ago = int(time_diff.total_seconds() // 60)

            if minutes_ago < 60:
                cache_status = f"Last data fetch: {minutes_ago} minutes ago"
            else:
                hours_ago = minutes_ago // 60
                cache_status = f"Last data fetch: {hours_ago} hours ago"
            
            st.caption(cache_status)
        
        if league == "NFL" or league == "MLB":
            season = st.selectbox("Season", [2024, 2023], index=0)
        else:
            season_str = st.selectbox("Season", ["2024-25", "2023-24"], index=0)
            season = 2025 if season_str == "2024-25" else 2024
        
        # Only show week slider for NFL
        if league == "NFL":
            max_week = st.slider("Weeks to Include", 1, 18, 10)
            weeks_param = range(1, max_week + 1)
        else:
            weeks_param = None
        
        if st.button("🔄 Fetch Data & Train Model", type="primary", use_container_width=True):
            with st.spinner(f"Fetching {league} data..."):
                try:
                    predictor = SportsPredictor(league)
                    
                    # Collect data
                    if league == "NFL":
                        df = predictor.collect_and_prepare_data(season=season, weeks=weeks_param)
                    else:
                        df = predictor.collect_and_prepare_data(season=season)
                    
                    if df.empty:
                        st.error("❌ Could not fetch data. Check your internet connection.")
                    else:
                        st.info(f"✓ Collected {len(df)} games")
                        
                        # Train model
                        accuracy, feature_importance = predictor.train_model(df)
                        
                        # Get team stats for predictions
                        team_stats = predictor.collector.calculate_team_stats(df)
                        
                        st.session_state.predictor = predictor
                        st.session_state.model_trained = True
                        st.session_state.accuracy = accuracy
                        st.session_state.team_stats = team_stats
                        st.session_state.feature_importance = feature_importance
                        
                        st.success(f"✓ Model trained! Accuracy: {accuracy:.1%}")
                except Exception as e:
                    st.error(f"❌ An error occurred: {e}")
        
        st.markdown("---")
        if st.session_state.model_trained:
            st.metric("Model Accuracy", f"{st.session_state.accuracy:.1%}")
            st.info("✅ Ready for predictions")
        else:
            st.warning("⚠️ Click above to fetch data and train model")
        if st.button("Clear Cache", type="secondary", use_container_width=True):
            st.session_state.last_data_fetch = None
            st.success("Cache cleared ✅")
    
    # Main content
    if st.session_state.model_trained:
        tab1, tab2, tab3 = st.tabs(["🎯 Make Prediction", "📊 Model Insights", "📝 Prediction History"])
        
        with tab1:
            st.header("Predict Game Outcome")
            
            team_stats = st.session_state.team_stats
            teams = sorted(team_stats['team'].unique())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏠 Home Team")
                home_team = st.selectbox("Select Home Team", teams, key="home")
                
                if home_team:
                    home_data = team_stats[team_stats['team'] == home_team].iloc[0]
                    st.metric("Win Rate", f"{home_data['win_rate']:.1%}")
                    st.metric("Points Per Game", f"{home_data['points_per_game']:.1f}")
                    st.metric("Points Allowed", f"{home_data['points_allowed_per_game']:.1f}")
            
            with col2:
                st.subheader("✈️ Away Team")
                away_team = st.selectbox("Select Away Team", [t for t in teams if t != home_team], key="away")
                
                if away_team:
                    away_data = team_stats[team_stats['team'] == away_team].iloc[0]
                    st.metric("Win Rate", f"{away_data['win_rate']:.1%}")
                    st.metric("Points Per Game", f"{away_data['points_per_game']:.1f}")
                    st.metric("Points Allowed", f"{away_data['points_allowed_per_game']:.1f}")
            
            if home_team and away_team:
                st.markdown("---")
                st.subheader("⚔️Head-to-Head History")

                try:
                    predictor = st.session_state.predictor

                    if league == "NFL":
                        schedule_df = predictor.collector.get_season_schedule(season=season, weeks=range(1, 19))
                    else:
                        schedule_df = predictor.collector.get_season_schedule(season=season)
                    
                    h2h_games = schedule_df[
                        ((schedule_df['home_team'] == home_team) & (schedule_df['away_team'] == away_team)) |
                        ((schedule_df['home_team'] == away_team) & (schedule_df['away_team'] == home_team))
                    ].copy()

                    if not h2h_games.empty:
                        home_h2h_wins = len(h2h_games[
                            ((h2h_games['home_team'] == home_team) & (h2h_games['home_wins'] == 1)) |
                            ((h2h_games['away_team'] == home_team) & (h2h_games['home_wins'] == 0))
                        ])
                        away_h2h_wins = len(h2h_games) - home_h2h_wins

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{home_team} wins", home_h2h_wins)
                        with col2:
                            st.metric(f"{away_team} wins", away_h2h_wins)
                        with col3:
                            st.metric("Total Games", len(h2h_games))
                    else:
                        st.info("No head-to-head games found for {home_team} and {away_team} this season.")

                except Exception as e:
                    st.error(f"❌ An error occured fetching H2H data: {e}")
                    import logging
                    logging.error(f"Error fetching H2H data: {e}")
            
            if st.button("🔮 Predict Game", type="primary", use_container_width=True):
                home_stats = {
                    'win_rate': home_data['win_rate'],
                    'ppg': home_data['points_per_game'],
                    'papg': home_data['points_allowed_per_game'],
                    'margin': home_data['avg_margin']
                }
                
                away_stats = {
                    'win_rate': away_data['win_rate'],
                    'ppg': away_data['points_per_game'],
                    'papg': away_data['points_allowed_per_game'],
                    'margin': away_data['avg_margin']
                }
                
                prediction = st.session_state.predictor.predict_game(home_stats, away_stats)

                # Log prediction to history
                st.session_state.prediction_history.append({
                    'timestamp': datetime.now(),
                    'league': league,
                    'matchup': f"{home_team} vs {away_team}",
                    'prediction': home_team if prediction['predicted_winner'] == 'Home' else away_team,
                    'confidence': prediction['confidence'],
                    'home_prob': prediction['home_win_probability'],
                    'away_prob': prediction['away_win_probability']
                })
                
                st.markdown("---")
                st.header("🎯 Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                winner_team = home_team if prediction['predicted_winner'] == 'Home' else away_team
                st.balloons()
                
                with col1:
                    st.metric("Predicted Winner", winner_team)
                
                with col2:
                    st.metric("Win Probability", f"{prediction['confidence']:.1%}")
                
                with col3:
                    confidence_label = "High" if prediction['confidence'] > 0.7 else "Medium" if prediction['confidence'] > 0.55 else "Low"
                    st.metric("Confidence", confidence_label)
                
                # Probability chart
                fig = go.Figure(data=[
                    go.Bar(
                        x=[home_team, away_team],
                        y=[prediction['home_win_probability'], prediction['away_win_probability']],
                        marker_color=['#013369', '#D50A0A'],
                        text=[f"{prediction['home_win_probability']:.1%}", 
                              f"{prediction['away_win_probability']:.1%}"],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Win Probability",
                    yaxis_title="Probability",
                    yaxis=dict(tickformat='.0%'),
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.header("Model Performance")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Feature Importance")
                fig = px.bar(
                    st.session_state.feature_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Most Important Features"
                )
                fig.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Team Statistics")
                st.dataframe(
                    st.session_state.team_stats.sort_values('win_rate', ascending=False),
                    use_container_width=True,
                    height=400
                )
        
        with tab3:
            st.header("📝 Prediction History")

            if st.session_state.prediction_history:
                history_df = pd.DataFrame(st.session_state.prediction_history)
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                history_df = history_df.sort_values(by='timestamp', ascending=False)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(history_df))
                with col2:
                    avg_confidence = history_df['confidence'].mean()
                    st.metric("Avg. Confidence", f"{avg_confidence: .1%}")
                with col3:
                    league_counts = history_df['league'].value_counts()
                    top_league = league_counts.index[0] if len(league_counts) > 0 else "N/A"
                    st.metric("Most Used", top_league)
                
                st.markdown("---")
                for idx, row in history_df.iterrows():
                    with st.expander(f"{row['matchup']} - {row['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**League:** {row['league']}")
                            st.write(f"**Winner:** {row['prediction']}")
                        with col2:
                            st.write(f"**Confidence:** {row['confidence']:.1%}")
                            st.write(f"**Home Win %:** {row['home_prob']:.1%}")
                        with col3:
                            st.write(f"**Away Win %:** {row['away_prob']:.1%}")
                
                st.markdown("---")

                if st.button("Clear History", type="secondary", use_container_width=True):
                    st.session_state.prediction_history = []
                    st.success("Prediction history cleared ✅")
                    st.rerun()
            else:
                st.info("No predictions made yet. Click 'Make Prediction' to get started!")
    
    else:
        st.info("👈 Click 'Fetch Data & Train Model' in the sidebar to get started!")


# ============================================================================
# MAIN APP ROUTING
# ============================================================================

if st.session_state.page == 'home':
    show_home_page()
elif st.session_state.page == 'predictor':
    show_predictor_page()