import pandas as pd
import requests
import streamlit as st
import time
from datetime import datetime, timedelta
import logging

TTL_CACHE = 3600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalNHLCollector:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    @st.cache_data(ttl=TTL_CACHE, show_spinner=False)
    def get_espn_scoreboard(_self, season=2025, date=None):
        """Get game scores from ESPN API - CACHED
        Args:
            season: Season year
            date: Date in YYYYMMDD format (e.g., 20241115) or None for recent games
        
        """
        url = f"https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard"
        if date:
            params = {'dates': date}
        else:
            params = {'limit': 1000}
        
        try:
            response = _self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for event in data.get('events', []):
                competition = event.get('competitions', [{}])[0]
                competitors = competition.get('competitors', [])
                
                if len(competitors) != 2:
                    continue
                
                home_team = next((c for c in competitors if c.get('homeAway') == 'home'), None)
                away_team = next((c for c in competitors if c.get('homeAway') == 'away'), None)
                
                if not home_team or not away_team:
                    continue
                
                game = {
                    'season': season,
                    'home_team': home_team.get('team', {}).get('abbreviation'),
                    'home_score': int(home_team.get('score', 0)),
                    'away_team': away_team.get('team', {}).get('abbreviation'),
                    'away_score': int(away_team.get('score', 0)),
                    'status': event.get('status', {}).get('type', {}).get('name'),
                }
                
                if game['status'] == 'STATUS_FINAL':
                    game['home_wins'] = 1 if game['home_score'] > game['away_score'] else 0
                    games.append(game)
            
            return pd.DataFrame(games)
            
        except Exception as e:
            logger.error(f"Error fetching scoreboard: {e}")
            return pd.DataFrame()
    
    def get_season_schedule(self, season=2025, weeks=None):
        """Get full season schedule"""
        all_games = []
        
        if season == 2025:
            start_date = datetime(2024, 10, 10)
            end_date = datetime(2025, 4, 14)
        else:
            start_date = datetime(season - 1, 10, 1)
            end_date = datetime(season, 4, 30)
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            games = self.get_espn_scoreboard(season, date=date_str)
            if not games.empty:
                all_games.append(games)
            current_date += timedelta(days=1)
            time.sleep(0.1)
        
        if all_games:
            # combine fetched game frames, drop duplicates, log and return
            df = pd.concat(all_games, ignore_index=True)
            df = df.drop_duplicates(subset=['home_team', 'away_team', 'home_score', 'away_score'])
            logger.info(f"Fetched {len(df)} unique games for season {season}")
            return df
        return pd.DataFrame()
    
    def calculate_team_stats(self, schedule_df):
        if schedule_df.empty:
            return pd.DataFrame()
        
        teams = []
        team_abbrs = pd.concat([schedule_df['home_team'], schedule_df['away_team']]).unique()
        
        for team in team_abbrs:
            home_games = schedule_df[schedule_df['home_team'] == team]
            away_games = schedule_df[schedule_df['away_team'] == team]
            
            home_wins = home_games['home_wins'].sum()
            away_wins = len(away_games) - away_games['home_wins'].sum()
            total_wins = home_wins + away_wins
            total_games = len(home_games) + len(away_games)
            
            points_for = home_games['home_score'].sum() + away_games['away_score'].sum()
            points_against = home_games['away_score'].sum() + away_games['home_score'].sum()
            
            teams.append({
                'team': team,
                'games_played': total_games,
                'wins': total_wins,
                'losses': total_games - total_wins,
                'win_rate': total_wins / total_games if total_games > 0 else 0,
                'points_per_game': points_for / total_games if total_games > 0 else 0,
                'points_allowed_per_game': points_against / total_games if total_games > 0 else 0,
                'point_diff': points_for - points_against,
                'avg_margin': (points_for - points_against) / total_games if total_games > 0 else 0
            })
        
        return pd.DataFrame(teams)