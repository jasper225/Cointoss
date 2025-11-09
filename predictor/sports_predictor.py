import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from collectors import MinimalNFLCollector, MinimalNBACollector, MinimalMLBCollector, MinimalNHLCollector

class SportsPredictor:
    """Universal sports game prediction model"""
    
    def __init__(self, league):
        self.league = league
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None

        if league == "NFL":
            self.collector = MinimalNFLCollector()
        elif league == "NBA":
            self.collector = MinimalNBACollector()
        elif league == "MLB":
            self.collector = MinimalMLBCollector()
        elif league == "NHL":
            self.collector = MinimalNHLCollector()
    
    def collect_and_prepare_data(self, season=2024, weeks=None):
        """Collect real sports data and prepare for training"""
        if self.league == "NFL" and weeks is None:
            weeks = range(1, 11)
        
        if self.league == "NFL":
            schedule = self.collector.get_season_schedule(season, weeks)
        else:
            schedule = self.collector.get_season_schedule(season)
            
        team_stats = self.collector.calculate_team_stats(schedule)
        
        if schedule.empty or team_stats.empty:
            return pd.DataFrame()
        
        # Merge team stats with games
        training = schedule.merge(
            team_stats.add_suffix('_home'),
            left_on='home_team',
            right_on='team_home',
            how='left'
        )
        
        training = training.merge(
            team_stats.add_suffix('_away'),
            left_on='away_team',
            right_on='team_away',
            how='left'
        )
        
        # Create features
        training['win_rate_diff'] = training['win_rate_home'] - training['win_rate_away']
        training['ppg_diff'] = training['points_per_game_home'] - training['points_per_game_away']
        training['papg_diff'] = training['points_allowed_per_game_away'] - training['points_allowed_per_game_home']
        training['margin_diff'] = training['avg_margin_home'] - training['avg_margin_away']
        
        return training.dropna()
    
    def train_model(self, df):
        """Train prediction model"""
        feature_cols = [
            'win_rate_home', 'win_rate_away', 'win_rate_diff',
            'points_per_game_home', 'points_per_game_away', 'ppg_diff',
            'points_allowed_per_game_home', 'points_allowed_per_game_away', 'papg_diff',
            'avg_margin_home', 'avg_margin_away', 'margin_diff'
        ]
        
        X = df[feature_cols]
        y = df['home_wins']
        
        self.feature_columns = feature_cols
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return accuracy, feature_importance
    
    def predict_game(self, home_stats, away_stats):
        """Predict game outcome"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create feature vector
        features = {
            'win_rate_home': home_stats['win_rate'],
            'win_rate_away': away_stats['win_rate'],
            'win_rate_diff': home_stats['win_rate'] - away_stats['win_rate'],
            'points_per_game_home': home_stats['ppg'],
            'points_per_game_away': away_stats['ppg'],
            'ppg_diff': home_stats['ppg'] - away_stats['ppg'],
            'points_allowed_per_game_home': home_stats['papg'],
            'points_allowed_per_game_away': away_stats['papg'],
            'papg_diff': away_stats['papg'] - home_stats['papg'],
            'avg_margin_home': home_stats['margin'],
            'avg_margin_away': away_stats['margin'],
            'margin_diff': home_stats['margin'] - away_stats['margin']
        }
        
        X = pd.DataFrame([features])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        return {
            'predicted_winner': 'Home' if prediction == 1 else 'Away',
            'home_win_probability': probability[1],
            'away_win_probability': probability[0],
            'confidence': max(probability)
        }