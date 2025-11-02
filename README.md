# 🪙 Cointoss

A machine learning-powered web application that predicts sports outcomes using real-time data from ESPN's public API.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.29+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Overview

Cointoss is an interactive web application that uses machine learning to predict the outcomes of sports games. Built with Streamlit, it fetches real-time data, calculates team statistics, trains a Random Forest model, and provides win probability predictions for any team matchup.

## ✨ Features

- **Real-Time Data**: Fetches current season data from ESPN's public API
- **Interactive UI**: User-friendly interface built with Streamlit
- **Machine Learning**: Random Forest classifier trained on historical game data
- **Team Statistics**: View detailed performance metrics for all NFL, NBA, MLB, and NHL teams
- **Win Probability**: Get probability estimates with confidence levels
- **Visual Analytics**: Interactive charts and graphs using Plotly
- **Feature Importance**: Understand which factors drive predictions
- **Lightweight**: Minimal dependencies, no heavy packages required

## 🎬 Demo

### Main Prediction Interface

```
Select teams → View stats → Get prediction → See win probabilities
```

### Key Metrics Displayed

- Win Rate
- Points Per Game (Offense)
- Points Allowed Per Game (Defense)
- Average Margin of Victory
- Feature Importance Rankings

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Internet connection (for fetching data)

### Step 1: Clone the Repository

```bash
git clone https://github.com/jasper225/cointoss.git
cd cointoss
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```txt
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
plotly==5.18.0
requests==2.31.0
```

## 💻 Usage

### Running the Application

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Step-by-Step Guide

1. **Fetch Data**: Click "🔄 Fetch Data & Train Model" in the sidebar
2. **Wait for Training**: The app will fetch real data and train the model (~30-60 seconds)
3. **Select Teams**: Choose home and away teams from the dropdowns
4. **View Stats**: Review current statistics for both teams
5. **Predict**: Click "🔮 Predict Game" to see the prediction
6. **Analyze Results**: View win probabilities and confidence levels

### Configuration Options

- **Season**: Select which season to analyze (default: current)
- **Weeks to Include (NFL)**: Choose how many weeks of data to use (1-18)

## 🔧 How It Works

### 1. Data Collection

The app uses ESPN's public API to fetch:

- Game schedules and results
- Team scores (home and away)
- Game status (completed games only)

```python
# ESPN API endpoints
https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard
https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard
https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard
```

### 2. Feature Engineering

Calculated metrics for each team:

- **Win Rate**: Wins / Total Games
- **Points Per Game**: Average offensive output
- **Points Allowed Per Game**: Average defensive performance
- **Point Differential**: Net scoring advantage
- **Average Margin**: Average winning/losing margin

Derived features for predictions:

- Win Rate Difference (Home - Away)
- PPG Difference
- Defensive Rating Difference
- Margin Difference

### 3. Machine Learning Model

**Algorithm**: Random Forest Classifier

- **Estimators**: 100 trees
- **Train/Test Split**: 80/20
- **Feature Scaling**: StandardScaler normalization
- **Cross-Validation**: Stratified sampling

**Training Features** (12 total):

```python
[
    'win_rate_home', 'win_rate_away', 'win_rate_diff',
    'points_per_game_home', 'points_per_game_away', 'ppg_diff',
    'points_allowed_per_game_home', 'points_allowed_per_game_away', 'papg_diff',
    'avg_margin_home', 'avg_margin_away', 'margin_diff'
]
```

### 4. Prediction Output

For each prediction, the model provides:

- **Binary Prediction**: Home Win (1) or Away Win (0)
- **Probability Estimates**: Confidence for each outcome
- **Confidence Level**: High (>70%), Medium (55-70%), Low (<55%)

## 📁 Project Structure

```
cointoss/
│
├── app.py                      # Main Streamlit application
├── README.md                   # This file
├── .gitignore                  # Files Git is meant to ignor
```

## 🛠 Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| **Python**       | Core programming language      |
| **Streamlit**    | Web application framework      |
| **Pandas**       | Data manipulation and analysis |
| **NumPy**        | Numerical computing            |
| **Scikit-learn** | Machine learning algorithms    |
| **Plotly**       | Interactive visualizations     |
| **Requests**     | HTTP requests to ESPN API      |

## 📊 Data Sources

### ESPN Public API

- **Endpoint**: `https://site.api.espn.com`
- **Access**: Free, no authentication required
- **Rate Limits**: Respectful delays implemented (0.3s between requests)
- **Data Coverage**: Current and previous NFL, NBA, MLB, and NHL seasons

### Data Freshness

- Game results: Updated after each game completion
- Team statistics: Recalculated from latest games
- Model training: On-demand when user clicks "Fetch Data"

## 📈 Model Performance

### Typical Accuracy Metrics

- **Training Accuracy**: 65-75%
- **Test Accuracy**: 60-70%
- **Baseline Accuracy**: ~55% (home team advantage)

### Feature Importance (Typical Rankings)

1. Win Rate Difference (~25%)
2. Points Per Game Difference (~20%)
3. Defensive Rating Difference (~18%)
4. Average Margin Difference (~15%)
5. Other features (~22%)

### Known Limitations

- **Limited Historical Context**: Uses only current season data
- **No Player-Level Data**: Team-level statistics only
- **Home Field Advantage**: Simplified as binary feature
- **External Factors**: Weather, injuries not included
- **Sample Size**: Early season predictions less reliable

## 🚀 Future Enhancements

### Planned Features

- [ ] **Weather Data Integration**: Add temperature, wind, precipitation
- [ ] **Injury Reports**: Track key player injuries
- [ ] **Advanced Metrics**: For more realistic predictions
- [ ] **Historical Trends**: Multi-season training data
- [ ] **Head-to-Head Records**: Recent matchup history
- [ ] **Division/Conference Context**: Rivalry game adjustments
- [ ] **Live Updates**: Real-time score tracking during games
- [ ] **Betting Line Integration**: Compare against Vegas odds
- [ ] **Model Comparison**: Multiple algorithms (XGBoost, Neural Networks)
- [ ] **API Endpoint**: RESTful API for predictions
- [ ] **Mobile App**: React Native companion app
- [ ] **User Accounts**: Save predictions and track accuracy

### Technical Improvements

- [ ] Model persistence (save/load trained models)
- [ ] Database integration (PostgreSQL for data storage)
- [ ] Caching layer (Redis for API responses)
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Unit and integration tests
- [ ] Performance monitoring
- [ ] A/B testing framework

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Jasper Markarian]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

- **ESPN** for providing free public API access
- **Streamlit** team for the amazing framework
- **Scikit-learn** community for ML tools
- **NFL**, **NBA**, **MLB**, and **NHL** for the data and entertainment

**Built by [Jasper Markarian]**
