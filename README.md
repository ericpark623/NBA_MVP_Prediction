# NBA MVP Prediction

Predicts NBA MVP vote share using machine learning — two models (XGBoost and a neural network) trained on per-game and advanced stats from 2010–2024, validated with leave-one-season-out cross-validation.

## 2025 Predictions

| Rank | Player | Team | Win% | XGBoost Share | Neural Net Share |
|------|--------|------|------|---------------|-----------------|
| 🥇 | Shai Gilgeous-Alexander | OKC | 82.9% | 0.952 | 1.142 |
| 🥈 | Nikola Jokić | DEN | 61.0% | 0.797 | 0.883 |
| 🥉 | Giannis Antetokounmpo | MIL | 58.5% | 0.324 | 0.524 |

Full interactive report with charts and historical validation: open `report.html` in a browser.

## Approach

Rather than classifying players as MVP/non-MVP, both models predict continuous **vote share** (0–1), which captures the nuance of dominant vs. close races and avoids class imbalance issues. Models are evaluated using **leave-one-season-out cross-validation** — train on all seasons except one, predict that season, repeat for 2010–2024. This mirrors real-world use where the current season is always unseen.

XGBoost and an MLP regressor are trained on the same feature set for a direct comparison.

## Features

- `pts_per_game`, `trb_per_game`, `ast_per_game` — traditional counting stats
- `ts_percent`, `usg_percent` — efficiency and usage
- `vorp`, `per` — advanced value metrics
- `win_pct`, `playoffs` — team context
- `vorp_x_win`, `per_x_win`, `usg_x_win` — interaction terms
- `pts_rank_in_season`, `vorp_rank_in_season`, `per_rank_in_season` — per-season ranks to correct for scoring-era drift

## Model Performance (2010–2024 validation)

| Model | Avg Spearman r | Avg Top-3 Overlap | Winner Accuracy |
|-------|---------------|-------------------|-----------------|
| XGBoost | 0.79 | 2.40 / 3 | 73% |
| Neural Net | 0.79 | 2.47 / 3 | 80% |

Both models rank the eventual MVP in the top 3 for roughly 4 out of 5 seasons. The neural net edges out XGBoost on winner accuracy.

## Tech Stack

Python · pandas · XGBoost · scikit-learn · Plotly · Jinja2

## Quick Start

```bash
git clone https://github.com/ericpark623/NBA_MVP_Prediction.git
cd NBA_MVP_Prediction
pip install numpy pandas xgboost scikit-learn plotly jinja2 jupyter
jupyter notebook NBA_MVP.ipynb   # Run All → generates report.html
open report.html
```
