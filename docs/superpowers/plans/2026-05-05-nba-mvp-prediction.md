# NBA MVP Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure `NBA_MVP.ipynb` to predict MVP vote share (regression) using XGBoost and MLP side-by-side, with leave-one-season-out validation and a standalone `report.html`.

**Architecture:** Single notebook restructured into 7 labeled sections. Both models train on the same 13-feature set; MLP gets StandardScaler preprocessing. Final cell writes a self-contained `report.html` using Plotly charts embedded in a Jinja2 HTML template.

**Tech Stack:** Python 3.9, XGBoost (XGBRegressor), scikit-learn (MLPRegressor, StandardScaler, RandomizedSearchCV), scipy (spearmanr), Plotly, Jinja2, pandas, numpy

---

## File Map

| File | Action |
|------|--------|
| `NBA_MVP.ipynb` | Restructure all cells — replace model cells, update feature engineering, add validation + report sections |
| `report.html` | Generated output — created by final notebook cell |

---

### Task 1: Install missing dependencies

**Files:**
- Run in terminal (not notebook)

- [ ] **Step 1: Install libomp (required for XGBoost on macOS)**

```bash
brew install libomp
```

Expected: `==> Installing libomp` ... `Successfully installed libomp`

- [ ] **Step 2: Install plotly and jinja2**

```bash
pip3 install plotly jinja2
```

Expected: `Successfully installed plotly-... jinja2-...`

- [ ] **Step 3: Verify all imports work**

```bash
python3 -c "import xgboost, plotly, jinja2, scipy; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
cd /Users/ericpark/Projects/NBA_MVP_Prediction
git add -A
git commit -m "chore: install plotly, jinja2, libomp for xgboost"
```

---

### Task 2: Section 1 — Imports & Config

**Files:**
- Modify: `NBA_MVP.ipynb` — replace the existing imports cell

- [ ] **Step 1: Replace the imports cell with the following**

Clear the existing first cell and replace its contents with:

```python
# ===== SECTION 1: IMPORTS & CONFIG =====
import numpy as np
import pandas as pd
from pathlib import Path

# Models
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Stats
from scipy.stats import spearmanr

# Visualization & Report
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template

# ── Config ──────────────────────────────────────────────────────────────────
DATA_DIR = Path('/Users/ericpark/Projects/NBA_MVP_Prediction')
TRAIN_SEASONS = list(range(2010, 2025))
PREDICT_SEASON = 2025

FEATURE_COLS = [
    'trb_per_game', 'ast_per_game', 'pts_per_game',
    'ts_percent', 'usg_percent', 'vorp', 'per',
    'usg_x_win', 'vorp_x_win', 'per_x_win',
    'win_pct', 'playoffs',
    'pts_rank_in_season', 'vorp_rank_in_season', 'per_rank_in_season',
]
```

- [ ] **Step 2: Run the cell and verify no errors**

Expected: cell executes silently (no output).

---

### Task 3: Section 2 — Data Loading

**Files:**
- Modify: `NBA_MVP.ipynb` — replace the data loading cell (currently reads from `/Users/ericpark/NBA_MVP/`)

- [ ] **Step 1: Replace the data loading cell**

```python
# ===== SECTION 2: DATA LOADING =====
advanced = pd.read_csv(DATA_DIR / 'Advanced.csv')
advanced_df = advanced[advanced['season'] >= 2010].copy()

awards = pd.read_csv(DATA_DIR / 'Player_Award_Shares.csv')
awards_df = awards[awards['season'] >= 2010].copy()

per_game = pd.read_csv(DATA_DIR / 'Player_Per_Game.csv')
per_game_df = per_game[per_game['season'] >= 2010].copy()

team_stats = pd.read_csv(DATA_DIR / 'Team_Summaries.csv')
team_stats_df = (
    team_stats[team_stats['season'] >= 2010]
    [['season', 'abbreviation', 'playoffs', 'w', 'l']]
    .dropna()
    .copy()
)
team_stats_df['win_pct'] = team_stats_df['w'] / (team_stats_df['w'] + team_stats_df['l'])

mvp_df = (
    awards_df[awards_df['award'] == 'nba mvp']
    [['season', 'player_id', 'share', 'winner']]
    .copy()
)
mvp_df['mvp_rank'] = mvp_df.groupby('season')['share'].rank(method='max', ascending=False)
mvp_df['winner_int'] = mvp_df['winner'].astype(int)

print(f"Loaded {len(per_game_df)} player-seasons, {len(mvp_df)} MVP vote rows")
```

- [ ] **Step 2: Run cell and verify**

Expected output: `Loaded XXXX player-seasons, XXX MVP vote rows` (no errors, numbers > 0).

---

### Task 4: Section 3 — Feature Engineering

**Files:**
- Modify: `NBA_MVP.ipynb` — replace all existing feature engineering cells with one clean section

- [ ] **Step 1: Replace feature engineering cells**

```python
# ===== SECTION 3: FEATURE ENGINEERING =====

# ── Merge per-game + advanced stats ─────────────────────────────────────────
dup_cols = ['lg', 'player', 'age', 'pos', 'g', 'gs']
full_stats = per_game_df.merge(
    advanced_df.drop(columns=dup_cols),
    how='inner', on=['season', 'player_id', 'team']
)

# ── Merge team win% and playoffs ─────────────────────────────────────────────
full_stats = full_stats.merge(
    team_stats_df[['season', 'abbreviation', 'win_pct', 'playoffs']],
    how='left', left_on=['season', 'team'], right_on=['season', 'abbreviation']
)

# ── Traded player win% (games-weighted average across teams) ─────────────────
traded_players = full_stats.merge(
    full_stats.loc[full_stats['team'].isin(['2TM', '3TM']), ['player_id', 'season']],
    on=['player_id', 'season'], how='inner'
)
traded_players['win_pct_rel'] = traded_players['win_pct'] * traded_players['g']

traded_group = (
    traded_players.loc[~traded_players['team'].isin(['2TM', '3TM']),
                       ['season', 'player_id', 'g', 'win_pct_rel']]
    .groupby(['player_id', 'season']).sum()
)
traded_winpct = (traded_group['win_pct_rel'] / traded_group['g']).reset_index()
traded_winpct.rename(columns={0: 'win_pct'}, inplace=True)

full_stats_idx = full_stats.set_index(['player_id', 'season'])
full_stats_idx.update(traded_winpct.set_index(['player_id', 'season']))
full_stats = full_stats_idx.reset_index()

# ── Quality filter ───────────────────────────────────────────────────────────
full_stats = full_stats[(full_stats['g'] >= 50) & (full_stats['mp_per_game'] >= 25)].copy()

# ── Interaction features ─────────────────────────────────────────────────────
full_stats['usg_x_win']  = full_stats['usg_percent'] * full_stats['win_pct']
full_stats['vorp_x_win'] = full_stats['vorp'] * full_stats['win_pct']
full_stats['per_x_win']  = full_stats['per'] * full_stats['win_pct']

# ── Season-normalized rank features (rank 1 = best in league that season) ────
for stat, col in [('pts_per_game', 'pts_rank_in_season'),
                  ('vorp',         'vorp_rank_in_season'),
                  ('per',          'per_rank_in_season')]:
    full_stats[col] = full_stats.groupby('season')[stat].rank(ascending=False, method='min')

# ── Split predict season from training data ───────────────────────────────────
full_stats_2025 = full_stats[full_stats['season'] == PREDICT_SEASON].copy()
full_stats_rest = full_stats[full_stats['season'] != PREDICT_SEASON].copy()

# ── Initialize target columns, then update from MVP voting data ───────────────
full_stats_rest['share']      = 0.0
full_stats_rest['winner_int'] = 0
full_stats_rest['mvp_rank']   = np.nan

full_stats_rest_idx = full_stats_rest.set_index(['season', 'player_id'])
mvp_df_idx = mvp_df.set_index(['season', 'player_id'])
full_stats_rest_idx.update(mvp_df_idx[['share', 'winner_int', 'mvp_rank']])
full_stats_train = full_stats_rest_idx.reset_index()

# Keep aggregate row for traded players (2TM/3TM), drop individual team rows
full_stats_train.sort_values('team', key=lambda c: c != '2TM', inplace=True)
full_stats_train.drop_duplicates(subset=['season', 'player'], keep='first', inplace=True)

# Fill any remaining NaN in feature cols with 0 (e.g. playoffs for edge cases)
full_stats_train[FEATURE_COLS] = full_stats_train[FEATURE_COLS].fillna(0)
full_stats_2025[FEATURE_COLS]  = full_stats_2025[FEATURE_COLS].fillna(0)

print(f"Training rows: {len(full_stats_train)} | 2025 rows: {len(full_stats_2025)}")
print(f"Features: {FEATURE_COLS}")
```

- [ ] **Step 2: Run cell and verify**

Expected: `Training rows: ~2XXX | 2025 rows: ~XXX` — no KeyErrors, no NaN warnings.

- [ ] **Step 3: Spot-check feature values for a known MVP**

Add a temporary print, run, then delete it:

```python
jokic_2024 = full_stats_train[
    (full_stats_train['player'].str.contains('Joki')) &
    (full_stats_train['season'] == 2024)
][['player', 'share', 'vorp_rank_in_season', 'win_pct', 'playoffs']]
print(jokic_2024)
```

Expected: Jokić 2024 shows `share ≈ 0.935`, `playoffs = 1.0`, `vorp_rank_in_season = 1 or 2`.

- [ ] **Step 4: Commit**

```bash
git add NBA_MVP.ipynb
git commit -m "feat: restructure notebook sections 1-3, add rank features and fix paths"
```

---

### Task 5: Section 4 — XGBoost Regressor

**Files:**
- Modify: `NBA_MVP.ipynb` — replace old classifier cells with a regressor section

- [ ] **Step 1: Add XGBoost Regressor cell**

```python
# ===== SECTION 4: XGBOOST REGRESSOR =====
X_train = full_stats_train[FEATURE_COLS]
y_train = full_stats_train['share']

param_grid_xgb = {
    'n_estimators':      [100, 200, 300, 500],
    'learning_rate':     [0.01, 0.05, 0.1, 0.2],
    'max_depth':         [3, 4, 5, 6],
    'subsample':         [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree':  [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight':  [1, 3, 5],
    'gamma':             [0, 0.1, 0.2, 0.3],
}

xgb_search = RandomizedSearchCV(
    XGBRegressor(objective='reg:squarederror', eval_metric='rmse', random_state=42),
    param_grid_xgb,
    n_iter=50,
    scoring='neg_root_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_

print("Best XGB params:", xgb_search.best_params_)
print(f"Best CV RMSE: {-xgb_search.best_score_:.4f}")
```

- [ ] **Step 2: Run cell and verify**

Expected: prints best params dict and a RMSE < 0.20.

---

### Task 6: Section 5 — MLP Regressor

**Files:**
- Modify: `NBA_MVP.ipynb` — add MLP section after XGBoost section

- [ ] **Step 1: Add MLP Regressor cell**

```python
# ===== SECTION 5: MLP REGRESSOR =====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

param_grid_mlp = {
    'hidden_layer_sizes': [(64, 32, 16), (128, 64, 32), (32, 16)],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'alpha':              [0.0001, 0.001, 0.01],
}

mlp_search = RandomizedSearchCV(
    MLPRegressor(activation='relu', max_iter=500, random_state=42),
    param_grid_mlp,
    n_iter=20,
    scoring='neg_root_mean_squared_error',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
mlp_search.fit(X_train_scaled, y_train)
best_mlp = mlp_search.best_estimator_

print("Best MLP params:", mlp_search.best_params_)
print(f"Best CV RMSE: {-mlp_search.best_score_:.4f}")
```

- [ ] **Step 2: Run cell and verify**

Expected: prints best params and RMSE. MLP RMSE will likely be higher than XGBoost for tabular data — that is expected and fine.

- [ ] **Step 3: Commit**

```bash
git add NBA_MVP.ipynb
git commit -m "feat: add XGBoost regressor and MLP regressor sections"
```

---

### Task 7: Section 6 — Leave-One-Season-Out Validation

**Files:**
- Modify: `NBA_MVP.ipynb` — add validation section

- [ ] **Step 1: Add LOSO validation cell**

```python
# ===== SECTION 6: LEAVE-ONE-SEASON-OUT VALIDATION =====
seasons = sorted(full_stats_train['season'].unique())
loso_results = []

for season in seasons:
    tr = full_stats_train[full_stats_train['season'] != season]
    te = full_stats_train[full_stats_train['season'] == season].copy()

    X_tr, y_tr = tr[FEATURE_COLS], tr['share']
    X_te, y_te = te[FEATURE_COLS], te['share']

    # ── XGBoost ──────────────────────────────────────────────────────────────
    xgb_cv = XGBRegressor(**best_xgb.get_params())
    xgb_cv.fit(X_tr, y_tr)
    te['xgb_pred'] = xgb_cv.predict(X_te).clip(min=0)

    # ── MLP ──────────────────────────────────────────────────────────────────
    sc_cv = StandardScaler()
    mlp_cv = MLPRegressor(**best_mlp.get_params())
    mlp_cv.fit(sc_cv.fit_transform(X_tr), y_tr)
    te['mlp_pred'] = mlp_cv.predict(sc_cv.transform(X_te)).clip(min=0)

    # ── Metrics ───────────────────────────────────────────────────────────────
    voted = te[te['share'] > 0]
    xgb_r, _ = spearmanr(voted['share'], voted['xgb_pred']) if len(voted) > 1 else (np.nan, None)
    mlp_r, _ = spearmanr(voted['share'], voted['mlp_pred']) if len(voted) > 1 else (np.nan, None)

    actual_top3 = set(te.nlargest(3, 'share')['player'])
    xgb_top3   = set(te.nlargest(3, 'xgb_pred')['player'])
    mlp_top3   = set(te.nlargest(3, 'mlp_pred')['player'])

    actual_winner = te.nlargest(1, 'share')['player'].values[0]

    loso_results.append({
        'season':              season,
        'xgb_spearman':        xgb_r,
        'mlp_spearman':        mlp_r,
        'xgb_top3_overlap':    len(actual_top3 & xgb_top3),
        'mlp_top3_overlap':    len(actual_top3 & mlp_top3),
        'xgb_winner_correct':  int(actual_winner == te.nlargest(1, 'xgb_pred')['player'].values[0]),
        'mlp_winner_correct':  int(actual_winner == te.nlargest(1, 'mlp_pred')['player'].values[0]),
        'xgb_rmse':            np.sqrt(mean_squared_error(y_te, te['xgb_pred'])),
        'mlp_rmse':            np.sqrt(mean_squared_error(y_te, te['mlp_pred'])),
    })

loso_df = pd.DataFrame(loso_results)

print(loso_df[['season', 'xgb_spearman', 'mlp_spearman',
               'xgb_top3_overlap', 'mlp_top3_overlap',
               'xgb_winner_correct', 'mlp_winner_correct']].to_string(index=False))

print(f"\n── Summary ──────────────────────────────────────────────────")
print(f"XGB  Spearman (mean): {loso_df['xgb_spearman'].mean():.3f}")
print(f"MLP  Spearman (mean): {loso_df['mlp_spearman'].mean():.3f}")
print(f"XGB  Top-3 overlap:   {loso_df['xgb_top3_overlap'].mean():.2f}/3")
print(f"MLP  Top-3 overlap:   {loso_df['mlp_top3_overlap'].mean():.2f}/3")
print(f"XGB  Winner accuracy: {loso_df['xgb_winner_correct'].mean():.1%}")
print(f"MLP  Winner accuracy: {loso_df['mlp_winner_correct'].mean():.1%}")
```

- [ ] **Step 2: Run cell and verify**

Expected:
- Both models show mean Spearman > 0.7
- Top-3 overlap average >= 2.0/3
- Table prints 15 rows (2010–2024), no NaN errors

- [ ] **Step 3: Commit**

```bash
git add NBA_MVP.ipynb
git commit -m "feat: add leave-one-season-out validation with spearman + top-3 metrics"
```

---

### Task 8: Section 7 — 2025 Predictions

**Files:**
- Modify: `NBA_MVP.ipynb` — add predictions section

- [ ] **Step 1: Add predictions cell**

```python
# ===== SECTION 7: 2025 PREDICTIONS =====

# ── Train final models on full 2010-2024 data ─────────────────────────────
final_xgb = XGBRegressor(**best_xgb.get_params())
final_xgb.fit(X_train, y_train)

final_scaler = StandardScaler()
X_train_scaled_final = final_scaler.fit_transform(X_train)
final_mlp = MLPRegressor(**best_mlp.get_params())
final_mlp.fit(X_train_scaled_final, y_train)

# ── Predict 2025 ──────────────────────────────────────────────────────────
X_2025 = full_stats_2025[FEATURE_COLS]
full_stats_2025['xgb_share'] = final_xgb.predict(X_2025).clip(min=0)
full_stats_2025['mlp_share'] = final_mlp.predict(final_scaler.transform(X_2025)).clip(min=0)

# ── Display top 10 ────────────────────────────────────────────────────────
display_cols = ['player', 'team', 'win_pct', 'playoffs', 'xgb_share', 'mlp_share']
top10 = (
    full_stats_2025
    .sort_values('xgb_share', ascending=False)
    .head(10)[display_cols]
    .copy()
)
top10['win_pct']   = top10['win_pct'].apply(lambda x: f"{x*100:.1f}%")
top10['playoffs']  = top10['playoffs'].apply(lambda x: 'Yes' if x == 1 else 'No' if x == 0 else 'N/A')
top10['xgb_share'] = top10['xgb_share'].apply(lambda x: f"{x:.3f}")
top10['mlp_share'] = top10['mlp_share'].apply(lambda x: f"{x:.3f}")
print(top10.to_string(index=False))
```

- [ ] **Step 2: Run cell and verify**

Expected: top row is Jokić or SGA, both `xgb_share` and `mlp_share` values are between 0 and 1, no NaN.

---

### Task 9: Section 8 — HTML Report Generation

**Files:**
- Modify: `NBA_MVP.ipynb` — add report generation as the final section
- Create: `report.html` (generated output)

- [ ] **Step 1: Add feature importance chart helper cell**

```python
# ===== SECTION 8: HTML REPORT =====

# ── Chart 1: 2025 MVP Race ────────────────────────────────────────────────
top5_2025 = full_stats_2025.nlargest(5, 'xgb_share').sort_values('xgb_share')

fig_race = go.Figure()
fig_race.add_trace(go.Bar(
    y=top5_2025['player'], x=top5_2025['xgb_share'],
    name='XGBoost', orientation='h', marker_color='#1f77b4'
))
fig_race.add_trace(go.Bar(
    y=top5_2025['player'], x=top5_2025['mlp_share'],
    name='Neural Net', orientation='h', marker_color='#ff7f0e'
))
fig_race.update_layout(
    title='2025 MVP Race — Predicted Vote Share',
    barmode='group', xaxis_title='Predicted Vote Share',
    height=400, template='plotly_white'
)

# ── Chart 2: Model Comparison by Season ──────────────────────────────────
fig_compare = go.Figure()
fig_compare.add_trace(go.Scatter(
    x=loso_df['season'], y=loso_df['xgb_spearman'],
    mode='lines+markers', name='XGBoost', marker_color='#1f77b4'
))
fig_compare.add_trace(go.Scatter(
    x=loso_df['season'], y=loso_df['mlp_spearman'],
    mode='lines+markers', name='Neural Net', marker_color='#ff7f0e'
))
fig_compare.update_layout(
    title='Spearman Rank Correlation by Season (LOSO)',
    xaxis_title='Season', yaxis_title='Spearman r',
    height=350, template='plotly_white'
)

# ── Chart 3: Feature Importance (XGBoost) ────────────────────────────────
fi = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': final_xgb.feature_importances_
}).sort_values('importance')

fig_fi = go.Figure(go.Bar(
    x=fi['importance'], y=fi['feature'],
    orientation='h', marker_color='#2ca02c'
))
fig_fi.update_layout(
    title='XGBoost Feature Importance',
    xaxis_title='Importance', height=400, template='plotly_white'
)

# ── Convert charts to HTML divs ───────────────────────────────────────────
chart_race    = pio.to_html(fig_race,    full_html=False, include_plotlyjs=False)
chart_compare = pio.to_html(fig_compare, full_html=False, include_plotlyjs=False)
chart_fi      = pio.to_html(fig_fi,      full_html=False, include_plotlyjs=False)
```

- [ ] **Step 2: Run cell and verify** — no errors, three `chart_*` variables exist.

- [ ] **Step 3: Add historical validation table + report assembly cell**

```python
# ── Historical validation table ───────────────────────────────────────────
hist_rows = ""
for _, row in loso_df.iterrows():
    xgb_ok = "✅" if row['xgb_winner_correct'] else "❌"
    mlp_ok = "✅" if row['mlp_winner_correct'] else "❌"
    hist_rows += (
        f"<tr><td>{int(row['season'])}</td>"
        f"<td>{row['xgb_spearman']:.2f}</td><td>{row['xgb_top3_overlap']:.0f}/3</td><td>{xgb_ok}</td>"
        f"<td>{row['mlp_spearman']:.2f}</td><td>{row['mlp_top3_overlap']:.0f}/3</td><td>{mlp_ok}</td></tr>\n"
    )

summary_row = (
    f"<tr style='font-weight:bold;background:#f0f0f0'>"
    f"<td>Average</td>"
    f"<td>{loso_df['xgb_spearman'].mean():.2f}</td>"
    f"<td>{loso_df['xgb_top3_overlap'].mean():.2f}/3</td>"
    f"<td>{loso_df['xgb_winner_correct'].mean():.0%}</td>"
    f"<td>{loso_df['mlp_spearman'].mean():.2f}</td>"
    f"<td>{loso_df['mlp_top3_overlap'].mean():.2f}/3</td>"
    f"<td>{loso_df['mlp_winner_correct'].mean():.0%}</td>"
    f"</tr>"
)

# ── 2025 top-3 summary rows ────────────────────────────────────────────────
top3_rows = ""
medals = ["🥇", "🥈", "🥉"]
for i, (_, row) in enumerate(full_stats_2025.nlargest(3, 'xgb_share').iterrows()):
    top3_rows += (
        f"<tr><td>{medals[i]}</td><td>{row['player']}</td><td>{row['team']}</td>"
        f"<td>{row['win_pct']*100:.1f}%</td>"
        f"<td>{row['xgb_share']:.3f}</td><td>{row['mlp_share']:.3f}</td></tr>\n"
    )

# ── Assemble HTML ─────────────────────────────────────────────────────────
html_template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>NBA MVP Prediction 2025</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body { font-family: -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #222; }
  h1   { font-size: 2rem; border-bottom: 3px solid #1f77b4; padding-bottom: 8px; }
  h2   { font-size: 1.3rem; margin-top: 40px; color: #1f77b4; }
  table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 0.9rem; }
  th, td { padding: 8px 12px; border: 1px solid #ddd; text-align: center; }
  th { background: #1f77b4; color: white; }
  tr:nth-child(even) { background: #f9f9f9; }
</style>
</head>
<body>
<h1>🏀 NBA MVP Prediction — 2025 Season</h1>

<h2>2025 MVP Race</h2>
{{ chart_race }}
<table>
  <tr><th>Rank</th><th>Player</th><th>Team</th><th>Win%</th><th>XGBoost Share</th><th>Neural Net Share</th></tr>
  {{ top3_rows }}
</table>

<h2>Model Comparison</h2>
{{ chart_compare }}

<h2>Feature Importance (XGBoost)</h2>
{{ chart_fi }}

<h2>Historical Validation (2010–2024, Leave-One-Season-Out)</h2>
<table>
  <tr>
    <th rowspan="2">Season</th>
    <th colspan="3">XGBoost</th>
    <th colspan="3">Neural Net</th>
  </tr>
  <tr>
    <th>Spearman r</th><th>Top-3</th><th>Winner</th>
    <th>Spearman r</th><th>Top-3</th><th>Winner</th>
  </tr>
  {{ hist_rows }}
  {{ summary_row }}
</table>

<p style="color:#888;font-size:0.8rem;margin-top:40px">
  Generated {{ date }} · Training data: Basketball-Reference (2010–2024)
</p>
</body>
</html>
""")

html_out = html_template.render(
    chart_race=chart_race,
    chart_compare=chart_compare,
    chart_fi=chart_fi,
    top3_rows=top3_rows,
    hist_rows=hist_rows,
    summary_row=summary_row,
    date='2026-05-05',
)

report_path = DATA_DIR / 'report.html'
report_path.write_text(html_out)
print(f"Report written to {report_path}")
```

- [ ] **Step 4: Run cell and verify**

Expected output: `Report written to /Users/ericpark/Projects/NBA_MVP_Prediction/report.html`

- [ ] **Step 5: Open report in browser and visually verify all 4 sections render**

```bash
open /Users/ericpark/Projects/NBA_MVP_Prediction/report.html
```

Expected: browser opens, all 4 sections visible — bar chart, line chart, feature importance chart, and historical validation table with ✅/❌ columns.

- [ ] **Step 6: Final commit**

```bash
cd /Users/ericpark/Projects/NBA_MVP_Prediction
git add NBA_MVP.ipynb report.html
git commit -m "feat: add LOSO validation, 2025 predictions, and static HTML report"
```

---

## Success Criteria Checklist

- [ ] `python3 -c "import xgboost, plotly, jinja2"` exits 0
- [ ] Notebook runs end-to-end from a clean kernel restart (Kernel → Restart & Run All) with no errors
- [ ] Mean Spearman correlation > 0.7 for both models in LOSO output
- [ ] Mean top-3 overlap >= 2.0/3 for both models
- [ ] `report.html` opens without errors and shows all 4 sections
- [ ] Jokić or SGA appears as XGBoost's predicted 2025 winner
