# %% [markdown]
# # Energy Consumption Forecasting (Time Series)
# 
# **Forecasting workflow (5 stages):**
# 
# | Stage | Focus | Tasks |
# |-------|--------|-------|
# | **1. Data structure** | Time index, frequency, scale, missingness | Load, index, infer freq, report scale & gaps |
# | **2. Visualization** | Trend, seasonality, breaks, outliers | Plot raw series, trend, seasonal patterns, flag outliers |
# | **3. Decomposition** | Signal vs noise | Additive/multiplicative decomposition (trend, seasonal, residual) |
# | **4. Modeling** | ETS, ARIMA, regression-based | Fit models, produce forecasts |
# | **5. Evaluation & monitoring** | Accuracy, comparison | MAE, RMSE, MAPE; compare models; insights |
# 
# *Models come after understanding the data. Today's focus: Stages 1–3.*

# %% [markdown]
# ### Setup
# Run once: `pip install kagglehub pandas numpy matplotlib statsmodels`  
# **Dataset:** [Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption) (PJM). Place CSVs in `data/` or `~/Downloads/archive/`.

# %% [markdown]
# ## Stage 1 — Data structure
# 
# *(Time index, frequency, scale, missingness)*

# %%
import pandas as pd
import numpy as np
from pathlib import Path

base_dir = Path.cwd()
data_dir = base_dir / 'data'
archive_dir = Path.home() / 'Downloads' / 'archive'
search_dirs = [d for d in [data_dir, archive_dir] if d.exists()]

def load_one_csv(path):
    try:
        raw = pd.read_csv(path)
        if raw.empty or len(raw.columns) < 2:
            return None
        dt_col = next((c for c in raw.columns if 'date' in c.lower() or 'time' in c.lower()), raw.columns[0])
        mw_col = next((c for c in raw.columns if 'mw' in c.lower() or 'power' in c.lower()), raw.columns[1] if len(raw.columns) > 1 else raw.columns[0])
        df = raw[[dt_col, mw_col]].rename(columns={dt_col: 'datetime', mw_col: 'Global_active_power'})
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime', 'Global_active_power'])
        df = df.set_index('datetime').sort_index()
        df_hourly = df.copy()
        df_daily = df.resample('1D').mean().dropna()
        df_yearly = df.resample('1Y').mean().dropna()
        if len(df_daily) < 14:
            return None
        return df, df_hourly, df_daily, df_yearly
    except (ValueError, KeyError, OSError):
        return None

databases = {}
for folder in search_dirs:
    for f in sorted(folder.glob('*.csv')):
        key = f.stem
        if key in databases:
            continue
        out = load_one_csv(f)
        if out is not None:
            df, df_h, df_d, df_y = out
            databases[key] = {'df': df, 'hourly': df_h, 'daily': df_d, 'yearly': df_y}
            print(f"  Loaded: {f.name} -> '{key}'")

series_name = 'PJME_hourly' if 'PJME_hourly' in databases else (list(databases.keys())[0] if databases else None)

if series_name and series_name in databases:
    db = databases[series_name]
    df = db['df']
    df_hourly = db['hourly']
    df_daily = db['daily']
    df_yearly = db['yearly']
    print(f"Using series: {series_name}")
    print(f"Rows: {len(df)}, Hourly: {len(df_hourly)}, Daily: {len(df_daily)}, Yearly: {len(df_yearly)}")
    print(f"Date range: {df_hourly.index[0]} to {df_hourly.index[-1]}")
    print(f"Mean: {df_hourly['Global_active_power'].mean():.3f} kW, Std: {df_hourly['Global_active_power'].std():.3f}")
else:
    print("No CSV found. Using synthetic sample.")
    dates = pd.date_range('2007-01-01', periods=10000, freq='1h')
    np.random.seed(42)
    base = 32000 + 5000 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    df = pd.DataFrame({'Global_active_power': base + np.random.normal(0, 500, len(dates))}, index=dates)
    df.index.name = 'datetime'
    df_hourly = df.copy()
    df_daily = df.resample('1D').mean().dropna()
    df_yearly = df.resample('1Y').mean().dropna()
    databases = {}

print("Available:", list(databases.keys()) if databases else "[]")
df_hourly.head()

# %% [markdown]
# ### Stage 1 (continued) — Structure report
# 
# *(Time index, frequency, scale, missingness)*

# %%
# Stage 1 checklist: time index, frequency, scale, missingness
y = df_hourly['Global_active_power']
print("1. Time index:", y.index.name or "datetime", "| dtype:", y.index.dtype)
print("2. Frequency: inferred", pd.infer_freq(y.index[:100]) or "irregular (check gaps)")
print("3. Scale: mean = {:.2f}, std = {:.2f}, min = {:.2f}, max = {:.2f} (MW)".format(y.mean(), y.std(), y.min(), y.max()))
missing = y.isna().sum()
gaps = y.index.to_series().diff().gt(pd.Timedelta(hours=2)).sum() if hasattr(y.index, 'to_series') else 0
print("4. Missingness: {} NaN(s), ~{} gap(s) > 2h".format(missing, gaps))

# %% [markdown]
# ### Stage 1 — Graph data
# 
# *(By year, month, hour; box plot by month)*

# %%
# Stage 1 — Graph data as-is by year, month, and hour
import matplotlib.pyplot as plt

# 1) By year: mean yearly consumption
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

by_year = df_yearly['Global_active_power']  # already computed in Stage 1 load cell
axes[0].plot(by_year.index, by_year.values, color='steelblue')
axes[0].set_ylabel('Mean consumption (MW)')
axes[0].set_title('Energy Consumption by Year')
axes[0].grid(True, alpha=0.3)

# 2) By month: mean daily consumption by month of year (1–12)
by_month = df_daily['Global_active_power'].groupby(df_daily.index.month).mean()
axes[1].plot(by_month.index, by_month.values, marker='o', color='darkgreen')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Mean daily consumption (MW)')
axes[1].set_title('Energy Consumption by Month (1=Jan .. 12=Dec)')
axes[1].grid(True, alpha=0.3)

# 3) By hour: mean consumption by hour of day (0–23) from hourly data
y_h = df_hourly['Global_active_power']
by_hour = y_h.groupby(y_h.index.hour).mean()
axes[2].plot(by_hour.index, by_hour.values, marker='o', color='coral')
axes[2].set_xlabel('Hour of day')
axes[2].set_ylabel('Mean consumption (MW)')
axes[2].set_title('Energy Consumption by Hour of Day')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 4) Monthly box plot: distribution of daily consumption by month
fig, ax = plt.subplots(figsize=(12, 4))
daily_with_month = df_daily.copy()
daily_with_month['Month'] = daily_with_month.index.month
daily_with_month.boxplot(column='Global_active_power', by='Month', ax=ax)
ax.set_xlabel('Month')
ax.set_ylabel('Daily mean consumption (MW)')
ax.set_title('Energy Consumption by Month (Box Plot)')
plt.suptitle('')
plt.tight_layout()
plt.show()

# Identify "not normal" daily values (box-plot outliers) per month
q1 = daily_with_month.groupby('Month')['Global_active_power'].quantile(0.25)
q3 = daily_with_month.groupby('Month')['Global_active_power'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

bounds = daily_with_month['Month'].map(lower).to_frame('lower').join(
    daily_with_month['Month'].map(upper).to_frame('upper')
)

mask = (daily_with_month['Global_active_power'] < bounds['lower']) | \
       (daily_with_month['Global_active_power'] > bounds['upper'])
outliers = daily_with_month.loc[mask, ['Global_active_power', 'Month']]

print("\nDaily energy values flagged as box-plot outliers (per month):")
print(outliers.head(50))
print(f"Total outliers: {len(outliers)}")

# %% [markdown]
# ## Stage 2 — Visualization
# *(Trend, seasonality, breaks, outliers)*

# %%
%matplotlib inline
import matplotlib.pyplot as plt

y_h = df_hourly['Global_active_power']
y_d = df_daily['Global_active_power']

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)
# Raw series (last 2 weeks hourly)
y_h.tail(24*14).plot(ax=axes[0], title='Raw series (last 2 weeks)', ylabel='MW')
# Trend: rolling mean (daily)
y_d.rolling(30, center=True).mean().plot(ax=axes[1], title='Trend (30-day rolling mean)', ylabel='MW')
# Seasonality: average by hour of day (hourly)
y_h.groupby(y_h.index.hour).mean().plot(ax=axes[2], title='Seasonality (avg by hour of day)', ylabel='MW', xticks=range(0, 24, 2))
plt.tight_layout()
plt.show()

# Outliers: points beyond 3 std from rolling mean
roll = y_h.rolling(24*7, center=True).mean()
resid = (y_h - roll).dropna()
outliers = resid.abs() > (3 * resid.std())
print("Outliers (|resid| > 3*std):", outliers.sum(), "points")

# %% [markdown]
# ## Stage 3 — Decomposition
# 
# *(Separating signal from noise: trend, seasonal, residual)*

# %%
%matplotlib inline
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

y_hr = df_hourly['Global_active_power'].dropna()
y_decomp = y_hr.tail(24 * 7 * 4)
decomp = seasonal_decompose(y_decomp, model='additive', period=24)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomp.observed.plot(ax=axes[0], title='Observed'); axes[0].set_ylabel('kW')
decomp.trend.plot(ax=axes[1], title='Trend'); axes[1].set_ylabel('kW')
decomp.seasonal.plot(ax=axes[2], title='Seasonal (24h)'); axes[2].set_ylabel('kW')
decomp.resid.plot(ax=axes[3], title='Residual'); axes[3].set_ylabel('kW')
plt.tight_layout()
plt.suptitle('Stage 3: Decomposition (additive, period=24h)', y=1.02)
plt.show()

# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

y = df_daily['Global_active_power'].dropna()
train = y.iloc[:-7]
test = y.iloc[-7:]
forecasts = {}

model_arima = ARIMA(train, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0))
fit_arima = model_arima.fit()

print("\n" + "-" * 70)
print("Stage 4: ARIMA model")
print("-" * 70)
print(fit_arima.summary())

forecasts['ARIMA'] = fit_arima.forecast(steps=7)

try:
    model_ets = ExponentialSmoothing(train, seasonal_periods=7, trend='add', seasonal='add')
    fit_ets = model_ets.fit()
    forecasts['ETS'] = fit_ets.forecast(steps=7)
except Exception:
    model_ets = ExponentialSmoothing(train, trend='add')
    fit_ets = model_ets.fit()
    forecasts['ETS'] = fit_ets.forecast(steps=7)

print("\n" + "-" * 70)
print("Stage 4: ETS Model (Holt-Winters)")
print("-" * 70)
print(fit_ets.summary())
print(f"\nETS AIC: {fit_ets.aic:.2f}")

results = pd.DataFrame({'Actual': test.values}, index=test.index)
for name, vals in forecasts.items():
    if vals is not None:
        results[name] = vals
results

# %% [markdown]
# ## Stage 4 — Modeling
# 
# *(ETS, ARIMA, regression-based: fit and forecast)*

# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
import statsmodels.api as sm

y = df_daily['Global_active_power'].dropna()
# Use latest month (30 days) to train, forecast the following month (30 days)
MONTH_DAYS = 30
train = y.iloc[-MONTH_DAYS:]
forecast_steps = MONTH_DAYS
forecast_dates = pd.date_range(train.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecasts = {}
n_train = len(train)
train_idx = train.index
test = None  # no actuals for future month
test_idx = forecast_dates  # use forecast dates for indexing results

model_arima = ARIMA(train, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0))
fit_arima = model_arima.fit()

print("\n" + "-" * 70)
print("Stage 4: ARIMA model")
print("-" * 70)
print(fit_arima.summary())

forecasts['ARIMA'] = fit_arima.forecast(steps=forecast_steps)

try:
    model_ets = ExponentialSmoothing(train, seasonal_periods=7, trend='add', seasonal='add')
    fit_ets = model_ets.fit()
    forecasts['Holt-Winters'] = fit_ets.forecast(steps=forecast_steps)
except Exception:
    model_ets = ExponentialSmoothing(train, trend='add')
    fit_ets = model_ets.fit()
    forecasts['Holt-Winters'] = fit_ets.forecast(steps=forecast_steps)

print("\n" + "-" * 70)
print("Stage 4: ETS Model (Holt-Winters)")
print("-" * 70)
print(fit_ets.summary())
print(f"\nETS AIC: {fit_ets.aic:.2f}")

# Regression (trend + day-of-week dummies)
X_train = sm.add_constant(pd.DataFrame({'Time': np.arange(n_train), 'dow': train_idx.dayofweek}))
for d in range(6):
    X_train[f'D{d}'] = (X_train['dow'] == d).astype(int)
X_train = X_train[['const', 'Time'] + [f'D{d}' for d in range(6)]]
ols = sm.OLS(train.values, X_train).fit()
X_test = pd.DataFrame({'Time': np.arange(n_train, n_train + forecast_steps), 'dow': test_idx.dayofweek})
for d in range(6):
    X_test[f'D{d}'] = (X_test['dow'] == d).astype(int)
X_test = sm.add_constant(X_test[['Time'] + [f'D{d}' for d in range(6)]])
forecasts['Regression'] = ols.predict(X_test).values

# SES and Holt's (deseasonalize by day of week, then add seasonality back)
seasonal = train.groupby(train_idx.dayofweek).mean()
deseason = train.values - seasonal.reindex(train_idx.dayofweek).values + train.mean()
deseason = pd.Series(deseason, index=train_idx)
ses_fit = SimpleExpSmoothing(deseason).fit(optimized=True)
ses_fc = ses_fit.forecast(forecast_steps)
sev = seasonal.reindex(test_idx.dayofweek).fillna(seasonal.mean()).values
forecasts['SES'] = np.asarray(ses_fc) + sev - deseason.mean()
holt_fit = Holt(deseason).fit(optimized=True)
holt_fc = holt_fit.forecast(forecast_steps)
forecasts["Holt's"] = np.asarray(holt_fc) + sev - deseason.mean()

results = pd.DataFrame(index=forecast_dates)
for name, vals in forecasts.items():
    if vals is not None:
        results[name] = vals
print(f"Train: latest {MONTH_DAYS} days ({train.index[0].date()} to {train.index[-1].date()})")
print(f"Forecast: next {forecast_steps} days ({forecast_dates[0].date()} to {forecast_dates[-1].date()})")

# Graph: last month (actual) + next month (all model forecasts)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train.values, label='Actual (latest month)', color='steelblue', linewidth=2)
ax.plot(forecast_dates, forecasts['ARIMA'], label='ARIMA', color='coral', marker='s', linestyle='--', markersize=3)
ax.plot(forecast_dates, forecasts['Regression'], label='Regression', color='gray', marker='o', linestyle='--', markersize=3)
ax.plot(forecast_dates, forecasts['SES'], label='SES', color='orange', marker='^', linestyle='--', markersize=3)
ax.plot(forecast_dates, forecasts["Holt's"], label="Holt's", color='brown', marker='d', linestyle='--', markersize=3)
ax.plot(forecast_dates, forecasts['Holt-Winters'], label='Holt-Winters', color='darkgreen', marker='*', linestyle='--', markersize=3)
ax.set_ylabel('Consumption (MW)')
ax.set_xlabel('Date')
ax.set_title('Latest month (actual) and following month (forecast) — all models')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

results


# %% [markdown]
# ## Stage 5 — Evaluation & monitoring
# 
# *(Accuracy metrics, model comparison, insights)*

# %%
def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred) ** 2))
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

if test is not None:
    actual = test.values
    metrics = []
    for name, pred in forecasts.items():
        if pred is not None and len(pred) == len(actual):
            metrics.append({'Model': name, 'MAE': mae(actual, pred), 'RMSE': rmse(actual, pred), 'MAPE (%)': mape(actual, pred)})
    metrics_df = pd.DataFrame(metrics)
else:
    # No actuals (forecasting next month): use AIC to compare ARIMA vs Holt-Winters; others get NaN
    metrics_df = pd.DataFrame([
        {'Model': 'ARIMA', 'MAE': np.nan, 'RMSE': np.nan, 'AIC': fit_arima.aic},
        {'Model': 'Holt-Winters', 'MAE': np.nan, 'RMSE': np.nan, 'AIC': fit_ets.aic},
        {'Model': 'Regression', 'MAE': np.nan, 'RMSE': np.nan, 'AIC': np.nan},
        {'Model': 'SES', 'MAE': np.nan, 'RMSE': np.nan, 'AIC': np.nan},
        {"Model": "Holt's", 'MAE': np.nan, 'RMSE': np.nan, 'AIC': np.nan},
    ])
    print("Forecast period has no actuals; table shows AIC where available.")
metrics_df

# %% [markdown]
# ### Stage 5 — Insights

# %%
if test is not None and 'MAE' in metrics_df.columns and metrics_df['MAE'].notna().any():
    best_model = metrics_df.loc[metrics_df['MAE'].idxmin(), 'Model']
    best_mae = metrics_df['MAE'].min()
    best_rmse = metrics_df['RMSE'].min()
    print(f"Best model (by MAE): {best_model}")
    print(f"Best MAE: {best_mae:.4f} kW")
    print(f"Best RMSE: {best_rmse:.4f} kW")
else:
    # Pick best by AIC (ARIMA vs Holt-Winters)
    aic_sub = metrics_df[metrics_df['AIC'].notna()]
    best_model = aic_sub.loc[aic_sub['AIC'].idxmin(), 'Model'] if len(aic_sub) > 0 else 'Holt-Winters'
    print(f"Best model (by AIC, no actuals): {best_model}")
print("\nInsights: One-month-ahead forecast supports capacity planning; re-train as new data arrives.")

import matplotlib.pyplot as plt

# Graph 1: Latest month (actual) + next month (all model forecasts)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(train.index, train.values, label='Actual (latest month)', color='steelblue', linewidth=2)
ax.plot(forecast_dates, forecasts['ARIMA'], label='ARIMA', color='coral', marker='s', linestyle='--', markersize=3)
ax.plot(forecast_dates, forecasts['Regression'], label='Regression', color='gray', marker='o', linestyle='--', markersize=3)
ax.plot(forecast_dates, forecasts['SES'], label='SES', color='orange', marker='^', linestyle='--', markersize=3)
ax.plot(forecast_dates, forecasts["Holt's"], label="Holt's", color='brown', marker='d', linestyle='--', markersize=3)
ax.plot(forecast_dates, forecasts['Holt-Winters'], label='Holt-Winters', color='darkgreen', marker='*', linestyle='--', markersize=3)
ax.set_ylabel('Consumption (MW)')
ax.set_xlabel('Date')
ax.set_title('Latest month (actual) and following month (forecast) — all models')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Graph 2: Latest month + best model forecast only
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train.index, train.values, label='Actual (latest month)', color='steelblue')
ax.plot(forecast_dates, forecasts[best_model], label=f'{best_model} forecast', color='darkgreen', marker='s', linestyle='--', markersize=3)
ax.set_ylabel('Consumption (MW)')
ax.set_xlabel('Date')
ax.set_title(f'Latest month and following month — best model ({best_model})')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


