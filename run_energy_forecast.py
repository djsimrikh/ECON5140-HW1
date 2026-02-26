"""Run Energy Consumption Forecasting notebook code as a script.
Usage: python run_energy_forecast.py [SERIES_NAME]
  SERIES_NAME = key from loaded CSVs (e.g. PJME_hourly, PJMW_hourly). Default: first available.
  All CSVs in data/ and ~/Downloads/archive/ are loaded."""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# T1: Load all CSVs from project data/ and ~/Downloads/archive/
print("\n" + "=" * 70)
print("T1: Load and Aggregate Data")
print("=" * 70)

base_dir = Path(__file__).resolve().parent
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
    except Exception:
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

series_name = (sys.argv[1] if len(sys.argv) > 1 else None) or ('PJME_hourly' if 'PJME_hourly' in databases else (list(databases.keys())[0] if databases else None))

if series_name and series_name in databases:
    db = databases[series_name]
    df = db['df']
    df_hourly = db['hourly']
    df_daily = db['daily']
    df_yearly = db['yearly']
    print(f"\nUsing series: {series_name}")
else:
    print("No CSV found or unknown series. Using synthetic sample.")
    dates = pd.date_range('2007-01-01', periods=10000, freq='1h')
    np.random.seed(42)
    base = 32000 + 5000 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    df = pd.DataFrame({'Global_active_power': base + np.random.normal(0, 500, len(dates))}, index=dates)
    df.index.name = 'datetime'
    df_hourly = df.copy()
    df_daily = df.resample('1D').mean().dropna()
    df_yearly = df.resample('1Y').mean().dropna()

print(f"Available: {list(databases.keys()) if databases else '[]'}")
print(f"Original rows: {len(df)}, Hourly: {len(df_hourly)}, Daily: {len(df_daily)}, Yearly: {len(df_yearly)}")
print(f"Date range: {df_hourly.index[0]} to {df_hourly.index[-1]}")
print(f"Mean: {df_hourly['Global_active_power'].mean():.3f} kW, Std: {df_hourly['Global_active_power'].std():.3f}")
print("\nFirst 5 hourly rows:")
print(df_hourly.head())

# T2: Forecast
print("\n" + "=" * 70)
print("T2: Forecast 1-week ahead (ARIMA, ETS, Prophet)")
print("=" * 70)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

y = df_daily['Global_active_power'].dropna()
train = y.iloc[:-7]
test = y.iloc[-7:]
forecasts = {}

model_arima = ARIMA(train, order=(2, 1, 2), seasonal_order=(0, 0, 0, 0))
fit_arima = model_arima.fit()
forecasts['ARIMA'] = fit_arima.forecast(steps=7)

try:
    model_ets = ExponentialSmoothing(train, seasonal_periods=7, trend='add', seasonal='add')
    fit_ets = model_ets.fit()
    forecasts['ETS'] = fit_ets.forecast(steps=7)
except Exception:
    model_ets = ExponentialSmoothing(train, trend='add')
    fit_ets = model_ets.fit()
    forecasts['ETS'] = fit_ets.forecast(steps=7)

try:
    from prophet import Prophet
    df_prophet = train.reset_index().rename(columns={'datetime': 'ds', 'Global_active_power': 'y'})
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=7)
    pred_prophet = m.predict(future)
    forecasts['Prophet'] = pred_prophet['yhat'].tail(7).values
except ImportError:
    forecasts['Prophet'] = None
    print("Prophet not installed. Run: pip install prophet")

results = pd.DataFrame({'Actual': test.values}, index=test.index)
for name, vals in forecasts.items():
    if vals is not None:
        results[name] = vals
print(results.to_string())

# T3: Seasonality (skip plots in script, just print)
print("\n" + "=" * 70)
print("T3: Trends and Seasonality")
print("=" * 70)
from statsmodels.tsa.seasonal import seasonal_decompose
y_hr = df_hourly['Global_active_power'].dropna()
y_decomp = y_hr.tail(24 * 7 * 4)
decomp = seasonal_decompose(y_decomp, model='additive', period=24)
print("Decomposition completed (trend, seasonal, residual). Plots saved to T3_decomposition.png")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
decomp.observed.plot(ax=axes[0], title='Observed'); axes[0].set_ylabel('kW')
decomp.trend.plot(ax=axes[1], title='Trend'); axes[1].set_ylabel('kW')
decomp.seasonal.plot(ax=axes[2], title='Seasonal (24h)'); axes[2].set_ylabel('kW')
decomp.resid.plot(ax=axes[3], title='Residual'); axes[3].set_ylabel('kW')
plt.tight_layout()
plt.suptitle(f'T3: Seasonal Decomposition — {series_name or "synthetic"} (Additive, period=24h)', y=1.02)
out_path = base_dir / f'T3_decomposition_{series_name or "synthetic"}.png'
plt.savefig(out_path, dpi=100)
plt.close()
print(f"Saved plot to {out_path}")

# T4: Model comparison
print("\n" + "=" * 70)
print("T4: Model Comparison (MAE, RMSE, MAPE)")
print("=" * 70)

def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred) ** 2))
def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

actual = test.values
metrics = []
for name, pred in forecasts.items():
    if pred is not None and len(pred) == len(actual):
        metrics.append({'Model': name, 'MAE': mae(actual, pred), 'RMSE': rmse(actual, pred), 'MAPE (%)': mape(actual, pred)})
metrics_df = pd.DataFrame(metrics)
print(metrics_df.to_string(index=False))

# T5: Insights
print("\n" + "=" * 70)
print("T5: Evaluation and Insights")
print("=" * 70)
best_model = metrics_df.loc[metrics_df['MAE'].idxmin(), 'Model']
best_mae = metrics_df['MAE'].min()
best_rmse = metrics_df['RMSE'].min()
print(f"Best model (by MAE): {best_model}")
print(f"Best MAE: {best_mae:.4f} kW")
print(f"Best RMSE: {best_rmse:.4f} kW")
print("\nInsights: 7-day forecasts support capacity planning; daily/weekly seasonality show peak periods; re-train periodically.")
