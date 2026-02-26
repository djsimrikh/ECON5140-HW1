# Helper to get cell contents for Energy_Consumption_Forecasting.ipynb (Homework 2 format)

CELL_0_MD = """# Energy Consumption Forecasting (Time Series)

**Forecasting workflow (5 stages) — aligned with Homework 2 format:**

| Stage | Focus | Tasks |
|-------|--------|-------|
| **1. Data structure** | Time index, frequency, scale | Load data, print date range, **Energy Statistics** (describe) |
| **2. Visualization (A1)** | Trend, seasonality | Time series plot, seasonal subseries (box by day), decomposition (original, trend MA, detrended) |
| **3. Modeling** | Regression, ETS | A2 Regression, A3 ETS, A4 SES, A5 Holt, A6 Holt-Winters |
| **4. Evaluation** | Accuracy, comparison | Holdout MAE/RMSE/MAPE, forecast comparison table & plot |

*Dataset: Daily energy consumption (e.g. PJM). Seasonal period = 7 days (weekly).*"""

CELL_1_CODE = r'''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ====================================================================
# DATASET 1: ENERGY CONSUMPTION (Daily, for Regression & ETS)
# ====================================================================
print("\n" + "=" * 70)
print("DATASET 1: Daily Energy Consumption")
print("=" * 70)

from pathlib import Path
base_dir = Path.cwd()
data_dir = base_dir / "data"
archive_dir = Path.home() / "Downloads" / "archive"
search_dirs = [d for d in [data_dir, archive_dir] if d.exists()]

def load_one_csv(path):
    try:
        raw = pd.read_csv(path)
        if raw.empty or len(raw.columns) < 2:
            return None
        dt_col = next((c for c in raw.columns if "date" in c.lower() or "time" in c.lower()), raw.columns[0])
        mw_col = next((c for c in raw.columns if "mw" in c.lower() or "power" in c.lower()), raw.columns[1] if len(raw.columns) > 1 else raw.columns[0])
        df = raw[[dt_col, mw_col]].rename(columns={dt_col: "datetime", mw_col: "Global_active_power"})
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime", "Global_active_power"])
        df = df.set_index("datetime").sort_index()
        df_hourly = df.copy()
        df_daily = df.resample("1D").mean().dropna()
        if len(df_daily) < 14:
            return None
        return df, df_hourly, df_daily
    except (ValueError, KeyError, OSError):
        return None

databases = {}
for folder in search_dirs:
    for f in sorted(folder.glob("*.csv")):
        key = f.stem
        if key in databases:
            continue
        out = load_one_csv(f)
        if out is not None:
            df, df_h, df_d = out
            databases[key] = {"df": df, "hourly": df_h, "daily": df_d}
            print(f"  Loaded: {f.name} -> '{key}'")

series_name = "PJME_hourly" if "PJME_hourly" in databases else (list(databases.keys())[0] if databases else None)
if series_name and series_name in databases:
    db = databases[series_name]
    df_hourly = db["hourly"]
    df_daily = db["daily"]
else:
    dates = pd.date_range("2007-01-01", periods=10000, freq="1h")
    np.random.seed(42)
    base = 32000 + 5000 * np.sin(np.arange(len(dates)) * 2 * np.pi / 24)
    df = pd.DataFrame({"Global_active_power": base + np.random.normal(0, 500, len(dates))}, index=dates)
    df.index.name = "datetime"
    df_hourly = df.copy()
    df_daily = df.resample("1D").mean().dropna()
    databases = {}

# Build df_energy (daily) with same structure as Homework 2 df_sales
y_daily = df_daily["Global_active_power"].dropna()
dates_d = y_daily.index
n_days = len(dates_d)
t = np.arange(n_days)
df_energy = pd.DataFrame({
    "Date": dates_d,
    "Consumption": y_daily.values,
    "DayOfWeek": dates_d.dayofweek + 1,  # 1=Mon .. 7=Sun
    "Time": t
})
df_energy.set_index("Date", inplace=True)

print(f"Date range: {df_energy.index[0].date()} to {df_energy.index[-1].date()}")
print(f"Number of days: {len(df_energy)}")
print(f"\nEnergy (Consumption) Statistics:")
print(df_energy["Consumption"].describe())'''

CELL_1_CONTINUED_MD = """### Stage 1 (continued) — Structure report

*(Time index, frequency, scale, missingness)*"""

CELL_1_CONTINUED_CODE = r'''# Structure report
y = df_hourly["Global_active_power"]
print("1. Time index:", y.index.name or "datetime", "| dtype:", y.index.dtype)
print("2. Frequency: inferred", pd.infer_freq(y.index[:100]) or "irregular (check gaps)")
print("3. Scale: mean = {:.2f}, std = {:.2f}, min = {:.2f}, max = {:.2f} (MW)".format(y.mean(), y.std(), y.min(), y.max()))
missing = y.isna().sum()
gaps = y.index.to_series().diff().gt(pd.Timedelta(hours=2)).sum() if hasattr(y.index, "to_series") else 0
print("4. Missingness: {} NaN(s), ~{} gap(s) > 2h".format(missing, gaps))'''

CELL_1_GRAPH_MD = """### Stage 1 — Graph data

*(By year, month, hour; box plot by month)*"""

CELL_1B_CODE = r'''# Stage 1: Graph data as-is by year, month, and hour
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# By year: mean daily consumption per year
by_year = df_energy["Consumption"].resample("Y").mean()
axes[0].bar(range(len(by_year)), by_year.values, color="steelblue", edgecolor="navy", alpha=0.8)
axes[0].set_xticks(range(len(by_year)))
axes[0].set_xticklabels([d.strftime("%Y") for d in by_year.index], rotation=0)
axes[0].set_ylabel("Mean consumption (MW)")
axes[0].set_title("Energy Consumption by Year")
axes[0].grid(True, alpha=0.3)

# By month: mean consumption by month of year (1-12)
by_month = df_energy.groupby(df_energy.index.month)["Consumption"].mean()
axes[1].bar(by_month.index, by_month.values, color="darkgreen", edgecolor="navy", alpha=0.8)
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Mean consumption (MW)")
axes[1].set_title("Energy Consumption by Month (1=Jan .. 12=Dec)")
axes[1].grid(True, alpha=0.3)

# By hour: mean consumption by hour of day (0-23) from hourly data
y_h = df_hourly["Global_active_power"]
by_hour = y_h.groupby(y_h.index.hour).mean()
axes[2].bar(by_hour.index, by_hour.values, color="coral", edgecolor="navy", alpha=0.8)
axes[2].set_xlabel("Hour of day")
axes[2].set_ylabel("Mean consumption (MW)")
axes[2].set_title("Energy Consumption by Hour of Day")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Box plot: monthly data (distribution of daily consumption by month)
fig, ax = plt.subplots(figsize=(12, 4))
df_energy["Month"] = df_energy.index.month
df_energy.boxplot(column="Consumption", by="Month", ax=ax)
ax.set_xlabel("Month")
ax.set_ylabel("Consumption (MW)")
ax.set_title("Energy Consumption by Month (Box Plot)")
plt.suptitle("")
plt.tight_layout()
plt.show()'''

CELL_2_MD = """# ====================================================================
# PART A: REGRESSION AND ETS MODELS
# ===================================================================="""

CELL_3_CODE = r'''print("\n" + "=" * 70)
print("PART A: REGRESSION AND ETS MODELS")
print("=" * 70)

# --------------------------------------------------------------------
# A1: Time Series Visualization
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A1: Exploratory Visualization")
print("-" * 70)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_energy.index, df_energy["Consumption"], color="steelblue")
ax.set_title("Daily Energy Consumption")
ax.set_xlabel("Date")
ax.set_ylabel("Consumption (MW)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(12, 4))
df_energy.boxplot(column="Consumption", by="DayOfWeek", ax=ax)
ax.set_title("Consumption by Day of Week (Seasonal Subseries)")
ax.set_xlabel("Day of Week (1=Mon .. 7=Sun)")
ax.set_ylabel("Consumption")
plt.suptitle("")
plt.tight_layout()
plt.show()

daily_means = df_energy.groupby("DayOfWeek")["Consumption"].mean()
print("Days with highest average consumption:")
print(daily_means.sort_values(ascending=False).head())

# 3. Decomposition: Original, 7-day MA (trend), Detrended
trend_7d = df_energy["Consumption"].rolling(window=7, center=True).mean()
detrended = df_energy["Consumption"] - trend_7d

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(df_energy.index, df_energy["Consumption"], color="steelblue")
axes[0].set_ylabel("Consumption")
axes[0].set_title("Original Series")
axes[0].grid(True, alpha=0.3)
axes[1].plot(df_energy.index, trend_7d, color="darkgreen")
axes[1].set_ylabel("Consumption")
axes[1].set_title("7-Day Moving Average (Trend)")
axes[1].grid(True, alpha=0.3)
axes[2].plot(df_energy.index, detrended, color="coral")
axes[2].set_ylabel("Consumption")
axes[2].set_xlabel("Date")
axes[2].set_title("Detrended Series (Original - Trend)")
axes[2].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()'''

CELL_4_CODE = r'''# --------------------------------------------------------------------
# A2: Regression Model (Trend + Seasonal)
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A2: Regression Model")
print("-" * 70)

df_reg = df_energy.copy()
for d in range(1, 7):  # Day 1..6 dummies; Day 7 (Sun) is reference
    df_reg[f"Day_{d}"] = (df_reg["DayOfWeek"] == d).astype(int)

X = df_reg[["Time"] + [f"Day_{d}" for d in range(1, 7)]]
X = sm.add_constant(X)
y = df_reg["Consumption"]

model_reg = sm.OLS(y, X).fit()
print(model_reg.summary())
print(f"\nR-squared: {model_reg.rsquared:.4f}")'''

CELL_5_CODE = r'''# --------------------------------------------------------------------
# A3: ETS Model (Holt-Winters Exponential Smoothing)
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A3: ETS Model (Holt-Winters)")
print("-" * 70)

ets_model = ExponentialSmoothing(
    df_energy["Consumption"],
    seasonal_periods=7,
    trend="add",
    seasonal="add"
).fit()

print(ets_model.summary())
print(f"\nETS AIC: {ets_model.aic:.2f}")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_energy.index, df_energy["Consumption"], label="Actual", color="steelblue")
ax.plot(df_energy.index, ets_model.fittedvalues, label="ETS Fitted", color="darkgreen", alpha=0.8)
ax.set_title("Daily Energy: Actual vs ETS (Holt-Winters) Fitted")
ax.set_xlabel("Date")
ax.set_ylabel("Consumption")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()'''

CELL_6_CODE = r'''# --------------------------------------------------------------------
# A4: Simple Exponential Smoothing (SES)
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A4: Simple Exponential Smoothing")
print("-" * 70)

train_size = len(df_energy) - 7
holdout_size = 7
train_data = df_energy["Consumption"].iloc[:train_size]
train_index = df_energy.index[:train_size]

seasonal_component = train_data.groupby(train_index.dayofweek).mean()
seasonal_vals_train = seasonal_component.reindex(train_index.dayofweek).values
deseasonalized = train_data.values - seasonal_vals_train + train_data.mean()
deseasonalized = pd.Series(deseasonalized, index=train_index)

ses_model = SimpleExpSmoothing(deseasonalized).fit(optimized=True)
alpha = ses_model.params["smoothing_level"]
print(f"Optimized α (smoothing level): {alpha:.4f}")
print("Interpretation: α controls weight on recent observations.")
print(f"  - α = {alpha:.2f} means moderate responsiveness to recent changes.")

ses_forecast_deseason = ses_model.forecast(steps=holdout_size)
holdout_dates_a4 = df_energy.index[train_size:train_size + holdout_size]
seasonal_vals = seasonal_component.reindex(holdout_dates_a4.dayofweek).fillna(seasonal_component.mean())
ses_forecast = ses_forecast_deseason.values + seasonal_vals.values - deseasonalized.mean()

holdout_actual = df_energy["Consumption"].iloc[train_size:train_size + holdout_size].values
mae = np.mean(np.abs(ses_forecast - holdout_actual))
rmse = np.sqrt(np.mean((ses_forecast - holdout_actual) ** 2))
mape = np.mean(np.abs((ses_forecast - holdout_actual) / (holdout_actual + 1e-8))) * 100
print(f"\nForecast accuracy on holdout period (last {holdout_size} days):")
print(f"  MAE:  {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
print(f"  MAPE: {mape:.2f}%")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df_energy.index, df_energy["Consumption"], label="Actual", color="steelblue")
ax.plot(holdout_dates_a4, ses_forecast, label="SES Forecast (with seasonality)", color="coral", linestyle="--", marker="o")
ax.axvline(df_energy.index[train_size - 1], color="gray", linestyle=":", alpha=0.7, label="Train/Holdout split")
ax.set_title("A4: Simple Exponential Smoothing - Holdout Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Consumption")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()'''

CELL_7_CODE = r'''# --------------------------------------------------------------------
# A5: Holt's Linear Trend Method
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A5: Holt's Method")
print("-" * 70)

holt_model = Holt(deseasonalized).fit(optimized=True)
alpha_holt = holt_model.params["smoothing_level"]
beta_holt = holt_model.params["smoothing_trend"]
print(f"Optimized α (level smoothing): {alpha_holt:.4f}")
print(f"Optimized β (trend smoothing): {beta_holt:.4f}")

level = float(holt_model.level.iloc[-1])
trend_val = float(holt_model.trend.iloc[-1])
print(f"\nFinal state:")
print(f"  Level (ℓₜ): {level:.4f}")
print(f"  Trend (bₜ): {trend_val:.4f}")

holt_forecast_deseason = holt_model.forecast(steps=holdout_size)
seasonal_vals_holt = seasonal_component.reindex(holdout_dates_a4.dayofweek).fillna(seasonal_component.mean())
holt_forecast = np.asarray(holt_forecast_deseason) + seasonal_vals_holt.values - deseasonalized.mean()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_energy.index, df_energy["Consumption"], label="Actual", color="steelblue")
ax.plot(holdout_dates_a4, ses_forecast, label="SES Forecast", color="coral", linestyle="--", marker="o", markersize=4)
ax.plot(holdout_dates_a4, holt_forecast, label="Holt's Forecast", color="darkgreen", linestyle="--", marker="s", markersize=4)
ax.axvline(df_energy.index[train_size - 1], color="gray", linestyle=":", alpha=0.7, label="Train/Holdout split")
ax.set_title("A5: SES vs Holt's Linear Trend - Holdout Forecasts")
ax.set_xlabel("Date")
ax.set_ylabel("Consumption")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nWhich captures the trend better?")
print("Holt's method captures the trend better because it explicitly models the trend component (β).")'''

CELL_8_CODE = r'''# --------------------------------------------------------------------
# A6: Holt-Winters Seasonal Method
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A6: Holt-Winters Method")
print("-" * 70)

hw_train = df_energy["Consumption"].iloc[:train_size]
hw_model = ExponentialSmoothing(
    hw_train,
    trend="add",
    seasonal="add",
    seasonal_periods=7
).fit(optimized=True)

alpha_hw = hw_model.params["smoothing_level"]
beta_hw = hw_model.params["smoothing_trend"]
gamma_hw = hw_model.params["smoothing_seasonal"]
print(f"Optimized α (level): {alpha_hw:.4f}")
print(f"Optimized β (trend): {beta_hw:.4f}")
print(f"Optimized γ (seasonal): {gamma_hw:.4f}")

level_hw = hw_model.level.iloc[-1]
trend_hw = hw_model.trend.iloc[-1]
seasonal_hw = hw_model.season.iloc[-7:]
print(f"\nFinal state:")
print(f"  Level (ℓₜ): {level_hw:.4f}")
print(f"  Trend (bₜ): {trend_hw:.4f}")
print(f"  Seasonal indices (last 7):\n{seasonal_hw}")

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(hw_train.index, hw_train.values, color="steelblue")
axes[0].set_ylabel("Consumption")
axes[0].set_title("Original Series")
axes[0].grid(True, alpha=0.3)
axes[1].plot(hw_train.index, hw_model.level, color="darkgreen")
axes[1].set_ylabel("Level")
axes[1].set_title("Level (ℓₜ)")
axes[1].grid(True, alpha=0.3)
axes[2].plot(hw_train.index, hw_model.trend, color="orange")
axes[2].set_ylabel("Trend")
axes[2].set_title("Trend (bₜ)")
axes[2].grid(True, alpha=0.3)
axes[3].plot(hw_train.index, hw_model.season, color="coral")
axes[3].set_ylabel("Seasonal")
axes[3].set_xlabel("Date")
axes[3].set_title("Seasonal Component")
axes[3].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

hw_forecast = hw_model.forecast(steps=holdout_size)

def mae(y_true, y_pred):
    return np.mean(np.abs(np.asarray(y_pred) - y_true))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.asarray(y_pred) - y_true) ** 2))

holdout_dates = df_energy.index[train_size:train_size + holdout_size]
holdout_t = df_reg["Time"].iloc[train_size:train_size + holdout_size].values
# DayOfWeek in df_reg is 1=Mon..7=Sun (from dayofweek+1)
holdout_dayofweek = df_energy["DayOfWeek"].iloc[train_size:train_size + holdout_size].values
X_holdout_list = []
for i in range(holdout_size):
    row = {"Time": holdout_t[i]}
    for d in range(1, 7):
        row[f"Day_{d}"] = 1 if holdout_dayofweek[i] == d else 0
    X_holdout_list.append(row)
X_holdout = sm.add_constant(pd.DataFrame(X_holdout_list)[["Time"] + [f"Day_{d}" for d in range(1, 7)]])
reg_forecast_holdout = model_reg.predict(X_holdout).values

acc_df = pd.DataFrame({
    "Model": ["Regression", "SES", "Holt's", "Holt-Winters"],
    "MAE": [
        mae(holdout_actual, reg_forecast_holdout),
        mae(holdout_actual, ses_forecast),
        mae(holdout_actual, holt_forecast),
        mae(holdout_actual, hw_forecast)
    ],
    "RMSE": [
        rmse(holdout_actual, reg_forecast_holdout),
        rmse(holdout_actual, ses_forecast),
        rmse(holdout_actual, holt_forecast),
        rmse(holdout_actual, hw_forecast)
    ]
})
print("\nForecast accuracy comparison (holdout period):")
print(acc_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df_energy.index, df_energy["Consumption"], label="Actual", color="steelblue")
ax.plot(holdout_dates, reg_forecast_holdout, label="Regression", linestyle="--", alpha=0.8)
ax.plot(holdout_dates, ses_forecast, label="SES", linestyle="--", alpha=0.8)
ax.plot(holdout_dates, holt_forecast, label="Holt's", linestyle="--", alpha=0.8)
ax.plot(holdout_dates, hw_forecast, label="Holt-Winters", linestyle="--", alpha=0.8)
ax.axvline(df_energy.index[train_size - 1], color="gray", linestyle=":", alpha=0.7)
ax.set_title("A6: Forecast Comparison - Regression, SES, Holt's, Holt-Winters")
ax.set_xlabel("Date")
ax.set_ylabel("Consumption")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()'''

CELL_9_MD = """### Stage 5 — Insights

- Best model by MAE/RMSE from the comparison table above.
- Weekly (7-day) seasonality and trend are captured by Regression and Holt-Winters.
- Re-train periodically as new data arrives."""

CELL_10_CODE = r'''best_model = acc_df.loc[acc_df["MAE"].idxmin(), "Model"]
best_mae = acc_df["MAE"].min()
best_rmse = acc_df["RMSE"].min()
print(f"Best model (by MAE): {best_model}")
print(f"Best MAE: {best_mae:.2f}")
print(f"Best RMSE: {best_rmse:.2f}")
print("\nInsights: 7-day forecasts support capacity planning; daily/weekly seasonality show peak periods; re-train periodically.")'''

if __name__ == "__main__":
    print("Cells defined.")
    print("Total:", "CELL_0_MD..CELL_8_CODE, CELL_9_MD, CELL_10_CODE")