"""
ECON 5140 - Homework 1, Problem 5
Customer Purchase Prediction & Time Series Analysis
"""
# Disable OpenMP shared memory (avoids OMP SHM2 error in some environments)
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Logit, Probit, Poisson
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("ECON 5140 - HOMEWORK 1")
print("Part A: Generalized Linear Models")
print("Part B: Time Series Decomposition")
print("=" * 70)

# ====================================================================
# DATASET 1: CUSTOMER PURCHASE DATA (for GLM analysis)
# ====================================================================
print("\n" + "=" * 70)
print("DATASET 1: Customer Purchase Behavior")
print("=" * 70)

n_customers = 1000
# Generate customer features
age = np.random.normal(35, 10, n_customers)
income = np.random.normal(50, 15, n_customers)  # in thousands
time_on_site = np.random.gamma(2, 3, n_customers)  # in minutes
# True relationship (latent variable)
z = -3 + 0.05*age + 0.04*income + 0.15*time_on_site + np.random.normal(0, 1, n_customers)
# Generate binary outcome (Purchase: 1=Yes, 0=No)
purchase = (z > 0).astype(int)
# Create DataFrame
df_customers = pd.DataFrame({
    'Age': age,
    'Income': income,
    'TimeOnSite': time_on_site,
    'Purchase': purchase
})

print(f"Number of customers: {len(df_customers)}")
print(f"Purchase rate: {df_customers['Purchase'].mean():.2%}")
print(f"\nFirst 5 rows:")
print(df_customers.head())

# ====================================================================
# DATASET 2: E-COMMERCE SALES TIME SERIES
# ====================================================================
print("\n" + "=" * 70)
print("DATASET 2: E-commerce Daily Sales")
print("=" * 70)

# Create 2 years of daily data
dates = pd.date_range('2024-01-01', '2025-12-31', freq='D')
n_days = len(dates)
t = np.arange(n_days)
# Components
trend = 1000 + 2*t + 0.01*t**2
yearly_seasonal = 200 * np.sin(2*np.pi*t/365) + 150 * np.cos(2*np.pi*t/365)
weekly_seasonal = 100 * np.sin(2*np.pi*t/7)
# Special events
special_events = np.zeros(n_days)
for year in [2024, 2025]:
    # Black Friday
    bf_date = pd.Timestamp(f'{year}-11-24')
    bf_idx = (dates == bf_date)
    special_events[bf_idx] = 800
    # Christmas
    xmas_idx = (dates >= f'{year}-12-20') & (dates <= f'{year}-12-25')
    special_events[xmas_idx] = 400
# Random noise
noise = np.random.normal(0, 50, n_days)
# Combine components
sales = trend + yearly_seasonal + weekly_seasonal + special_events + noise
sales = np.maximum(sales, 0)
# Create DataFrame
df_sales = pd.DataFrame({
    'Date': dates,
    'Sales': sales,
    'DayOfWeek': dates.dayofweek,
    'Month': dates.month,
    'IsWeekend': dates.dayofweek >= 5
})
df_sales.set_index('Date', inplace=True)

print(f"Date range: {df_sales.index[0].date()} to {df_sales.index[-1].date()}")
print(f"Number of days: {len(df_sales)}")
print(f"\nSales Statistics:")
print(df_sales['Sales'].describe())

# ====================================================================
# PART A: GENERALIZED LINEAR MODELS
# ====================================================================
print("\n" + "=" * 70)
print("PART A: GENERALIZED LINEAR MODELS")
print("=" * 70)

# --------------------------------------------------------------------
# A1: Exploratory Data Analysis (GLM)
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A1: Exploratory Data Analysis")
print("-" * 70)

# 1. Box plots comparing Age, Income, TimeOnSite by Purchase
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
features = ['Age', 'Income', 'TimeOnSite']
for ax, feat in zip(axes, features):
    df_customers.boxplot(column=feat, by='Purchase', ax=ax)
    ax.set_title(f'{feat} by Purchase')
    ax.set_xlabel('Purchase (0=No, 1=Yes)')
plt.suptitle('A1: Feature Distribution by Purchase Status', y=1.02)
plt.tight_layout()
plt.savefig('A1_boxplots.png', bbox_inches='tight', dpi=100)
plt.show()

# 2. Mean values by group
purchasers = df_customers[df_customers['Purchase'] == 1]
non_purchasers = df_customers[df_customers['Purchase'] == 0]
print("Mean Age:  Purchasers = {:.2f}, Non-purchasers = {:.2f}".format(
    purchasers['Age'].mean(), non_purchasers['Age'].mean()))
print("Mean Income:  Purchasers = {:.2f}, Non-purchasers = {:.2f}".format(
    purchasers['Income'].mean(), non_purchasers['Income'].mean()))
print("Mean TimeOnSite:  Purchasers = {:.2f}, Non-purchasers = {:.2f}".format(
    purchasers['TimeOnSite'].mean(), non_purchasers['TimeOnSite'].mean()))

# 3. Correlation matrix heatmap
corr = df_customers[['Age', 'Income', 'TimeOnSite', 'Purchase']].corr()
plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
plt.colorbar(label='Correlation')
plt.xticks(range(4), ['Age', 'Income', 'TimeOnSite', 'Purchase'])
plt.yticks(range(4), ['Age', 'Income', 'TimeOnSite', 'Purchase'])
for i in range(4):
    for j in range(4):
        plt.text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center', fontsize=10)
plt.title('A1: Correlation Matrix')
plt.tight_layout()
plt.savefig('A1_correlation_heatmap.png', bbox_inches='tight', dpi=100)
plt.show()

# --------------------------------------------------------------------
# A2: Linear Probability Model (LPM)
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A2: Linear Probability Model")
print("-" * 70)

X_lpm = sm.add_constant(df_customers[['Age', 'Income', 'TimeOnSite']])
y = df_customers['Purchase']
model_lpm = sm.OLS(y, X_lpm).fit()
print(model_lpm.summary())

pred_lpm = model_lpm.predict(X_lpm)
invalid = np.sum((pred_lpm < 0) | (pred_lpm > 1))
pct_invalid = 100 * invalid / len(pred_lpm)
print(f"\nPredictions outside [0, 1]: {invalid} ({pct_invalid:.2f}%)")

plt.figure(figsize=(8, 4))
plt.hist(pred_lpm, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', label='Valid range [0,1]')
plt.axvline(1, color='red', linestyle='--')
plt.xlabel('Predicted probability')
plt.ylabel('Frequency')
plt.title('A2: LPM Predicted Probabilities')
plt.legend()
plt.tight_layout()
plt.savefig('A2_lpm_histogram.png', bbox_inches='tight', dpi=100)
plt.show()

# --------------------------------------------------------------------
# A3: Logistic Regression
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A3: Logistic Regression")
print("-" * 70)

X_logit = sm.add_constant(df_customers[['Age', 'Income', 'TimeOnSite']])
model_logit = Logit(y, X_logit).fit()
print(model_logit.summary())

coefs = model_logit.params
odds_ratios = np.exp(coefs)
pvals = model_logit.pvalues
print("\nCoefficients (log-odds):")
for name in coefs.index:
    print(f"  {name}: {coefs[name]:.4f} (p={pvals[name]:.4f})")
print("\nOdds ratios:")
for name in odds_ratios.index:
    print(f"  {name}: {odds_ratios[name]:.4f}")

print("\nInterpretation:")
print("  Age: One year increase → log-odds of purchase increases by {:.4f}.".format(coefs['Age']))
print("  Income: One unit ($1k) increase → log-odds increases by {:.4f}.".format(coefs['Income']))
print("  TimeOnSite: One minute increase → log-odds increases by {:.4f}.".format(coefs['TimeOnSite']))

pred_logit = model_logit.predict(X_logit)
print(f"\nAll predicted probabilities in [0,1]: {np.all((pred_logit >= 0) & (pred_logit <= 1))}")

plt.figure(figsize=(8, 4))
plt.hist(pred_logit, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(0, color='red', linestyle='--')
plt.axvline(1, color='red', linestyle='--')
plt.xlabel('Predicted probability')
plt.ylabel('Frequency')
plt.title('A3: Logistic Predicted Probabilities')
plt.tight_layout()
plt.savefig('A3_logit_histogram.png', bbox_inches='tight', dpi=100)
plt.show()

# --------------------------------------------------------------------
# A4: Prediction for New Customers
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("A4: Predictions for New Customers")
print("-" * 70)

new_customers = pd.DataFrame({
    'Age': [25, 35, 45, 55],
    'Income': [30, 50, 70, 90],
    'TimeOnSite': [2, 5, 8, 10]
})
X_new = sm.add_constant(new_customers)
prob_new = model_logit.predict(X_new)
class_new = (prob_new > 0.5).astype(int)

table = new_customers.copy()
table['Predicted_Probability'] = prob_new
table['Predicted_Purchase'] = class_new
print(table.to_string(index=False))

prob_arr = np.asarray(prob_new)
idx_max = int(np.argmax(prob_arr))
row = new_customers.iloc[idx_max]
print(f"\nMost likely to purchase: Customer with Age={row['Age']}, Income={row['Income']}, TimeOnSite={row['TimeOnSite']}")
print(f"  Probability = {prob_arr[idx_max]:.4f}. Higher income and time on site increase purchase probability.")

# ====================================================================
# PART B: TIME SERIES ANALYSIS
# ====================================================================
print("\n" + "=" * 70)
print("PART B: TIME SERIES ANALYSIS")
print("=" * 70)

# --------------------------------------------------------------------
# B1: Time Series Visualization
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B1: Time Series Visualization")
print("-" * 70)

plt.figure(figsize=(12, 4))
plt.plot(df_sales.index, df_sales['Sales'], linewidth=0.8)
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('B1: Daily E-commerce Sales (2024–2025)')
plt.tight_layout()
plt.savefig('B1_timeseries.png', bbox_inches='tight', dpi=100)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df_sales.boxplot(column='Sales', by='DayOfWeek', ax=axes[0])
axes[0].set_title('Sales by Day of Week')
axes[0].set_xlabel('Day (0=Mon, 6=Sun)')
df_sales.boxplot(column='Sales', by='Month', ax=axes[1])
axes[1].set_title('Sales by Month')
axes[1].set_xlabel('Month')
plt.suptitle('B1: Seasonal Subseries', y=1.02)
plt.tight_layout()
plt.savefig('B1_seasonal_subseries.png', bbox_inches='tight', dpi=100)
plt.show()

print("Mean sales by day of week:")
print(df_sales.groupby('DayOfWeek')['Sales'].mean().to_string())
print("\nMean sales by month:")
print(df_sales.groupby('Month')['Sales'].mean().to_string())
print("\nPatterns: Weekly and yearly seasonality; trend; spikes at Black Friday and Christmas.")

# --------------------------------------------------------------------
# B2: Stationarity Assessment
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B2: Stationarity Check")
print("-" * 70)

sales_series = df_sales['Sales'].values
roll_mean = pd.Series(sales_series).rolling(30).mean()
roll_std = pd.Series(sales_series).rolling(30).std()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(df_sales.index, sales_series, linewidth=0.6)
axes[0].set_ylabel('Sales')
axes[0].set_title('Original Series')
axes[1].plot(df_sales.index, roll_mean, color='orange')
axes[1].set_ylabel('Sales')
axes[1].set_title('30-day Rolling Mean')
axes[2].plot(df_sales.index, roll_std, color='green')
axes[2].set_ylabel('Sales')
axes[2].set_xlabel('Date')
axes[2].set_title('30-day Rolling Std')
plt.suptitle('B2: Stationarity Assessment', y=1.02)
plt.tight_layout()
plt.savefig('B2_stationarity.png', bbox_inches='tight', dpi=100)
plt.show()

print("The series is NOT stationary: rolling mean and variance change over time (trend and seasonality).")

first_6m = df_sales.loc['2024-01-01':'2024-06-30', 'Sales']
last_6m = df_sales.loc['2025-07-01':'2025-12-31', 'Sales']
print("First 6 months: mean = {:.2f}, std = {:.2f}".format(first_6m.mean(), first_6m.std()))
print("Last 6 months:  mean = {:.2f}, std = {:.2f}".format(last_6m.mean(), last_6m.std()))

# --------------------------------------------------------------------
# B3: Autocorrelation Analysis
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B3: Autocorrelation Function")
print("-" * 70)

fig, ax = plt.subplots(figsize=(10, 4))
plot_acf(df_sales['Sales'], lags=60, ax=ax)
plt.title('B3: ACF of Sales (up to 60 lags)')
plt.tight_layout()
plt.savefig('B3_acf.png', bbox_inches='tight', dpi=100)
plt.show()

for lag in [1, 7, 30]:
    r = np.corrcoef(sales_series[:-lag], sales_series[lag:])[0, 1]
    print(f"Lag {lag}: autocorrelation = {r:.4f}")

print("Weekly pattern: strong correlation at lag 7. Autocorrelation is persistent (slow decay).")

# --------------------------------------------------------------------
# B4: STL Decomposition
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B4: STL Decomposition")
print("-" * 70)

stl = STL(df_sales['Sales'], seasonal=7, robust=True)
result_stl = stl.fit()

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
axes[0].plot(df_sales.index, result_stl.observed)
axes[0].set_ylabel('Sales')
axes[0].set_title('Observed')
axes[1].plot(df_sales.index, result_stl.trend)
axes[1].set_ylabel('Sales')
axes[1].set_title('Trend')
axes[2].plot(df_sales.index, result_stl.seasonal)
axes[2].set_ylabel('Sales')
axes[2].set_title('Seasonal (weekly)')
axes[3].plot(df_sales.index, result_stl.resid)
axes[3].set_ylabel('Sales')
axes[3].set_xlabel('Date')
axes[3].set_title('Remainder')
plt.suptitle('B4: STL Decomposition', y=1.02)
plt.tight_layout()
plt.savefig('B4_stl.png', bbox_inches='tight', dpi=100)
plt.show()

print("Trend: upward and slightly quadratic. Seasonal: 7-day cycle. Special events (Black Friday, Christmas) appear in remainder.")

# --------------------------------------------------------------------
# B5: Remainder Diagnostics
# --------------------------------------------------------------------
print("\n" + "-" * 70)
print("B5: Remainder Analysis")
print("-" * 70)

remainder = result_stl.resid

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(df_sales.index, remainder, linewidth=0.6)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Remainder')
axes[0].set_title('Remainder Time Series')
axes[1].hist(remainder, bins=40, edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Remainder')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Remainder Histogram')
plot_acf(remainder.dropna(), lags=40, ax=axes[2])
axes[2].set_title('ACF of Remainder')
plt.suptitle('B5: Remainder Diagnostics', y=1.02)
plt.tight_layout()
plt.savefig('B5_remainder.png', bbox_inches='tight', dpi=100)
plt.show()

print("Remainder - Mean: {:.4f}, Std: {:.4f}".format(remainder.mean(), remainder.std()))
_, pnorm = stats.normaltest(remainder.dropna())
print("Normality test (D'Agostino): p-value = {:.4f}".format(pnorm))

threshold = 3 * remainder.std()
outliers = remainder[np.abs(remainder) > threshold]
print(f"\nOutliers (|remainder| > 3*std): {len(outliers)} dates")
if len(outliers) > 0:
    print(outliers.head(15).to_string())
    print("... (Black Friday and Christmas spikes can appear as large positive remainder)")

print("\n" + "=" * 70)
print("END OF PROBLEM 5")
print("=" * 70)
