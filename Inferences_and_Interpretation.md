# Statistical and Economic Inferences — Energy Consumption Forecasting

This document explains **what we did**, **why we did it**, and **what to infer** from each section and graph in the notebook.

---

## Stage 1 — Data structure

### What we did
- Loaded hourly energy consumption (e.g. PJM), resampled to **daily** and **yearly** for analysis.
- Reported **date range**, **row counts**, and **Energy (Consumption) Statistics** (count, mean, std, min, 25%, 50%, 75%, max).
- **Structure report:** time index type, inferred frequency, scale (mean, std, min, max in MW), and missingness (NaN count, gaps > 2h).
- **Graphs:** (1) Consumption **by year** (bar/line of mean per year), (2) **by month** (1–12), (3) **by hour of day** (0–23), (4) **box plot by month** (distribution of daily consumption each month), plus a list of **box-plot outliers** (days flagged as unusual within each month).

### Why we did it
- **Scale and frequency** define the right model (e.g. daily vs hourly, level in MW).
- **Missingness and gaps** affect imputation and model validity.
- **By year/month/hour** and **box plot** show where level and volatility change and where outliers sit—essential before modeling.

### Statistical inference
- **Describe()** gives location, spread, and quartiles; heavy skew or huge std relative to mean suggests transformation or robust methods.
- **Outliers by month** indicate days that are unusual given that month’s distribution (e.g. extreme demand or data issues).

### Economic inference
- **By year:** long-run growth or shifts in demand (policy, efficiency, economic cycles).
- **By month:** seasonal demand (e.g. heating/cooling, industrial activity).
- **By hour:** within-day pattern (peak hours, night troughs)—drives capacity and pricing.
- **Box plot by month:** which months are more variable (e.g. summer vs winter) and whether outliers are demand spikes or data errors.

---

## Stage 2 — Visualization

### What we did
- **Raw series:** last 2 weeks at hourly resolution.
- **Trend:** 30-day rolling mean of daily series.
- **Seasonality:** average consumption by hour of day (0–23).
- **Outliers:** count of points beyond 3× standard deviation from a 7-day rolling mean.

### Why we did it
- Visual checks for **trend**, **seasonality**, and **outliers** guide choice of model (e.g. trend + seasonal, need for differencing or dummies).
- Rolling mean separates slow movement (trend) from noise and short-run fluctuation.

### Statistical inference
- **Trend** suggests whether to include a trend term or differencing.
- **Hour-of-day pattern** suggests **daily seasonality** (e.g. period 24 for hourly, or day-of-week for daily).
- **Outlier count** indicates how much the series is driven by extremes vs “typical” days.

### Economic inference
- **Trend** can reflect economic growth, efficiency gains, or structural change.
- **Peak hours** inform capacity and demand-side policies (e.g. time-of-use tariffs).
- **Outliers** may be heat waves, cold snaps, holidays, or data issues—useful for policy and data cleaning.

---

## Stage 3 — Decomposition

### What we did
- **Additive decomposition** (e.g. `seasonal_decompose`) with a chosen period (e.g. 24 for hourly, or 7 for daily):  
  **Observed = Trend + Seasonal + Residual.**
- Plotted: **Observed**, **Trend**, **Seasonal**, **Residual** (4 panels).

### Why we did it
- Decomposition separates **trend**, **seasonal**, and **irregular** components so we can see each and choose models accordingly (e.g. ARIMA vs ETS, need for seasonal terms).
- Additive is appropriate when seasonal amplitude does not grow with level.

### Statistical inference
- **Trend** shows slow, smooth movement; **Seasonal** repeats every period; **Residual** should look like noise (no clear pattern).
- If **residual** is still seasonal or trending, the chosen period or model form may be wrong.

### Economic inference
- **Trend** = long-run demand path.
- **Seasonal** = recurring within-day or within-week patterns (workweek, time of day).
- **Residual** = unpredictable shocks (weather, events, measurement error)—important for uncertainty and risk.

---

## Stage 3 (continued) — Stationarity and ACF/PACF

### What we did
- **ADF (Augmented Dickey–Fuller):** H0 = “series has a unit root” (non-stationary). Low p-value → reject H0 → support for stationarity.
- **KPSS:** H0 = “series is stationary”. High p-value → fail to reject H0 → support for stationarity.
- **ACF and PACF** of the **first-differenced** series (and optionally of levels) to suggest AR/MA orders for ARIMA.

### Why we did it
- **ARIMA** assumes stationarity (or stationarity after differencing). ADF/KPSS tell us whether we need to difference and whether the series is suitable for ARIMA.
- **ACF/PACF** help choose **(p, d, q)** and seasonal orders; they show how past values and past shocks correlate with the current value.

### Statistical inference
- **ADF p < 0.05** and **KPSS p > 0.05** → series (or differenced series) is consistent with stationarity.
- **ACF** tailing off, **PACF** cutting off → AR signature; opposite → MA. Both tailing off → ARMA. Seasonal spikes at lags 7, 14, … suggest seasonal AR/MA.

### Economic inference
- **Non-stationarity** (e.g. unit root) implies shocks have permanent effects; **stationarity** implies mean-reversion—relevant for long-run forecasts and policy.
- **Significant ACF at lag 7** (or 24 for hourly) supports **weekly (or daily) seasonality** in demand.

---

## Stage 4 — Modeling

### What we did
- **Train/test split:** train on all but last 60 days, **test** = first 30 days of that holdout (so we have actuals to evaluate).
- **SARIMA(2,1,2)(1,1,1,7):** seasonal ARIMA with weekly seasonality; printed **summary** and **AIC/BIC**.
- **Residual diagnostics:** time plot of residuals, **histogram**, **ACF of residuals**, **Q–Q plot**, and **Ljung–Box** test.
- **Prediction intervals:** ARIMA **95% forecast interval** (e.g. `get_forecast` + `conf_int`), plotted with training tail, test actuals, point forecast, and shaded CI.
- **SARIMAX:** same SARIMA with **exogenous regressors** (e.g. day of week, month, is_weekend); forecast with exog for test period.
- **Regression:** OLS with **trend + day-of-week dummies** (e.g. 6 dummies, one day reference).
- **SES:** Simple exponential smoothing on **deseasonalized** series (by day of week), then **add seasonality back** for the forecast.
- **Holt’s:** linear trend (no seasonal) on the same deseasonalized series, then add seasonality back.
- **Holt–Winters:** ETS with **additive trend and seasonal** (e.g. `seasonal_periods=7`).
- **Graphs:** (1) Training tail + test actuals + ARIMA forecast + 95% CI; (2) Actual vs all models (ARIMA, SARIMAX, Regression, SES, Holt’s, Holt–Winters) on the **test** window.

### Why we did it
- **Historical test set** allows proper evaluation (MAE, RMSE, MAPE) and model comparison.
- **SARIMA(1,1,1,7)** captures **weekly seasonality** in line with Holt–Winters; **(2,1,2)** allows richer short-run dynamics.
- **Residual checks** validate “no autocorrelation” and approximate normality; **Ljung–Box** tests residual autocorrelation.
- **Prediction intervals** quantify **forecast uncertainty** for planning and risk.
- **Exogenous variables** (day of week, month, weekend) let demand depend on calendar in a transparent way.
- **Regression** gives an interpretable **baseline** (trend + seasonal dummies); **SES/Holt** are simple smoothing benchmarks; **Holt–Winters** is the main **seasonal ETS** benchmark.

### Statistical inference
- **SARIMA summary:** coefficient significance, AIC/BIC vs other models, residual diagnostics support or question the specification.
- **Residual ACF** and **Ljung–Box** p-value > 0.05 suggest residuals are not autocorrelated; **Q–Q** suggests normality.
- **Wide prediction intervals** mean higher uncertainty (e.g. volatile period or model uncertainty).
- **Lower MAE/RMSE** on the same test set indicates better point forecast accuracy for that horizon.

### Economic inference
- **Trend** in regression or ETS = long-run growth or decline in demand.
- **Day-of-week and weekend** effects = workday vs weekend demand (industrial, residential mix).
- **Seasonal (weekly)** = recurring weekly pattern (e.g. weekday vs weekend, or weekly industrial cycles).
- **Prediction intervals** support **capacity and reserve planning** and **risk management** (e.g. probability of exceeding a threshold).

---

## Stage 4 (continued) — Rolling forecast cross-validation

### What we did
- **Rolling origin CV:** for several splits, train on expanding window up to time \(T_i\), forecast next 30 days, compute **MAE** vs actuals; report **mean MAE ± std** over splits for **ARIMA** and **Holt–Winters**.

### Why we did it
- A single train/test split can be lucky or unlucky; **rolling CV** gives a more stable estimate of **out-of-sample performance** and variability across different “forecast origins.”
- Uses only past data in each split, so it mimics real forecasting practice.

### Statistical inference
- **Mean CV MAE** approximates expected absolute error for a 30-day-ahead forecast; **std** indicates how sensitive performance is to the training window.
- Comparing **ARIMA vs Holt–Winters** CV MAE shows which tends to do better on average for this horizon.

### Economic inference
- **Stable (low std) CV performance** suggests the model is robust across different history lengths—useful for operational forecasting.
- **Relative CV MAE** between models informs which to use for **capacity planning** or **demand response** design.

---

## Stage 5 — Evaluation and insights

### What we did
- **Metrics table:** for each model (ARIMA, SARIMAX, Regression, SES, Holt’s, Holt–Winters): **MAE**, **RMSE**, **MAPE (%)**, and **AIC/BIC** where available; rows **sorted by MAE**.
- **Best model:** model with **lowest MAE** on the test set.
- **Graphs:** (1) **Actual (test)** vs **all models’ forecasts**; (2) **Actual (test)** vs **best model’s forecast** only.

### Why we did it
- **MAE/RMSE** measure **average absolute and squared error**; **MAPE** gives a **scale-free** percentage error—all standard for comparing point forecasts.
- **AIC/BIC** compare **fit vs complexity** (with and without penalty for parameters); useful when test set is fixed and we want to rank models.
- **Plots** show **bias** (systematic over/under), **timing** (leads/lags), and **volatility** (where errors are large).

### Statistical inference
- **Best by MAE** = best for minimizing average absolute error on this test set; **best by AIC** = best trade-off of fit and parsimony in sample.
- **Systematic over- or under-forecasting** in the plot suggests bias (e.g. level shift, missing regressor); **large errors in spikes** suggest the model under-predicts extremes.

### Economic inference
- **Which model is “best”** informs **operational choice** (e.g. for scheduling, contracts, or demand response).
- **MAPE** helps communicate accuracy in **percentage terms** to non-technical stakeholders.
- **Persistent bias** may point to **structural change** (e.g. new policy, technology) not captured by the model—suggests re-estimation or extra variables.

---

## Summary table: section → what we did → why → inference

| Section | What we did | Why | Key inference |
|--------|-------------|-----|----------------|
| **1. Data structure** | Load, describe, structure report, graphs by year/month/hour, box plot by month, outlier list | Establish scale, frequency, missingness, and where level/variance/outliers sit | Level (MW), seasonality (hour, month), and outlier months/days drive model and policy choices |
| **2. Visualization** | Raw series, 30-day trend, hour-of-day seasonality, outlier count | Visual check of trend, seasonality, and extremes | Trend and seasonality guide model form; outliers may need treatment or explanation |
| **3. Decomposition** | Additive trend + seasonal + residual (4-panel plot) | Separate signal from noise and check seasonal shape | Trend = long-run path; seasonal = recurring pattern; residual = unpredictable shock |
| **3. Stationarity & ACF/PACF** | ADF, KPSS, ACF/PACF of (differenced) series | Validate stationarity and suggest ARIMA orders | Stationarity and lag structure justify SARIMA and seasonal order (e.g. 7) |
| **4. Modeling** | Train/test, SARIMA, diagnostics, 95% CI, SARIMAX (exog), Regression, SES, Holt’s, Holt–Winters, plots | Compare models and quantify uncertainty with a proper test set | Best specification and interval width inform forecasting and planning |
| **4. Rolling CV** | Rolling-origin CV, mean MAE ± std for ARIMA and Holt–Winters | Robust estimate of out-of-sample performance | Average 30-day-ahead error and its variability across origins |
| **5. Evaluation** | Metrics table (MAE, RMSE, MAPE, AIC/BIC), best model, actual vs all and vs best | Compare accuracy and parsimony, visualize fit | Best model and bias/accuracy by period support operational and policy use |

---

---

### Stage 5 — Insights (wrap-up)

**What we did:** Summarized the best model (by MAE), listed metrics for all models, and showed actual vs forecast plots.

**Why we did it:** To give a clear, actionable answer: which model to use and how accurate it is.

**Statistical inference:** The chosen model minimizes average absolute error on the 30-day holdout; AIC/BIC help balance fit and complexity when comparing specifications.

**Economic inference:** The best model is the one to use for **short-term operational forecasts** (scheduling, procurement, reserves). Prediction intervals support **risk and capacity planning**. Systematic bias or large errors in peaks suggest **re-estimation**, **extra regressors**, or **separate treatment of extreme days**.

---

## Issues & Improvements Needed

These items address methodological consistency, over-differencing risk, model diagnostics, and deployment readiness. Implementing them raises the project to production-grade (see **Updated Grade** below).

---

### 1. Inconsistent Data Usage (CRITICAL)

**Issue:** Stationarity checks (Stage 3) may use a different slice of data (e.g. last 2 years) than the series actually used for modeling (e.g. `train = y.iloc[:train_end]`). Conclusions about unit roots and seasonality should apply to the **same** data we fit the model on.

**Fix:** Run ADF and KPSS on the **actual training series** after defining `train` in Stage 4:

```python
# After defining train
print("\nStationarity Tests on Training Data:")
print("="*70)
adf_result = adfuller(train, autolag='AIC')
print("Augmented Dickey-Fuller test:")
print(f"  ADF Statistic: {adf_result[0]:.4f}")
print(f"  p-value: {adf_result[1]:.4f}")
print(f"  Critical values: {adf_result[4]}")
if adf_result[1] < 0.05:
    print("  ✓ Series is STATIONARY")
else:
    print("  ✗ Series is NON-STATIONARY (differencing needed)")

kpss_result = kpss(train, regression='ct', nlags='auto')
print("\nKPSS test:")
print(f"  KPSS Statistic: {kpss_result[0]:.4f}")
print(f"  p-value: {kpss_result[1]:.4f}")
if kpss_result[1] > 0.05:
    print("  ✓ Series is STATIONARY")
else:
    print("  ✗ Series is NON-STATIONARY")
```

---

### 2. Over-Differencing Risk

**Issue:** Using both regular differencing (`d=1`) and seasonal differencing (`D=1`) in SARIMA may over-difference the series and weaken forecasts.

**Fix:** Test whether seasonal differencing is needed on the **training** series:

```python
# Test seasonal differencing necessity
train_seasonal_diff = train.diff(7).dropna()
adf_seasonal = adfuller(train_seasonal_diff)
print(f"\nADF on seasonally differenced data: {adf_seasonal[1]:.4f}")
if adf_seasonal[1] < 0.05:
    print("  → Use D=1 (seasonal differencing needed)")
else:
    print("  → Use D=0 (seasonal differencing NOT needed)")
    # Try: seasonal_order=(1, 0, 1, 7) instead
```

**Recommendation:** Start with `(2,1,2)(1,0,1,7)` and only add `D=1` if the test supports it.

---

### 3. SARIMAX Variables Not Analyzed

**Issue:** Exogenous variables (e.g. `day_of_week`, `month`, `is_weekend`) are included in SARIMAX but their significance is not reported.

**Fix:** After fitting SARIMAX, print the full summary and check each exog coefficient:

```python
print("\n" + "-"*70)
print("SARIMAX Model Summary")
print("-"*70)
print(fit_sarimax.summary())

print("\nExogenous Variable Significance:")
print("-"*70)
params = fit_sarimax.params
pvalues = fit_sarimax.pvalues
for var in ['day_of_week', 'month', 'is_weekend']:
    if var in pvalues.index:
        p_val = pvalues[var]
        coef = params[var]
        sig = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
        print(f"  {var:15s}: coef={coef:8.4f}, p={p_val:.4f} {sig}")
```

---

### 4. Rolling CV Incomplete

**Issue:** Rolling forecast cross-validation only evaluates ARIMA and Holt–Winters; SARIMAX is missing.

**Fix:** Extend `rolling_forecast_cv` to support SARIMAX (build exog from the train/test index for each fold) and report mean MAE ± std for **ARIMA**, **SARIMAX**, and **Holt–Winters**. See the notebook for the full function with SARIMAX branch and exog construction per fold.

---

### 5. Missing Forecast Error Analysis

**Issue:** No breakdown of *where* forecasts fail (bias, worst over/under-estimation by date).

**Fix:** After computing forecasts, for each model report mean error (bias), std of errors, and dates of max overestimate and max underestimate; then plot forecast errors over time (Actual − Predicted) for ARIMA, SARIMAX, and Holt–Winters with a zero reference line.

---

### 6. Information Criteria Comparison

**Issue:** AIC/BIC are in the metrics table but not compared in one clear block.

**Fix:** Build a small table with Model, AIC, BIC (for ARIMA, SARIMAX, Holt–Winters), sort by AIC, and print “Best by AIC” and “Best by BIC” with a note that lower is better.

---

### 7. Final Recommendation Section

**Fix:** Add a **FINAL RECOMMENDATIONS** block that:

- Names the **best model by cross-validation** (and optionally by test MAE).
- Summarizes model characteristics (e.g. SARIMAX: calendar effects; ARIMA: parsimonious; Holt–Winters: simple and fast).
- Suggests a **deployment strategy**: retrain frequency, monitoring (e.g. alert if MAE > 2× current), use of prediction intervals, possible ensemble of top 2–3 models.
- Lists **known limitations**: no weather/holiday variables, assumption of stable seasonality, etc.
- Suggests **next steps**: weather/holiday regressors, structural break tests, forecast combination, automated retraining.

---

### Updated Grade

| Category | Original | Your Revision | With Fixes |
|----------|----------|---------------|------------|
| Structure & Documentation | 18/20 | 19/20 | 20/20 |
| Exploratory Analysis | 16/20 | 18/20 | 19/20 |
| Model Implementation | 14/20 | 18/20 | 20/20 |
| Diagnostics & Validation | 10/20 | 17/20 | 20/20 |
| Evaluation & Interpretation | 12/20 | 16/20 | 19/20 |
| Technical Correctness | 15/20 | 18/20 | 20/20 |
| **Overall** | **B+ (85/100)** | **A- (92/100)** | **A+ (98/100)** |

**Summary:** The main remaining improvements are: test stationarity on actual training data; check if seasonal differencing (D=1) is needed; analyze SARIMAX variable significance; add SARIMAX to rolling CV; add forecast error analysis; and add deployment recommendations. The notebook has been updated to implement these fixes where applicable.

---

You can use this document as a **narrative for your report or presentation**: for each section/graph, state what was done, why it was done, and what statistical and economic inference you draw from it. The file lives next to `Energy_Consumption_Forecasting.ipynb` in the same folder.
