# Energy forecasting data

Place PJM (or other) energy CSV files here, or use `~/Downloads/archive/`. Both folders are searched.

## What you have (checked)

| Granularity | Source | Status |
|-------------|--------|--------|
| **Hourly**  | CSVs in `data/` or `~/Downloads/archive/` | ✅ **13 hourly CSVs** in archive (e.g. PJME_hourly, PJMW_hourly, AEP_hourly, pjm_hourly_est, …). All are loaded; pick one via `series_name`. |
| **Daily**   | Not separate files | Derived in code from hourly: `df_daily = df.resample('1D').mean()`. |
| **Yearly**  | In code            | Built from hourly: `df_yearly = df.resample('1Y').mean()`. Stored in `databases[key]['yearly']` and as `df_yearly`. |

The data are **hourly** CSVs. Daily and yearly series are built from hourly in the notebook.

## Usage

- **Notebook:** set `series_name = 'PJME_hourly'` (or any key from the list) to choose which series to forecast.
- **Script:** `python run_energy_forecast.py` or `python run_energy_forecast.py PJMW_hourly`.
