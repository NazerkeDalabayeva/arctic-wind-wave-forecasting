# AI-Enabled Arctic Maritime Forecasting and Decision Support Under Uncertainty

Development of AI-enabled forecasting workflows for Arctic maritime operations using ERA5 atmospheric, wave, oceanographic and sea-ice datasets. The study investigates short-term environmental prediction and uncertainty quantification to support decision-making in ice-affected waters.

---

## Overview

This project investigates short-term forecasting of Arctic marine conditions relevant to maritime operations in ice-affected environments.

Using ERA5 reanalysis data from the Copernicus Climate Data Store, the project integrates atmospheric, oceanographic, wave, and sea-ice variables to explore machine learning approaches for forecasting environmental conditions under uncertainty.

---

## Research Motivation

Arctic maritime operations require reliable environmental forecasts to support navigation, operational planning, and risk management.

Rapid changes in wind, wave, and ice conditions can significantly affect vessel safety and accessibility. This project explores how machine learning methods can leverage environmental observations to forecast Arctic marine conditions at short time horizons.

The work is inspired by challenges related to Arctic navigation, environmental uncertainty, and decision-support systems.

---

## Study Area

A representative offshore location near Svalbard was selected:

- Latitude: 78.0°N
- Longitude: 10.0°E

The region is relevant to Arctic maritime operations and experiences highly variable meteorological and oceanographic conditions.

---

## Data Source

**ERA5 Reanalysis Data**

Source:
- Copernicus Climate Data Store (CDS)

Time Period:
- January 2024 to March 2024
- Hourly observations

---

## Variables

### Atmospheric Variables

- 10m U Wind Component (`u10`)
- 10m V Wind Component (`v10`)
- 2m Temperature (`t2m`)
- Mean Sea Level Pressure (`msl`)

### Ocean and Sea-Ice Variables

- Sea Surface Temperature (`sst`)
- Sea Ice Concentration (`siconc`)

### Wave Variables

- Significant Wave Height (`swh`)
- Mean Wave Period (`mwp`)
- Mean Wave Direction (`mwd`)

### Derived Variables

- Wind Speed

\[
WindSpeed = \sqrt{u10^2 + v10^2}
\]

---

## Methodology

### Data Processing

- NetCDF data ingestion using Xarray
- Spatio-temporal extraction of a Svalbard offshore location
- Variable engineering
- Time-series preprocessing

### Machine Learning Models

- Linear Regression
- XGBoost Regression

### Forecasting Task

Forecast horizon:

- 24 hours ahead

Target variable:

- Wind Speed

---

## Results

## Figures

### Wind Speed at Svalbard

./figures/wind_speed_svalbard.png

### Significant Wave Height at Svalbard

./figures/wave_height_svalbard.png

### Forecast Model Comparison

./figures/model_comparison.png

| Model | Mean Absolute Error (MAE) |
|---------|---------|
| Linear Regression | **3.43 m/s** |
| XGBoost | 3.85 m/s |

Linear Regression achieved the best forecasting accuracy for the selected forecasting horizon and study period.

The results suggest that relatively simple models can effectively capture a substantial share of the predictive signal within the available dataset.

---

## Key Findings

- Wind Speed and Significant Wave Height exhibited a strong positive correlation (~0.77).
- Wave conditions contained useful predictive information for Arctic wind forecasting.
- Linear Regression outperformed XGBoost on the selected dataset.
- Environmental forecasting performance was strongly influenced by the limited temporal coverage of the dataset.

---

## Repository Structure

```text
data/
├── raw/
└── processed/

figures/

src/
├── data_exploration.py
├── merge_data.py
├── linear_regression.py
├── xgboost_model.py
└── model_comparison.py

notebooks/
