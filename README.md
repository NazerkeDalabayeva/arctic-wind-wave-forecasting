# AI-Enabled Arctic Maritime Forecasting and Decision Support Under Uncertainty

Development of AI-enabled forecasting workflows for Arctic maritime operations using ERA5 atmospheric, oceanographic, wave, and sea-ice datasets. The project investigates short-term environmental prediction and uncertainty quantification to support decision-making in ice-affected waters and the Arctic Marginal Ice Zone (MIZ).

---

## Overview

This project explores machine learning approaches for forecasting Arctic sea-ice conditions and environmental variability using ERA5 reanalysis data.

Atmospheric, oceanographic, wave, and sea-ice variables were combined to create an integrated forecasting framework for Arctic maritime operations. In addition to predictive modeling, uncertainty quantification techniques were implemented to assess forecast confidence during major sea-ice events.

---

## Research Motivation

Arctic maritime operations are increasingly influenced by changing environmental conditions and growing marine activity in ice-affected waters.

Accurate forecasts of sea-ice concentration and associated environmental factors are essential for:

- Navigation safety
- Route planning
- Operational risk assessment
- Maritime decision support
- Arctic offshore activities

This project investigates how machine learning can support forecasting under uncertainty in dynamically changing Arctic environments.

---

## Study Area

Representative Marginal Ice Zone (MIZ) location near Svalbard:

- Latitude: 76.0°N
- Longitude: 15.0°E

This region experiences interactions among sea ice, atmosphere, ocean conditions, and wave activity, making it relevant for Arctic maritime operations.

---

## Data Source

### ERA5 Reanalysis Data

Source:

- Copernicus Climate Data Store (CDS)

Period:

- January 2024 to June 2024
- Hourly observations

Total observations:

- 4,344 hourly records

---

## Variables

### Atmospheric Variables

- 10 m U Wind Component (`u10`)
- 10 m V Wind Component (`v10`)
- 2 m Temperature (`t2m`)
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

```text
Wind Speed = √(u10² + v10²)
```

---

## Methodology

### Data Processing

- NetCDF ingestion using Xarray
- Integration of weather and wave datasets
- Spatial extraction of an Arctic MIZ location
- Feature engineering
- Time-series preprocessing

### Forecasting Task

Target:

- Sea-Ice Concentration (`siconc`)

Forecast Horizon:

- 24 Hours Ahead

### Machine Learning Models

- Linear Regression
- XGBoost Regression

### Uncertainty Quantification

- Bootstrap ensemble forecasting
- Prediction interval estimation
- Decision-tree ensemble approach

---

## Results

### Sea-Ice Forecasting Performance

| Model | MAE |
|---------|---------|
| Linear Regression | **0.0004** |
| XGBoost | 0.0013 |

Linear Regression achieved the lowest overall forecasting error for the selected forecasting horizon.

XGBoost demonstrated the ability to capture major sea-ice transitions and ice-event dynamics.

---

## Key Findings

- Sea-surface temperature exhibited a strong negative relationship with sea-ice concentration (**r = -0.69**).
- Air temperature also showed a notable negative relationship with sea ice (**r = -0.41**).
- Wind speed and significant wave height showed a strong positive relationship (**r = 0.79**).
- Current sea-ice concentration was the most important predictor of future sea-ice conditions.
- Forecast uncertainty increased during rapid sea-ice transitions and major ice events.

---

## Figures

### Arctic Sea-Ice Forecasting Model Comparison

./figures/sea_ice_model_comparison.png

### Sea-Ice Forecast Uncertainty During a Major Ice Event

./figures/forecast_uncertainty.png

### Arctic Environmental Correlation Matrix

./figures/environmental_correlation_matrix.png

### Arctic Wind Speed Time Series

./figures/arctic_wind_speed_timeseries.png

### Arctic Significant Wave Height Time Series

./figures/arctic_wave_height_timeseries.png

---

## Environmental Relationships

The analysis revealed several physically meaningful environmental relationships:

| Relationship | Correlation |
|--------------|-------------|
| Sea Ice vs SST | -0.69 |
| Sea Ice vs Temperature | -0.41 |
| Wind Speed vs Wave Height | 0.79 |
| Wave Height vs Wave Period | 0.63 |

These relationships demonstrate the interconnected nature of Arctic atmosphere-ocean-sea-ice systems.

---

## Repository Structure

data/
├── raw/
└── processed/

figures/
├── arctic_wind_speed_timeseries.png
├── arctic_wave_height_timeseries.png
├── sea_ice_model_comparison.png
├── forecast_uncertainty.png
└── environmental_correlation_matrix.png

src/
├── data_exploration.py
├── merge_data.py
├── sea_ice_linear_regression.py
├── sea_ice_xgboost.py
├── sea_ice_model_comparison.py
└── uncertainty_forecasting.py

---

## Technologies

- Python
- Pandas
- NumPy
- Xarray
- Matplotlib
- Scikit-Learn
- XGBoost

---

## Future Work

- Multi-step sea-ice forecasting
- Longer ERA5 time series
- Physics-informed machine learning
- Explainable AI for Arctic forecasting
- Operational Arctic decision-support systems
- Hybrid AI-physics forecasting frameworks

---

## Relevance to Arctic Maritime Operations

This project demonstrates how machine learning and uncertainty quantification can be applied to Arctic environmental forecasting using integrated atmosphere-ocean-sea-ice datasets.

The developed workflow provides a foundation for future research on Arctic maritime decision support, sea-ice prediction, uncertainty-aware forecasting, and AI-enabled operational planning in ice-affected waters.
