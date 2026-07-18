# arctic-wind-wave-forecasting
Machine learning forecasting of Arctic marine conditions under uncertainty using ERA5 environmental data
# Arctic Marine Forecasting Under Uncertainty

## Overview

This project investigates short-term forecasting of Arctic marine conditions using ERA5 reanalysis data and machine learning.
The objective is to forecast environmental variables relevant to Arctic maritime operations under uncertainty.

## Research Motivation
Safe navigation in Arctic waters requires reliable forecasts of environmental conditions.
Rapid changes in wind, waves, and sea-ice conditions can affect navigability and operational decision-making. This project explores machine-learning approaches for Arctic marine forecasting using environmental observations from ERA5.

## Data
Source:
- ERA5 Reanalysis Data (Copernicus Climate Data Store)
Variables:
- 10m u-component of wind
- 10m v-component of wind
- Mean sea level pressure
- 2m temperature
- Significant wave height
- Mean wave direction
- Mean wave period
- Peak wave period
- Sea surface temperature
- Sea-ice cover

## Forecast Horizons
- 24-hour forecast
- 48-hour forecast
- 72-hour forecast

## Methods
- Feature Engineering
- Linear Regression
- XGBoost
- Time-Series Analysis

## Repository Structure
data/
notebooks/
src/
models/
figures/

## Future Work
- Sea-ice forecasting
- Uncertainty quantification
- Hybrid AI-physics forecasting
- Arctic decision-support applications

## Results

Linear Regression MAE: 3.43 m/s
XGBoost MAE: 3.85 m/s. 
Linear Regression achieved the best performance for 24-hour wind speed forecasting at an offshore location near Svalbard.
