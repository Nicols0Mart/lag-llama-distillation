# %% [markdown]
# # GluonTS MultivariateGrouper Example
# 
# This notebook demonstrates how to use the MultivariateGrouper object from GluonTS for a multivariate time series dataset.

# %%
# Install required libraries
# !pip install gluonts pandas numpy matplotlib

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.torch import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import MultivariateEvaluator

# %% [markdown]
# ## Step 1: Prepare the Dataset

# %%
# Generate sample multivariate time series data
num_series = 3
num_timesteps = 1000
freq = 'H'

dates = pd.date_range(start='2021-01-01', periods=num_timesteps, freq=freq)
series_data = np.random.randn(num_timesteps, num_series)

df = pd.DataFrame(series_data, index=dates, columns=[f'series_{i}' for i in range(num_series)])
print(df.head())

# Plot the time series
df.plot(figsize=(12, 6))
plt.title('Multivariate Time Series Data')
plt.show()

# %% [markdown]
# ## Step 2: Prepare GluonTS Dataset

# %%
# Convert DataFrame to GluonTS dataset format
def dataframe_to_gluonts_dataset(df: pd.DataFrame, freq: str) -> ListDataset:
    return ListDataset(
        [
            {
                "start": df.index[0],
                "target": df[col].values,
                "item_id": col
            } for col in df.columns
        ],
        freq=freq
    )

dataset = dataframe_to_gluonts_dataset(df, freq)

# Split the dataset into training and test sets
train_length = int(0.8 * len(df))
train_data = dataframe_to_gluonts_dataset(df.iloc[:train_length], freq)
test_data = dataframe_to_gluonts_dataset(df.iloc[train_length:], freq)

# %% [markdown]
# ## Step 3: Use MultivariateGrouper

# %%
# Create MultivariateGrouper object
grouper = MultivariateGrouper(max_target_dim=num_series)

# Apply grouper to the datasets
train_data_grouped = grouper(train_data)
test_data_grouped = grouper(test_data)

print(f"Number of grouped time series: {len(train_data_grouped)}")

# %% [markdown]
# ## Step 4: Train a Model

# %%
# Set up the estimator
estimator = DeepAREstimator(
    prediction_length=24,
    context_length=24 * 7,  # use one week of context
    freq=freq,
    trainer=Trainer(epochs=10),
    num_layers=2,
    num_cells=40,
    target_dim=num_series
)

# Train the model
predictor = estimator.train(train_data_grouped)

# %% [markdown]
# ## Step 5: Make Predictions and Evaluate

# %%
# Make predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_data_grouped,
    predictor=predictor,
    num_samples=100
)
forecasts = list(forecast_it)
tss = list(ts_it)

# Evaluate the model
evaluator = MultivariateEvaluator(quantiles=[0.1, 0.5, 0.9])
agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=num_series)

print(f"Mean CRPS: {agg_metrics['mean_wQuantileLoss']}")
print(f"MSE: {agg_metrics['MSE']}")

# %% [markdown]
# ## Step 6: Visualize Predictions

# %%
def plot_prob_forecasts(ts_entry, forecast_entry):
    plot_length = 150
    prediction_intervals = (50.0, 90.0)
    legend = ['observations', 'median prediction'] + [f'{k}% prediction interval' for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ts_entry[-plot_length:].plot(ax=ax)
    forecast_entry.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which='both')
    plt.legend(legend, loc='upper left')
    plt.show()

# Plot predictions for each series
for i in range(num_series):
    ts_entry = tss[0][0]['target'][:, i]
    forecast_entry = forecasts[0].copy_dim(i)
    plot_prob_forecasts(ts_entry, forecast_entry)


