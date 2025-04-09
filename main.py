import numpy as np
import pandas as pd
import os, sys, datetime, importlib
import utils
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
import joblib

# Reload utils in case of changes
importlib.reload(utils)
from utils import (plot_series, plot_series_with_names, plot_series_bar,
                   plot_dataframe, get_universe_adjusted_series, scale_weights_to_one, scale_to_book_long_short,
                   backtest, metrics_from_holdings)

#########################################
# LOAD DATA
#########################################

data_dir = "Learning/"

# Load features, returns, and universe data
# Here features is assumed to be a dict-like object with keys "f1", "f2", ... "f25"
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
returns  = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

#########################################
# DEFINE EXPECTED FEATURE NAMES
#########################################

all_features = [f"f{i}" for i in range(1, 26)]  # f1, f2, ..., f25

#########################################
# POOL DATA ACROSS ALL STOCKS AND TRAIN A SINGLE MODEL
#########################################

# Get list of stock identifiers from one of the feature DataFrames.
stocks = features[all_features[0]].columns

# Create a list to collect data from all stocks.
data_list = []

# Loop over each stock and build training samples.
for stock in stocks:
    # Build DataFrame X for the stock from each feature.
    X = pd.DataFrame({feat: features[feat][stock] for feat in all_features},
                     index=features[all_features[0]].index)
    # Ensure all expected columns are present and fill missing with 0.
    X = X.reindex(columns=all_features).fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Target y is the returns series for this stock.
    y = returns[stock].copy().fillna(0)
    y = y.replace([np.inf, -np.inf], 0)
    
    # Use yesterday's features to predict today's return.
    X_shifted = X.shift(1).iloc[1:]  # drop first row after shifting
    y_aligned = y.iloc[1:]
    
    # Align indices.
    common_idx = X_shifted.index.intersection(y_aligned.index)
    if common_idx.empty:
        continue
    X_shifted = X_shifted.loc[common_idx].fillna(0).replace([np.inf, -np.inf], 0)
    y_aligned = y_aligned.loc[common_idx].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Append the data with an added column 'target' to the list.
    stock_data = X_shifted.copy()
    stock_data['target'] = y_aligned
    data_list.append(stock_data)

# Concatenate data from all stocks.
if len(data_list) == 0:
    raise ValueError("No training data available from any stock.")
training_data = pd.concat(data_list, axis=0)

print("Combined training data shape:", training_data.shape)

# Separate features and target.
X_train = training_data[all_features]
y_train = training_data['target']

# Train a single Random Forest Regressor on the pooled data.
pooled_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
pooled_model.fit(X_train, y_train)

# Optionally save the trained pooled model.
joblib.dump(pooled_model, 'pooled_model.joblib')
print("Training complete. Pooled model trained on data from {} examples.".format(len(training_data)))


#########################################
# DEFINE THE GET_WEIGHTS FUNCTION
#########################################


def get_weights(features: pd.DataFrame, today_universe: pd.Series) -> dict:
    """
    Calculate stock weights for the portfolio on the current trading day.
    
    Vectorized implementation: builds a feature matrix (rows: tradable stocks, columns: features)
    and uses a single .predict call to get predictions for all stocks.
    
    Parameters:
    -----------
    features : pd.DataFrame
        A DataFrame whose keys are feature names (e.g., "f1", "f2", ..., "f25")
        and values are DataFrames with historical data.
        Each such DataFrame has:
          - Index: Datetime (chronological order).
          - Columns: Stock identifiers.
          
    today_universe : pd.Series
        A Series indicating the tradable stocks for the current day.
        - Index: Stock identifiers.
        - Values: 0 or 1 (1 means tradable).
    
    Returns:
    --------
    dict
        A dictionary where keys are stock identifiers (strings) and values are 
        the computed weights for the current trading day.
    """
    # If no feature data is available, return an empty dict.
    if features[all_features[0]].shape[0] == 0:
        return {}
    
    # Get tradable stocks from the universe.
    tradable_stocks = today_universe[today_universe == 1].index
    if len(tradable_stocks) == 0:
        return {}
    
    # Build a feature matrix for these stocks using the latest available row.
    feature_data = {
        feat: features[feat].iloc[-1].reindex(tradable_stocks).fillna(0)
        for feat in all_features
    }
    feature_matrix = pd.DataFrame(feature_data, index=tradable_stocks)
    feature_matrix = feature_matrix.replace([np.inf, -np.inf], 0)
    
    # Check if the feature matrix is empty.
    if feature_matrix.shape[0] == 0:
        return {}
    
    # Vectorized prediction using model for stock '1' (for testing).
    preds = trained_models['1'].predict(feature_matrix)
    # print("Predictions: ", preds)
    
    pred_series = pd.Series(preds, index=tradable_stocks)
    if pred_series.empty:
        return {}
    
    # Determine k for long/short selection.
    n_tradable = pred_series.shape[0]
    k_fixed = 5
    k = k_fixed if n_tradable >= 2 * k_fixed else n_tradable // 2
    if k < 1:
        k = 1
    
    longs = pred_series.nlargest(k)
    shorts = pred_series.nsmallest(k)
    
    # Remove any intersection (should not occur, but safety check)
    intersection = longs.index.intersection(shorts.index)
    longs = longs.drop(intersection)
    shorts = shorts.drop(intersection)
    
    # Assign equal weights.
    weights = pd.Series(0, index=pred_series.index, dtype=float)
    weight_value = 1 / (2 * k)
    weights.loc[longs.index] = weight_value
    weights.loc[shorts.index] = -weight_value
    
    # Enforce maximum weight constraint.
    weights = weights.clip(lower=-0.1, upper=0.1)
    
    # Ensure universe constraint.
    weights = weights.reindex(today_universe.index).fillna(0)
    
    # Scale weights for unit capital and dollar neutrality.
    weights = scale_to_book_long_short(weights)
    
    # Remove zeros and NaNs.
    weights = weights.replace(0, np.nan).dropna()
    
    # Return only tradable stocks.
    final_weights = {stock: w for stock, w in weights.to_dict().items() if today_universe.get(stock, 0) == 1}
    print("Returning final weights")

    return final_weights



#########################################
# BACKTESTING AND SUBMISSION
#########################################

positions, sr = backtest(
    get_weights,
    features,
    returns,
    universe,
    "2005-01-03",
    "2014-12-31",
    True,
    True
)

positions.to_csv("submission.csv")
nsr = metrics_from_holdings(positions, returns, universe)
print("Net Sharpe Ratio: ", nsr)
