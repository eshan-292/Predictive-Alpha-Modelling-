import numpy as np
import pandas as pd
import os, sys, datetime, importlib
import utils
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from joblib import Parallel, delayed
import joblib

# Reload utils in case of changes
importlib.reload(utils)
from utils import (plot_series, plot_series_with_names, plot_series_bar,
                   plot_dataframe, get_universe_adjusted_series, scale_weights_to_one, scale_to_book_long_short,
                   backtest, metrics_from_holdings)

#########################################
# LOAD DATA
#########################################

# data_dir = "Learning/"
data_dir = "Validation/"

# Load features, returns, and universe data
# Here features is assumed to be a dict-like object with keys "f1", "f2", ... "f25"
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
returns  = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))

print("Features shape: ", features.shape)
exit()

#########################################
# DEFINE EXPECTED FEATURE NAMES
#########################################

all_features = [f"f{i}" for i in range(1, 26)]  # f1, f2, ..., f25

#########################################
# TRAIN RANDOM FOREST MODELS FOR EACH STOCK (PARALLELIZED)
#########################################

def train_model(stock):
    """
    Trains a Random Forest regressor for the given stock using its historical features
    (shifted by one day) to predict today's return.
    
    Returns a tuple (stock, trained_model).
    """
    # Build DataFrame X for the stock from each feature.
    X = pd.DataFrame({feat: features[feat][stock] for feat in all_features},
                     index=features[all_features[0]].index)
    X = X.reindex(columns=all_features).fillna(0)
    X = X.replace([np.inf, -np.inf], 0)
    
    # Target y is the returns series for the stock.
    y = returns[stock].copy().fillna(0)
    y = y.replace([np.inf, -np.inf], 0)
    
    # Use yesterday's features to predict today's return.
    X_shifted = X.shift(1).iloc[1:]  # Drop first row (NaN after shifting)
    y_aligned = y.iloc[1:]
    
    # Align indices.
    common_idx = X_shifted.index.intersection(y_aligned.index)
    X_shifted = X_shifted.loc[common_idx].fillna(0).replace([np.inf, -np.inf], 0)
    y_aligned = y_aligned.loc[common_idx].fillna(0).replace([np.inf, -np.inf], 0)
    
    if X_shifted.empty or y_aligned.empty:
        # Skip this stock if no training data is available.
        return stock, None
    
    # Train Random Forest Regressor.
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_shifted, y_aligned)
    
    return stock, rf



# # Get list of stock identifiers from one of the feature DataFrames.
# stocks = features[all_features[0]].columns

# print("Training Random Forest models for each stock in parallel...")


# # Parallelize training using all available CPU cores.
# trained_results = Parallel(n_jobs=-1)(delayed(train_model)(stock) for stock in stocks)


# # Build the dictionary of trained models (skip any None models).
# trained_models = {stock: model for stock, model in trained_results if model is not None}

# # save the trained models

# joblib.dump(trained_models, 'trained_models.joblib')



# load the trained models
trained_models = joblib.load('trained_models.joblib')

print("Training complete. Trained models for {} stocks.".format(len(trained_models)))

#########################################
# DEFINE THE GET_WEIGHTS FUNCTION (VECTORIZED)
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
    
    if feature_matrix.shape[0] == 0:
        return {}
    
    # preds = trained_models[stock].predict(feature_matrix)
    # Vectorized prediction using the model for stock '1' (for testing).
    preds = trained_models['1'].predict(feature_matrix)

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
    
    # Remove any intersection (safety check).
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
    
    # Scale weights to satisfy unit capital and dollar neutrality.
    weights = scale_to_book_long_short(weights)
    
    # Remove zeros and NaNs.
    weights = weights.replace(0, np.nan).dropna()
    
    # Filter out any stocks that are not tradable (for safety).
    final_weights = {stock: w for stock, w in weights.to_dict().items() if today_universe.get(stock, 0) == 1}
    
    return final_weights

#########################################
# BACKTESTING AND SUBMISSION
#########################################

# positions, sr = backtest(
#     get_weights,
#     features,
#     returns,
#     universe,
#     "2005-01-03",
#     "2014-12-31",
#     True,
#     True
# )




#########################################
# PARALLELIZED BACKTEST FUNCTION
#########################################

def parallel_backtest(contestant_get_weights, entire_features, returns, universe, start_date, end_date):
    """
    Parallelized backtest that computes daily portfolio positions using the provided
    contestant_get_weights function in parallel across trading days.
    """
    # Validate date formats.
    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt   = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    cutoff_date = datetime.datetime.strptime('2005-01-01', '%Y-%m-%d')
    
    if start_dt >= end_dt:
        raise ValueError("start_date must be earlier than end_date.")
    if start_dt < cutoff_date:
        raise ValueError("start_date must be later than '2005-01-01'.")
    
    trading_days = universe.index[(universe.index >= start_dt) & (universe.index <= end_dt)]
    if len(trading_days) == 0:
        raise ValueError("No Trading Days in the specified dates")
    
    # Helper function to compute positions for a single day.
    def get_daily_positions(day):
        current_universe = universe.loc[day]
        # Filter features to only include data before the current day.
        filtered_features = entire_features[entire_features.index < day]
        # Get portfolio weights.
        weights_dict = contestant_get_weights(filtered_features, current_universe)
        current_weights = pd.Series(weights_dict)
        
        # Constraint checks.
        stocks_with_weight = set(current_weights.keys())
        stocks_not_in_universe = set(current_universe[current_universe == 0].index.tolist())
        if len(stocks_with_weight & stocks_not_in_universe) != 0:
            raise ValueError(f"Your returned weights dictionary has a stock which is not in the universe on {day}")
        if abs(current_weights.sum()) > 1e-2:
            raise ValueError(f"On {day}, Dollar Neutral Constraint is violated")
        if (current_weights.abs().sum() - 1) > 1e-2:
            raise ValueError(f"On {day}, Unit Capital Constraint is violated")
        if current_weights.abs().max() > 0.1:
            raise ValueError(f"On {day}, Maximum Weight Constraint is violated")
        
        # Return the positions for the day.
        return day, current_weights.reindex(current_universe.index, fill_value=np.nan)
    
    # Parallelize over all trading days.
    results = Parallel(n_jobs=-1)(delayed(get_daily_positions)(day) for day in trading_days)
    
    # Assemble results into a DataFrame.
    holdings = pd.DataFrame({day: weights for day, weights in results}).T
    holdings = holdings.sort_index()
    
    # Fill any missing positions with 0.
    holdings = holdings.fillna(0)
    
    # Continue with the backtest as before.
    rets = returns.loc[trading_days].fillna(0)
    gross_pnl = (holdings * rets).sum(axis=1)
    traded = holdings.diff(1).abs().sum(axis=1).fillna(0)
    book_value = holdings.abs().sum(axis=1)
    turnover = (traded.mean() / book_value.mean()) * 100
    net_pnl = gross_pnl - traded * 1e-4
    gross_sharpe_ratio = (gross_pnl.mean() / gross_pnl.std()) * np.sqrt(252)
    net_sharpe_ratio = (net_pnl.mean() / net_pnl.std()) * np.sqrt(252)
    
    print("Gross Sharpe Ratio: ", round(gross_sharpe_ratio, 3))
    print("Net Sharpe Ratio: ", round(net_sharpe_ratio, 3))
    print("Turnover %: ", round(turnover, 3))
    
    try:
        plot_series_with_names([gross_pnl.cumsum(), net_pnl.cumsum()],
                                ["Gross PnL", "Net PnL"],
                                "Cumulative PnL Plot",
                                yaxis_title="PnL", xaxis_title="Date")
    except Exception as e:
        print("Plotting error: ", e)
    
    return holdings, round(net_sharpe_ratio, 3)


positions, sr = parallel_backtest(
    get_weights,
    features,
    returns,
    universe,
    "2005-01-03",
    "2019-12-31"
)

positions.to_csv("submission.csv")
nsr = metrics_from_holdings(positions, returns, universe)
print("Net Sharpe Ratio: ", nsr)



