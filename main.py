import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from sklearn.metrics import mean_squared_log_error

# 1. Load the training data
df_train = pd.read_csv('datasets/train.csv', sep='|', parse_dates=['date'])

# 2. Create the unique ID (product-channel combination) BEFORE renaming
df_train['unique_id'] = df_train['product_id'].astype(str) + '_' + df_train['channel_type'].astype(str)

# 3. Rename columns to MLForecast's expected format
df_train = df_train.rename(columns={
    'date': 'ds',                            # 'ds' stands for date-stamp
    'sales_quantity_sum': 'y'                # 'y' stands for the target variable
})

# 4. Filter to only the columns needed for training
df = df_train[['unique_id', 'ds', 'y']].copy()

print("Data Head:\n", df.head())
print("\nData Shape:", df.shape)
print("\nUnique IDs:", df['unique_id'].nunique())

# 5. Initialize and train the MLForecast model
models = [LGBMRegressor(verbose=-1)]  # verbose=-1 suppresses LightGBM output

fcst = MLForecast(
    models=models,
    freq='D',  # Daily frequency (adjust if needed based on your data)
    lags=[1, 7, 14, 30],  # Common lags for time series
    date_features=['dayofweek', 'month'],  # Extract date features
)

# Train the model
fcst.fit(df)

print("\nModel trained successfully!")

# 6. Function to evaluate on validation data
def evaluate_on_validation(validation_path):
    """
    Evaluate model performance on validation dataset using Mean Squared Log Error.
    
    Args:
        validation_path: Path to validation CSV file (should have same format as train.csv)
    
    Returns:
        Mean Squared Log Error score
    """
    print(f"\n{'='*60}")
    print("EVALUATING ON VALIDATION DATA")
    print(f"{'='*60}")
    
    # Load validation data
    df_val = pd.read_csv(validation_path, sep='|', parse_dates=['date'])
    
    # Create unique_id and rename columns (same format as training)
    df_val['unique_id'] = df_val['product_id'].astype(str) + '_' + df_val['channel_type'].astype(str)
    df_val = df_val.rename(columns={
        'date': 'ds',
        'sales_quantity_sum': 'y_true'  # Keep actual values as y_true
    })
    
    print(f"Validation data shape: {df_val.shape}")
    print(f"Unique IDs in validation: {df_val['unique_id'].nunique()}")
    
    # Calculate forecast horizon for each series
    last_train_dates = df.groupby('unique_id')['ds'].max()
    df_val_with_last = df_val.merge(
        last_train_dates.reset_index(), 
        on='unique_id', 
        suffixes=('_val', '_last_train')
    )
    df_val_with_last['horizon'] = (df_val_with_last['ds_val'] - df_val_with_last['ds_last_train']).dt.days
    
    max_horizon = int(df_val_with_last['horizon'].max())
    print(f"Maximum forecast horizon needed: {max_horizon} days")
    
    # Make predictions
    print("\nMaking predictions on validation data...")
    predictions = fcst.predict(h=max_horizon)
    
    # Merge predictions with actual values
    df_val_with_preds = df_val.merge(
        predictions[['unique_id', 'ds', 'LGBMRegressor']],
        on=['unique_id', 'ds'],
        how='left'
    )
    
    # Fill NaN predictions with 0 (for series that were dropped during training)
    df_val_with_preds['LGBMRegressor'] = df_val_with_preds['LGBMRegressor'].fillna(0)
    
    # Ensure predictions are non-negative
    df_val_with_preds['LGBMRegressor'] = df_val_with_preds['LGBMRegressor'].clip(lower=0)
    
    # Filter to rows where we have both predictions and actual values
    df_eval = df_val_with_preds.dropna(subset=['y_true', 'LGBMRegressor'])
    
    if len(df_eval) == 0:
        print("WARNING: No matching predictions found for validation data!")
        return None
    
    # Calculate Mean Squared Log Error
    # MSLE = mean((log(1 + y_true) - log(1 + y_pred))^2)
    # Note: sklearn's mean_squared_log_error requires non-negative values
    y_true = df_eval['y_true'].values
    y_pred = df_eval['LGBMRegressor'].values
    
    # Ensure non-negative (should already be done, but double-check)
    y_pred = np.maximum(y_pred, 0)
    y_true = np.maximum(y_true, 0)
    
    msle = mean_squared_log_error(y_true, y_pred)
    
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Number of predictions evaluated: {len(df_eval):,}")
    print(f"Mean Squared Log Error (MSLE): {msle:.6f}")
    print(f"{'='*60}\n")
    
    return msle

# 7. Function to make predictions on test data for submission
def make_submission_predictions(test_path='datasets/test.csv', output_path='submission.csv'):
    """
    Make predictions on test data and save to submission file.
    
    Args:
        test_path: Path to test CSV file
        output_path: Path to save submission file
    """
    print(f"\n{'='*60}")
    print("MAKING PREDICTIONS FOR SUBMISSION")
    print(f"{'='*60}")
    
    # Load test data
    df_test = pd.read_csv(test_path, sep='|', parse_dates=['date'])
    
    # Create unique_id and rename columns
    df_test['unique_id'] = df_test['product_id'].astype(str) + '_' + df_test['channel_type'].astype(str)
    df_test = df_test.rename(columns={'date': 'ds'})
    
    print(f"Test data shape: {df_test.shape}")
    print(f"Unique IDs in test: {df_test['unique_id'].nunique()}")
    
    # Calculate forecast horizon
    last_train_dates = df.groupby('unique_id')['ds'].max()
    df_test_with_last = df_test.merge(
        last_train_dates.reset_index(), 
        on='unique_id', 
        suffixes=('_test', '_last_train')
    )
    df_test_with_last['horizon'] = (df_test_with_last['ds_test'] - df_test_with_last['ds_last_train']).dt.days
    
    max_horizon = int(df_test_with_last['horizon'].max())
    print(f"Maximum forecast horizon needed: {max_horizon} days")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = fcst.predict(h=max_horizon)
    
    # Merge predictions with test data
    df_test_with_preds = df_test.merge(
        predictions[['unique_id', 'ds', 'LGBMRegressor']],
        on=['unique_id', 'ds'],
        how='left'
    )
    
    # Fill NaN predictions with 0
    df_test_with_preds['LGBMRegressor'] = df_test_with_preds['LGBMRegressor'].fillna(0)
    
    # Ensure predictions are non-negative
    df_test_with_preds['LGBMRegressor'] = df_test_with_preds['LGBMRegressor'].clip(lower=0)
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'product_id': df_test_with_preds['product_id'],
        'channel_type': df_test_with_preds['channel_type'],
        'week_start_date': df_test_with_preds['ds'].dt.strftime('%Y-%m-%d'),
        'prediction': df_test_with_preds['LGBMRegressor'].round().astype(int)
    })
    
    # Save submission file
    submission.to_csv(output_path, sep='|', index=False)
    print(f"\nSubmission saved to {output_path}!")
    print(f"Submission shape: {submission.shape}")
    print(f"Submission Head:\n{submission.head()}")
    print(f"{'='*60}\n")
    
    return submission

make_submission_predictions()

