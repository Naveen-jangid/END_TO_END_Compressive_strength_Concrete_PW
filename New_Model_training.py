import pandas as pd
import time
import logging
import argparse
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configure Logging
LOG_FILE = "model_training.log"
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

def load_data(file_path):
    """Load dataset from file."""
    try:
        logging.info(f"📂 Loading dataset from {file_path}...")
        df = pd.read_csv(file_path)
        
        # Ensure target column exists
        if "concrete_compressive_strength" not in df.columns:
            raise ValueError("❌ Dataset missing 'concrete_compressive_strength' column!")
        
        logging.info("✅ Dataset loaded successfully.")
        return df
    except Exception as e:
        logging.error(f"❌ Error loading dataset: {e}")
        raise

def split_data(df):
    """Split dataset into train, validation, and test sets."""
    try:
        logging.info("📊 Splitting dataset into training, validation, and test sets...")
        X = df.drop(columns=["concrete_compressive_strength"])
        y = df["concrete_compressive_strength"]
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)
        logging.info("✅ Data split successfully.")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        logging.error(f"❌ Error splitting dataset: {e}")
        raise

def train_and_evaluate(models, X_train, y_train, X_val, y_val):
    """Train and evaluate multiple models."""
    logging.info("🚀 Training and evaluating models...")
    model_results = {}
    
    for name, model in models.items():
        try:
            start_time = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            end_time = time.time()
            
            mse = mean_squared_error(y_val, y_pred)
            rmse = mse ** 0.5
            r2 = r2_score(y_val, y_pred)
            latency = round(end_time - start_time, 3)

            model_results[name] = {"MSE": mse, "RMSE": rmse, "R² Score": r2, "Latency (s)": latency}
            logging.info(f"✅ {name} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, R² Score: {r2:.4f}, Latency: {latency}s")
        except Exception as e:
            logging.error(f"❌ Error training {name}: {e}")
    
    return pd.DataFrame(model_results).T

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for Random Forest and save best model."""
    try:
        logging.info("🔍 Performing hyperparameter tuning for Random Forest...")
        param_grid = {"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15]}
        rf_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Save best model
        joblib.dump(best_model, "best_rf_model.pkl")
        logging.info(f"✅ Best RF Params: {grid_search.best_params_}, Best RF R² Score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_
    except Exception as e:
        logging.error(f"❌ Error in hyperparameter tuning: {e}")
        raise

def is_jupyter_notebook():
    """Detect if the script is running in a Jupyter Notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

if __name__ == "__main__":
    # Check execution environment
    if is_jupyter_notebook():
        logging.info("🛠 Running in Jupyter Notebook. Using default dataset.")
        data_path = "concrete_data.csv"
    else:
        parser = argparse.ArgumentParser(description="Train ML models on concrete dataset.")
        parser.add_argument("--data", type=str, default="concrete_data.csv", help="Path to dataset")
        args = parser.parse_args()
        data_path = args.data

    try:
        # Load and split data
        df = load_data(data_path)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

        # Define models
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(objective="reg:squarederror", n_estimators=50, random_state=42)
        }

        # Train & evaluate
        results_df = train_and_evaluate(models, X_train, y_train, X_val, y_val)
        print("\n📊 Model Evaluation Results:\n", results_df)

        # Hyperparameter tuning
        best_params, best_score = hyperparameter_tuning(X_train, y_train)
        print("\n🏆 Best RF Params:", best_params)
        print("🎯 Best RF R² Score:", best_score)

    except Exception as e:
        logging.error(f"❌ Critical Error in Main Execution: {e}")
        print(f"❌ Error: {e}")
        

