import joblib
model = joblib.load("best_rf_model.pkl")
print(type(model))  # Should print <class 'sklearn.ensemble._forest.RandomForestRegressor'>

# Check if the model has a predict method
if hasattr(model, "predict"):
    print("Model is correctly saved and has a predict method.")
else:
    print("Error: Model does not have a predict method.")

