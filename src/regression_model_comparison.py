# Activity 16 - Regression Model Comparison
# Team 4 - Biomedical ICU dataset, target = cardiac_output_l_min

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/regression_team4_biomedical.csv")

X = df.drop(columns=["cardiac_output_l_min"])
y = df["cardiac_output_l_min"]

# same split for every model (random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# we add the features one by one, starting with the ones that matter most
# (cardiac output = heart rate x stroke volume)
models = {
    "Model 1": ["heart_rate_bpm"],
    "Model 2": ["heart_rate_bpm", "stroke_volume_ml"],
    "Model 3": ["heart_rate_bpm", "stroke_volume_ml", "systemic_vascular_resistance"],
    "Model 4": ["heart_rate_bpm", "stroke_volume_ml", "systemic_vascular_resistance",
                "mean_arterial_pressure_mmhg"],
    "Model 5": ["heart_rate_bpm", "stroke_volume_ml", "systemic_vascular_resistance",
                "mean_arterial_pressure_mmhg", "systolic_bp_mmhg"],
}

results = []
predictions = {}

for name, features in models.items():
    model = LinearRegression()
    model.fit(X_train[features], y_train)

    pred_train = model.predict(X_train[features])
    pred_test = model.predict(X_test[features])
    predictions[name] = pred_test

    r2_train = r2_score(y_train, pred_train)
    r2_test = r2_score(y_test, pred_test)

    # adjusted R2 = 1 - (1 - R2) * (n - 1) / (n - p - 1)
    n = len(y_test)
    p = len(features)
    adj_r2 = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)

    mae = mean_absolute_error(y_test, pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred_test))

    results.append([name, p, r2_train, r2_test, adj_r2, mae, rmse])

results = pd.DataFrame(results, columns=["Model", "Nb features", "Train R2", "Test R2",
                                         "Adjusted R2", "MAE", "RMSE"])
print(results.round(4))
results.round(4).to_csv("reports/activity16_model_comparison.csv", index=False)

# residual plots for the 2 best models (best adjusted R2)
best_two = results.sort_values("Adjusted R2", ascending=False)["Model"].head(2)

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

for i, name in enumerate(best_two):
    residuals = y_test - predictions[name]

    axes[i, 0].scatter(predictions[name], residuals, alpha=0.6)
    axes[i, 0].axhline(0, color="red", linestyle="--")
    axes[i, 0].set_title(name + " - Fitted vs Residuals")
    axes[i, 0].set_xlabel("Fitted values")
    axes[i, 0].set_ylabel("Residuals")

    stats.probplot(residuals, dist="norm", plot=axes[i, 1])
    axes[i, 1].set_title(name + " - Q-Q plot")

plt.tight_layout()
plt.savefig("reports/figures/activity16_residual_diagnostics.png")
plt.show()
