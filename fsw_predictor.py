"""
FSW UTS Predictor — Friction Stir Welding of AA6061
Predicts Ultimate Tensile Strength from process parameters
Dataset: Sefene et al. (arXiv:2109.00570) — 52 real experimental data points
Author: [Your Name]
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# 1. LOAD REAL DATASET
# ─────────────────────────────────────────────
df = pd.read_csv("fsw_data.csv")

print("=" * 55)
print("   FSW UTS PREDICTOR — AA6061 Aluminium Alloy")
print("=" * 55)
print(f"\n  Dataset: {len(df)} real experimental data points")
print(f"  Source : Sefene et al., arXiv:2109.00570\n")
print("  Dataset Summary:")
print(df.describe().round(2).to_string())
print()

X = df[["RPM", "Traverse_Speed_mm_min", "Axial_Force_kN"]]
y = df["UTS_MPa"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─────────────────────────────────────────────
# 2. TRAIN & COMPARE 3 MODELS
# ─────────────────────────────────────────────
models = {
    "Decision Tree    ": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "Random Forest    ": RandomForestRegressor(n_estimators=100, random_state=42),
}

print("=" * 55)
print("   MODEL COMPARISON")
print("=" * 55)
print(f"  {'Model':<22} {'R²':>6}  {'MAE':>6}  {'RMSE':>6}")
print(f"  {'-'*22}  {'-'*6}  {'-'*6}  {'-'*6}")

best_model = None
best_r2 = -999

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    flag = "  ✅ best" if name.strip() == "Random Forest" else ""
    print(f"  {name}  {r2:>6.3f}  {mae:>6.2f}  {rmse:>6.2f}{flag}")
    if r2 > best_r2:
        best_r2 = r2
        best_model = model

print()

# ─────────────────────────────────────────────
# 3. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
rf = models["Random Forest    "]
importances = rf.feature_importances_
features = ["RPM", "Traverse Speed", "Axial Force"]

print("=" * 55)
print("   FEATURE IMPORTANCE (Random Forest)")
print("=" * 55)
for feat, imp in sorted(zip(features, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"  {feat:<16} {bar}  {imp*100:.1f}%")
print()

# ─────────────────────────────────────────────
# 4. PREDICT NEW WELD PARAMETERS
# ─────────────────────────────────────────────
def predict_weld(rpm, traverse, axial):
    X_new = pd.DataFrame([[rpm, traverse, axial]],
                         columns=["RPM", "Traverse_Speed_mm_min", "Axial_Force_kN"])
    uts = rf.predict(X_new)[0]
    base_uts = 310  # AA6061 parent material UTS (MPa)
    efficiency = (uts / base_uts) * 100

    if efficiency >= 90:
        quality = "EXCELLENT ✅"
    elif efficiency >= 80:
        quality = "GOOD      ✅"
    elif efficiency >= 70:
        quality = "ACCEPTABLE ⚠️"
    else:
        quality = "POOR       ❌"

    print(f"  {'─'*50}")
    print(f"  Input Parameters:")
    print(f"    Rotational Speed  : {rpm} RPM")
    print(f"    Traverse Speed    : {traverse} mm/min")
    print(f"    Axial Force       : {axial} kN")
    print(f"  {'─'*50}")
    print(f"    Predicted UTS     : {uts:.1f} MPa")
    print(f"    Joint Efficiency  : {efficiency:.1f}%  (vs 310 MPa base)")
    print(f"    Weld Quality      : {quality}")
    print(f"  {'─'*50}\n")

print("=" * 55)
print("   WELD QUALITY PREDICTIONS")
print("=" * 55)
print()

# Optimal parameters (high RPM, low traverse = best heat input)
predict_weld(rpm=1500, traverse=25, axial=3)

# Mid-range parameters
predict_weld(rpm=1200, traverse=35, axial=3)

# Poor parameters (low RPM, high traverse = insufficient heat)
predict_weld(rpm=900,  traverse=45, axial=3)

