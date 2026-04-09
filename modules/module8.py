################################# 8 ###################################

import pandas as pd

data = [
    ["1984", "Chevrolet", "Corvette", 18, 1.255272505, 324, "$1,600.00", 3.204119983],
    ["1956", "Chevrolet", "Corvette 265/225-hp", 19, 1.278753601, 361, "$4,000.00", 3.602059991],
    ["1963", "Chevrolet", "Corvette coupe (340-bhp 4-speed)", 18, 1.255272505, 324, "$1,000.00", 3.0],
    ["1978", "Chevrolet", "Corvette coupe Silver Anniversary", 19, 1.278753601, 361, "$1,300.00", 3.113943352],
    ["1960–1963", "Ferrari", "250 GTE 2+2", 16, 1.204119983, 256, "$350.00", 2.544068044],
    ["1962–1964", "Ferrari", "250 GTL Lusso", 19, 1.278753601, 361, "$2,650.00", 3.423245874],
    ["1962", "Ferrari", "250 GTO", 18, 1.255272505, 324, "$375.00", 2.574031268],
    ["1967–1968", "Ferrari", "275 GTB/4 NART Spyder", 17, 1.230448921, 289, "$450.00", 2.653212514],
    ["1968–1973", "Ferrari", "365 GTB/4 Daytona", 17, 1.230448921, 289, "$140.00", 2.146128036],
    ["1962–1967", "Jaguar", "E-type OTS", 15, 1.176091259, 225, "$77.50", 1.889301703],
    ["1969–1971", "Jaguar", "E-type Series II OTS", 14, 1.146128036, 196, "$62.00", 1.792391689],
    ["1971–1974", "Jaguar", "E-type Series III OTS", 16, 1.204119983, 256, "$125.00", 2.096910013],
    ["1951–1954", "Jaguar", "XK 120 roadster (steel)", 17, 1.230448921, 289, "$400.00", 2.602059991],
    ["1950–1953", "Jaguar", "XK C-type", 16, 1.204119983, 256, "$250.00", 2.397940009],
    ["1956–1957", "Jaguar", "XKSS", 13, 1.113943352, 169, "$70.00", 1.84509804],
]

columns = [
    "Year",
    "Make",
    "Model",
    "Rating",
    "LogRating",
    "Rating_Sqr",
    "Price",
    "LogPrice"
]

df = pd.DataFrame(data, columns=columns)


df["Price"] = (
    df["Price"]
    .str.replace("$", "", regex=False)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df



############################################### 19 ###########################################


import pandas as pd
import itertools
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Your data
data = [
    [12, 57, 152, 0],
    [24, 67, 163, 0],
    [13, 58, 155, 0],
    [56, 86, 177, 1],
    [28, 59, 196, 0],
    [51, 76, 189, 1],
    [18, 56, 155, 1],
    [31, 78, 120, 0],
    [37, 80, 135, 1],
    [15, 78, 98, 0],
    [22, 71, 152, 0],
    [36, 70, 173, 1],
    [15, 67, 135, 1],
    [48, 77, 209, 1],
    [15, 60, 199, 0],
    [36, 82, 119, 1],
    [8, 66, 166, 0],
    [34, 80, 125, 1],
    [3, 62, 117, 0],
    [37, 59, 207, 1]
]

columns = ["Risk", "Age", "Pressure", "Smoker"]
df = pd.DataFrame(data, columns=columns)

# Target variable
y = df["Smoker"]

# Predictor columns
features = ["Risk", "Age", "Pressure"]

results = []

# Try all non-empty combinations of predictors
for r in range(1, len(features) + 1):
    for combo in itertools.combinations(features, r):
        X = df[list(combo)]

        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)

        results.append((combo, r2))

# Sort results by R² (descending)
results.sort(key=lambda x: x[1], reverse=True)

# Print results
for combo, r2 in results:
    print(f"Features: {combo}, R²: {r2:.4f}")

# Best combination
best_combo, best_r2 = results[0]
print("\nBest combination:")
print(f"{best_combo} with R² = {best_r2:.4f}")



######################################## 24 #############################################

import pandas as pd
import statsmodels.api as sm

# Drying times (minutes) by paint type
paint_data = {
    "Paint 1": [128, 137, 135, 124, 141],
    "Paint 2": [144, 133, 142, 146, 130],
    "Paint 3": [133, 143, 137, 136, 131],
    "Paint 4": [150, 142, 135, 140, 153]
}

df_wide = pd.DataFrame(paint_data)

df_long = df_wide.melt(
    var_name="Paint",
    value_name="y"
)


df = pd.get_dummies(df_long, columns=["Paint"])

# Keep only three dummies (Paint 4 is reference)
df["d1"] = df["Paint_Paint 1"]
df["d2"] = df["Paint_Paint 2"]
df["d3"] = df["Paint_Paint 3"]

X = df[["d1", "d2", "d3"]]
X = sm.add_constant(X)   # intercept
y = df["y"]



model = sm.OLS(y, X).fit()
print(model.summary())


df_long.groupby("Paint")["y"].mean()
