import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
np.random.seed(42)

branches = ["Central City", "Old Town", "Tech Park", "Market Area"]
day_types = ["Weekday", "Weekend"]
meals = ["Breakfast", "Lunch", "Dinner"]
items = [
    "Idli", "Dosa", "Poori", "Upma", "Vada",
    "Tea", "Coffee",
    "Veg Biryani", "Chicken Biryani", "Mutton Biryani",
    "Chicken Curry", "Paneer Curry", "Meals Plate",
    "Samosa", "Veg Puff", "Cake Slice", "Biscuit Pack", "Haleem"
]
categories = ["Breakfast", "Main", "Bakery", "Festival Special"]
festival = ["Yes", "No"]

data = []

for _ in range(6000):
    branch = np.random.choice(branches)
    day_type = np.random.choice(day_types)
    meal = np.random.choice(meals)
    item = np.random.choice(items)
    category = np.random.choice(categories)
    fest = np.random.choice(festival)
    price = np.random.randint(20, 300)

    day = np.random.randint(1, 31)
    month = np.random.randint(1, 13)
    weekday = np.random.randint(0, 7)

    # Simulate realistic demand
    demand = (
        100
        + (20 if day_type == "Weekend" else 0)
        + (30 if meal == "Dinner" else 0)
        + (40 if fest == "Yes" else 0)
        - int(price / 10)
        + np.random.randint(-20, 20)
    )

    demand = max(10, demand)

    data.append([
        branch, day_type, meal, item,
        category, price, fest,
        day, month, weekday, demand
    ])

columns = [
    "Branch_Name", "Day_Type", "Meal_Time",
    "Item_Name", "Category", "Price", "Festival",
    "Day", "Month", "Weekday", "Quantity_Sold"
]

df = pd.DataFrame(data, columns=columns)
df.head()
encoders = {}
categorical_cols = [
    "Branch_Name", "Day_Type", "Meal_Time",
    "Item_Name", "Category", "Festival"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    X = df.drop("Quantity_Sold", axis=1)
y = df["Quantity_Sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Model Performance:")
print("R² Score:", round(r2, 4))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
joblib.dump(rf_model, "menu_demand_model.pkl")
joblib.dump(encoders, "encoders.pkl")

print("Model and Encoders saved successfully.")