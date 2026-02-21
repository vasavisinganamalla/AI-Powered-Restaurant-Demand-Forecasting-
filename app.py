from flask import Flask, render_template, request
import joblib
import pandas as pd
import datetime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import os
import numpy as np

app = Flask(__name__)

# Load model + encoders
model = joblib.load("menu_demand_model.pkl")
encoders = joblib.load("encoders.pkl")

# Dropdown values
branch_options = encoders["Branch_Name"].classes_
meal_options = encoders["Meal_Time"].classes_
item_options = encoders["Item_Name"].classes_[:20]
festival_options = encoders["Festival"].classes_
daytype_options = encoders["Day_Type"].classes_

# Auto Category Map
item_category_map = {
    "Idli": "Breakfast",
    "Dosa": "Breakfast",
    "Poori": "Breakfast",
    "Chicken Biryani": "Main",
    "Mutton Biryani": "Main",
    "Veg Puff": "Bakery",
    "Biscuit Pack": "Bakery",
    "Cake Slice": "Bakery",
    "Haleem": "Festival Special"
}

# Menu Images Mapping
menu_images = {
    "Biscuit Pack": "biscuit.jpg",
    "Cake Slice": "cake.jpg",
    "Chicken Biryani": "biryani.jpg",
    "Chicken Curry": "chickencurry.jpg",
    "Coffee": "coffee.jpg",
    "Dosa": "dosa.jpg",
    "Haleem": "haleem.jpg",
    "Idli": "idli.jpg",
    "Meals Plate": "meals.jpg",
    "Mutton Biryani": "mutton.jpg",
    "Poori": "poori.jpg",
    "Samosa": "samosa.jpg",
    "Tea": "tea.jpg",
    "Upma": "upma.jpg",
    "Vada": "vada.jpg",
    "Veg Biryani": "vegbiryani.jpg",
    "Veg Puff": "vegpuff.jpg"
}

# Fixed Menu Prices (Auto Pricing)
item_prices = {
    "Idli": 50,
    "Dosa": 80,
    "Poori": 70,
    "Upma": 60,
    "Vada": 50,

    "Tea": 20,
    "Coffee": 40,

    "Veg Biryani": 160,
    "Chicken Biryani": 220,
    "Mutton Biryani": 280,

    "Chicken Curry": 200,
    "Paneer Curry": 180,
    "Meals Plate": 150,

    "Samosa": 30,
    "Veg Puff": 35,
    "Egg Puff": 45,

    "Cake Slice": 90,
    "Biscuit Pack": 60,

    "Haleem": 250
}

#  Home Page
@app.route("/")
def home():
    return render_template("index.html")


#  Prediction Form Page
@app.route("/predict")
def predict():
    return render_template(
        "predict.html",
        branches=branch_options,
        meals=meal_options,
        items=item_options,
        festivals=festival_options,
        daytypes=daytype_options
    )


# Main Prediction Route
@app.route("/result", methods=["POST"])
def result():

    branch = request.form["branch"]
    meal = request.form["meal"]
    item = request.form["item"]
    festival = request.form["festival"]
    day_type = request.form["day_type"]

    # Auto Price
    price = item_prices.get(item, 150)

    # Category + Food Image
    category = item_category_map.get(item, "Main")
    selected_image = menu_images.get(item, "default.jpg")

    # Date Features
    today = datetime.date.today()
    day = today.day
    month = today.month
    weekday = today.weekday()

    # Input DataFrame
    input_df = pd.DataFrame({
        "Branch_Name": [branch],
        "Day_Type": [day_type],
        "Meal_Time": [meal],
        "Item_Name": [item],
        "Category": [category],
        "Price": [price],
        "Festival": [festival],
        "Day": [day],
        "Month": [month],
        "Weekday": [weekday]
    })

    categorical_cols = [
        "Branch_Name", "Day_Type", "Meal_Time",
        "Item_Name", "Category", "Festival"
    ]

    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    #  Predict Demand
    prediction = model.predict(input_df)[0]

    #  Demand Insight
    if prediction > 180:
        level = "High Demand"
        tip = "Prepare additional stock to avoid shortages."
    elif prediction > 100:
        level = "Moderate Demand"
        tip = "Normal preparation is sufficient."
    else:
        level = "Low Demand"
        tip = "Reduce preparation to minimize waste."

    # Weekly Trend Graph
    days_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    demand_values = np.random.randint(int(prediction) - 30,
                                      int(prediction) + 30, size=7)

    plt.figure()
    plt.plot(days_list, demand_values, marker="o")
    plt.title("Weekly Demand Trend")
    plt.xlabel("Days")
    plt.ylabel("Orders")

    graph_path = os.path.join(app.root_path, "static", "trend.png")
    plt.savefig(graph_path)
    plt.close()

    # Send Everything to result.html
    return render_template(
        "result.html",
        demand=int(prediction),
        level=level,
        tip=tip,
        price=price,               #Correct Fix
        graph="trend.png",
        food_image=selected_image,

        branch=branch,
        meal=meal,
        item=item,
        festival=festival,
        day_type=day_type
    )


# Scenario Simulator Route
@app.route("/simulate", methods=["POST"])
def simulate():

    branch = request.form["branch"]
    meal = request.form["meal"]
    item = request.form["item"]
    day_type = request.form["day_type"]

    new_price = int(request.form["new_price"])
    new_festival = request.form["new_festival"]

    category = item_category_map.get(item, "Main")
    selected_image = menu_images.get(item, "default.jpg")

    today = datetime.date.today()
    day = today.day
    month = today.month
    weekday = today.weekday()

    input_df = pd.DataFrame({
        "Branch_Name": [branch],
        "Day_Type": [day_type],
        "Meal_Time": [meal],
        "Item_Name": [item],
        "Category": [category],
        "Price": [new_price],
        "Festival": [new_festival],
        "Day": [day],
        "Month": [month],
        "Weekday": [weekday]
    })

    categorical_cols = [
        "Branch_Name", "Day_Type", "Meal_Time",
        "Item_Name", "Category", "Festival"
    ]

    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    new_prediction = model.predict(input_df)[0]

    return render_template(
        "result.html",
        demand=int(new_prediction),
        level="Simulated Scenario",
        tip="Demand updated based on the new conditions.",
        price=new_price,
        graph="trend.png",
        food_image=selected_image,

        branch=branch,
        meal=meal,
        item=item,
        festival=new_festival,
        day_type=day_type
    )


#  Analytics Dashboard
@app.route("/analytics")
def analytics():

    top_items = {
        "Chicken Biryani": 210,
        "Mutton Biryani": 185,
        "Idli": 140,
        "Veg Puff": 120,
        "Haleem": 250
    }

    items = list(top_items.keys())
    values = list(top_items.values())

    plt.figure()
    plt.bar(items, values)
    plt.title("Top Selling Menu Items")
    plt.ylabel("Average Demand")
    plt.xticks(rotation=20)

    chart_path = os.path.join(app.root_path, "static", "top_items.png")
    plt.savefig(chart_path)
    plt.close()

    return render_template(
        "analytics.html",
        total_items=len(items),
        avg_demand=int(sum(values) / len(values)),
        festival=220,
        normal=150,
        chart="top_items.png"
    )


if __name__ == "__main__":
    app.run(debug=True)