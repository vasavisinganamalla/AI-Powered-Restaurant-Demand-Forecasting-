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

# Category mapping
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

# Menu images
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

# Pricing
item_prices = {
    "Idli": 50, "Dosa": 80, "Poori": 70,
    "Upma": 60, "Vada": 50,
    "Tea": 20, "Coffee": 40,
    "Veg Biryani": 160,
    "Chicken Biryani": 220,
    "Mutton Biryani": 280,
    "Chicken Curry": 200,
    "Meals Plate": 150,
    "Samosa": 30, "Veg Puff": 35,
    "Cake Slice": 90,
    "Biscuit Pack": 60,
    "Haleem": 250
}

@app.route("/")
def home():
    return render_template("index.html")

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

@app.route("/result", methods=["POST"])
def result():
    branch = request.form["branch"]
    meal = request.form["meal"]
    item = request.form["item"]
    festival = request.form["festival"]
    day_type = request.form["day_type"]

    price = item_prices.get(item, 150)
    category = item_category_map.get(item, "Main")
    selected_image = menu_images.get(item, "default.jpg")

    today = datetime.date.today()

    input_df = pd.DataFrame({
        "Branch_Name": [branch],
        "Day_Type": [day_type],
        "Meal_Time": [meal],
        "Item_Name": [item],
        "Category": [category],
        "Price": [price],
        "Festival": [festival],
        "Day": [today.day],
        "Month": [today.month],
        "Weekday": [today.weekday()]
    })

    categorical_cols = [
        "Branch_Name", "Day_Type", "Meal_Time",
        "Item_Name", "Category", "Festival"
    ]

    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    prediction = int(model.predict(input_df)[0])

    revenue = prediction * price
    cost = int(revenue * 0.6)
    profit = revenue - cost
    margin = round((profit / revenue) * 100, 2) if revenue != 0 else 0
    
    # Calculate per-unit metrics for clarity
    cost_per_unit = int(price * 0.6)
    profit_per_unit = int(price * 0.4)

    if prediction > 180:
        level = "High Demand"
        tip = "Prepare additional stock."
    elif prediction > 100:
        level = "Moderate Demand"
        tip = "Normal preparation is sufficient."
    else:
        level = "Low Demand"
        tip = "Reduce preparation."

    days_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    demand_values = np.clip(
        np.random.randint(prediction - 30, prediction + 30, size=7),
        0, None
    )

    plt.figure(figsize=(6,4))
    plt.plot(days_list, demand_values, marker="o", color="#3b82f6", linewidth=2)
    plt.title(f"Weekly Demand Trend - {item}", fontsize=14, fontweight='bold')
    plt.xlabel("Day", fontsize=10)
    plt.ylabel("Expected Orders", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    graph_path = os.path.join(app.root_path, "static", "trend.png")
    plt.savefig(graph_path, dpi=100, bbox_inches='tight')
    plt.close()

    return render_template(
        "result.html",
        demand=prediction,
        level=level,
        tip=tip,
        price=price,
        revenue=revenue,
        cost=cost,
        profit=profit,
        margin=margin,
        graph="trend.png",
        food_image=selected_image,
        branch=branch,
        meal=meal,
        item=item,
        festival=festival,
        day_type=day_type,
        cost_per_unit=cost_per_unit,
        profit_per_unit=profit_per_unit,
        days_list=days_list,
        demand_values=demand_values
    )

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

    input_df = pd.DataFrame({
        "Branch_Name": [branch],
        "Day_Type": [day_type],
        "Meal_Time": [meal],
        "Item_Name": [item],
        "Category": [category],
        "Price": [new_price],
        "Festival": [new_festival],
        "Day": [today.day],
        "Month": [today.month],
        "Weekday": [today.weekday()]
    })

    categorical_cols = [
        "Branch_Name","Day_Type","Meal_Time",
        "Item_Name","Category","Festival"
    ]

    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])

    new_prediction = int(model.predict(input_df)[0])
    
    # Calculate updated metrics
    new_revenue = new_prediction * new_price
    new_cost = int(new_revenue * 0.6)
    new_profit = new_revenue - new_cost
    new_margin = round((new_profit / new_revenue) * 100, 2) if new_revenue != 0 else 0
    
    # Per-unit metrics
    cost_per_unit = int(new_price * 0.6)
    profit_per_unit = int(new_price * 0.4)
    
    # Determine level and tip based on new prediction
    if new_prediction > 180:
        level = "High Demand"
        tip = "Prepare additional stock."
    elif new_prediction > 100:
        level = "Moderate Demand"
        tip = "Normal preparation is sufficient."
    else:
        level = "Low Demand"
        tip = "Reduce preparation."
    
    # Generate weekly trend for simulated scenario
    days_list = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    demand_values = np.clip(
        np.random.randint(new_prediction - 30, new_prediction + 30, size=7),
        0, None
    )
    
    # Create new graph for simulated scenario
    plt.figure(figsize=(6,4))
    plt.plot(days_list, demand_values, marker="o", color="#8b5cf6", linewidth=2)
    plt.title(f"Weekly Demand Trend - {item} (Simulated)", fontsize=14, fontweight='bold')
    plt.xlabel("Day", fontsize=10)
    plt.ylabel("Expected Orders", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    graph_path = os.path.join(app.root_path, "static", "trend.png")
    plt.savefig(graph_path, dpi=100, bbox_inches='tight')
    plt.close()

    return render_template(
        "result.html",
        demand=new_prediction,
        level=level,
        tip=tip,
        price=new_price,
        revenue=new_revenue,
        cost=new_cost,
        profit=new_profit,
        margin=new_margin,
        graph="trend.png",
        food_image=selected_image,
        branch=branch,
        meal=meal,
        item=item,
        festival=new_festival,
        day_type=day_type,
        cost_per_unit=cost_per_unit,
        profit_per_unit=profit_per_unit,
        days_list=days_list,
        demand_values=demand_values
    )

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
    
    # Calculate monthly figures
    monthly_values = [v * 30 for v in values]

    plt.figure(figsize=(8,5))
    bars = plt.bar(items, values, color=['#3b82f6', '#8b5cf6', '#059669', '#f59e0b', '#ef4444'])
    plt.xticks(rotation=20, fontsize=10)
    plt.title("Top Selling Items - Average Daily Demand", fontsize=14, fontweight='bold')
    plt.ylabel("Orders per Day", fontsize=11)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{val}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    chart_path = os.path.join(app.root_path, "static", "top_items.png")
    plt.savefig(chart_path, dpi=100, bbox_inches='tight')
    plt.close()

    return render_template(
        "analytics.html",
        total_items=len(items),
        avg_demand=int(sum(values)/len(values)),
        festival=220,
        normal=150,
        chart="top_items.png",
        items=items,
        daily_values=values,
        monthly_values=monthly_values
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)