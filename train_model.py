import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# --- Load Data ---
orders = pd.read_csv("List of Orders.csv")
details = pd.read_csv("Order Details.csv")
targets = pd.read_csv("Sales target.csv")  # Optional, not used in training

# --- Clean column names ---
orders.columns = [col.strip().lower().replace(" ", "_") for col in orders.columns]
details.columns = [col.strip().lower().replace(" ", "_") for col in details.columns]

# --- Merge orders and details on order_id ---
if 'order_id' not in orders.columns or 'order_id' not in details.columns:
    raise KeyError("❌ 'order_id' missing from one or both datasets.")

merged = pd.merge(details, orders, on='order_id')

print("📋 Columns after merge:", merged.columns.tolist())

# --- Define correct column names directly ---
unit_price_col = 'amount'
quantity_col = 'quantity'
category_col = 'category'
sub_category_col = 'sub-category'

# --- Compute total_price ---
merged['total_price'] = merged[unit_price_col] * merged[quantity_col]

# --- Group by category and sub-category ---
group_cols = [sub_category_col, category_col]
for col in group_cols:
    if col not in merged.columns:
        raise KeyError(f"❌ Missing required column: {col}")

product_prices = merged.groupby(group_cols).agg({
    unit_price_col: 'mean',
    'total_price': 'sum',
    quantity_col: 'sum'
}).reset_index()
# --- Create and Save Sub-Category to Category Mapping ---
sub_category_map = dict(product_prices[['sub-category', 'category']].values)

import json
with open("model/sub_category_map.json", "w") as f:
    json.dump(sub_category_map, f)


# --- Rename for modeling ---
product_prices.rename(columns={
    unit_price_col: 'avg_unit_price',
    'total_price': 'total_revenue'
}, inplace=True)

# --- Encode categorical features ---
product_prices.dropna(inplace=True)

le_product = LabelEncoder()
le_category = LabelEncoder()

product_prices['product_encoded'] = le_product.fit_transform(product_prices[sub_category_col])
product_prices['category_encoded'] = le_category.fit_transform(product_prices[category_col])

print("✅ Sub-category classes:", le_product.classes_)
print("✅ Category classes:", le_category.classes_)


# --- Prepare data ---
X = product_prices[['product_encoded', 'category_encoded']]
y = product_prices['avg_unit_price']

# --- Train/test split and model training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# --- Save model and encoders ---
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/price_model.pkl')
joblib.dump(le_product, 'model/label_product.pkl')
joblib.dump(le_category, 'model/label_category.pkl')

print("✅ Model training completed and saved.")





