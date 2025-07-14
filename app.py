from flask import Flask, render_template, request
import joblib
import numpy as np
import json

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model/price_model.pkl')
le_product = joblib.load('model/label_product.pkl')
le_category = joblib.load('model/label_category.pkl')
with open("model/sub_category_map.json", "r") as f:
    sub_category_map = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    predicted_price = None
    selected_category = None
    selected_sub_category = None

    categories = sorted(le_category.classes_.tolist())
    sub_categories = sorted(le_product.classes_.tolist())

    if request.method == 'POST':
        selected_category = request.form.get('category')
        selected_sub_category = request.form.get('sub_category')

        try:
            cat_encoded = le_category.transform([selected_category])[0]
            sub_cat_encoded = le_product.transform([selected_sub_category])[0]
            prediction = model.predict(np.array([[sub_cat_encoded, cat_encoded]]))
            predicted_price = round(prediction[0], 2)
        except Exception:
            predicted_price = "❌ Error: Unknown product or category."

    return render_template('index.html',
                           predicted_price=predicted_price,
                           categories=categories,
                           sub_categories=sub_categories,
                           selected_category=selected_category,
                           selected_sub_category=selected_sub_category,
                           sub_category_map=sub_category_map,
                           sub_category_map_json=json.dumps(sub_category_map))

if __name__ == "__main__":
    app.run(debug=True)
