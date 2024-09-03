from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the dataset and the trained model
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("LassoModel.pkl", 'rb'))

@app.route('/')
def index():
    # Extract unique values for dropdown options
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect data from the form
    bedrooms = request.form.get('beds')
    bathrooms = request.form.get('baths')
    size = request.form.get('size')
    zipcode = request.form.get('zip_code')

    # Create a DataFrame with the input data
    input_data = pd.DataFrame([[bedrooms, bathrooms, size, zipcode]],
                              columns=['beds', 'baths', 'size', 'zip_code'])

    # Convert 'baths' column to numeric with errors='coerce'
    input_data['baths'] = pd.to_numeric(input_data['baths'], errors='coerce')

    # Convert input data to appropriate types
    input_data = input_data.astype({'beds': int, 'baths': float, 'size': float, 'zip_code': int})

    # Handle unknown categories in the input data
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            print(f"Unknown categories in {column}: {unknown_categories}")
            # Replace unknown categories with the most frequent value in the dataset
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])

    # Ensure 'price_per_sqft' is calculated or added
    if 'price_per_sqft' not in input_data.columns:
        # Example: You might need to calculate 'price_per_sqft' if you have the necessary information
        # Assuming you have the 'price' column, otherwise, this part should be modified accordingly
        input_data['price_per_sqft'] = data['price'] / data['size']

    print("Processed Input Data:")
    print(input_data)

    # Predict the price using the trained model
    prediction = pipe.predict(input_data)[0]

    return str(prediction)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
