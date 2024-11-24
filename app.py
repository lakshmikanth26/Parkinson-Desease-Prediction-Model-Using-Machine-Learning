import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except FileNotFoundError:
    print("Error: model.pkl file not found. Make sure the file is in the same directory.")

@app.route('/')
def index():
    # Render the input form
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input from the form
        input_text = request.form['text']
        input_text_sp = input_text.split(',')

        # Convert input to numpy array
        np_data = np.asarray(input_text_sp, dtype=np.float32)

        # Validate input size
        expected_features = model.n_features_in_
        if len(np_data) != expected_features:
            return render_template("index.html", message=f"Error: Model expects {expected_features} features, but got {len(np_data)}.")

        # Make a prediction
        prediction = model.predict(np_data.reshape(1, -1))

        # Generate output message
        if prediction == 1:
            output = "This person has Parkinson's disease."
        else:
            output = "This person does not have Parkinson's disease."

    except ValueError:
        output = "Error: Please ensure all inputs have numeric values."
    except Exception as e:
        output = f"An unexpected error occurred: {e}"

    # Render the result
    return render_template("index.html", message=output)

if __name__ == "__main__":
    app.run(debug=True)
