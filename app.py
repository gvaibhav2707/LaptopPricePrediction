from flask import Flask, render_template, request
import pandas as pd
from pickle import load

app = Flask(__name__)

# Load the dataset (preprocessed)
file_path = 'laptop_data.csv'
data = pd.read_csv(file_path)

# Extract unique values for dropdowns
company_options = sorted(data['Company'].unique())
type_options = sorted(data['TypeName'].unique())
Inches_options = sorted(data['Inches'].unique())
screen_resolution_options = sorted(data['ScreenResolution'].unique())
cpu_options = sorted(data['Cpu'].apply(lambda x: " ".join(x.split()[:3])).unique())
ram_options = sorted(data['Ram'].unique())
memory_options = sorted(data['Memory'].unique())
gpu_options = sorted(data['Gpu'].unique())
Op_Sys_options = sorted(data['OpSys'].unique())

@app.route('/')
def index():
    return render_template('index.html', 
                           company_options=company_options, 
                           type_options=type_options,
                           Inches_options=Inches_options,
                           screen_resolution_options=screen_resolution_options,
                           cpu_options=cpu_options,
                           ram_options=ram_options,
                           memory_options=memory_options,
                           gpu_options=gpu_options,
                           Op_Sys_options=Op_Sys_options)

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    company = request.form.get('company')
    type_name = request.form.get('type')
    display_size = request.form.get('displaySize')
    screen_resolution = request.form.get('screenResolution')
    cpu = request.form.get('cpu')
    ram = request.form.get('ram')
    memory = request.form.get('memory')
    gpu = request.form.get('graphicCard')
    op_sys = request.form.get('os')

    # Load the model, encoders, and polynomial feature transformer
    with open("model.pkl", "rb") as f:
        model = load(f)

    with open("label_encoders.pkl", "rb") as f:
        label_encoders = load(f)

    with open("pfeatures.pkl", "rb") as f:
        pfeatures = load(f)

    # Prepare input data as a DataFrame
    input_data = {
        'Company': company,
        'TypeName': type_name,
        'Inches': float(display_size),
        'ScreenResolution': screen_resolution,
        'Cpu': cpu,
        'Ram': ram,
        'Memory': memory,
        'Gpu': gpu,
        'OpSys': op_sys,
    }
    df = pd.DataFrame([input_data])

    # Apply label encoders to categorical data with fallback for unseen labels
    for column in df.columns:
        if column in label_encoders:
            try:
                df[column] = label_encoders[column].transform(df[column])
            except ValueError:
                # Default to the most common label if unseen
                most_common_label = label_encoders[column].classes_[0]
                df[column] = label_encoders[column].transform([most_common_label])

    # Transform the input features using PolynomialFeatures
    transformed_input = pfeatures.transform(df)

    # Predict the price
    result = model.predict(transformed_input)

    # Format and return the message with the rupee symbol
    msg = f"The estimated price of the laptop is around ₹{int(result[0])}"

    # Pass the message and options back to the template
    return render_template('index.html', msg=msg, 
                           company_options=company_options, 
                           type_options=type_options,
                           Inches_options=Inches_options,
                           screen_resolution_options=screen_resolution_options,
                           cpu_options=cpu_options,
                           ram_options=ram_options,
                           memory_options=memory_options,
                           gpu_options=gpu_options,
                           Op_Sys_options=Op_Sys_options)




if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
