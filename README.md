# Crop Prediction Model
This project uses machine learning to predict suitable crops based on input parameters like soil type, temperature, and rainfall.

## Project Structure
├── crop.py # Core logic for crop recommendation
├── ml.py # ML model training and prediction code
├── model.pkl # Trained machine learning model
├── model.zip # Zipped version of the model
├── crop_prediction_model_one.csv # Dataset used for training/testing
├── config.toml # Configuration file for the app
├── requirements.txt # Python dependencies
├── runtime.txt # Runtime environment definition
└── README.md # Project documentation

## Model Description
- The model is trained using the dataset `crop_prediction_model_one.csv`.
- A serialized version of the trained model is stored in `model.pkl`.
- `ml.py` contains the training pipeline and utilities.
- `crop.py` is used for making predictions using the trained model.

## Installation
1. **Clone the repository**  
   git clone <your-repo-url>
   cd <project-directory>
Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
Install dependencies
pip install -r requirements.txt

**Usage**
Train the Model (if needed)
python ml.py
Predict Using the Model
python crop.py
You may need to edit crop.py to supply input data or wrap it with a web framework (e.g., Flask) for API usage.

**Deployment**
This project can be deployed on cloud platforms like Heroku, Render, etc.
runtime.txt defines the Python version.
config.toml holds configuration variables like host and port.

**Configuration**
Example config.toml:
[app]
host = "0.0.0.0"
port = 8000
debug = true
**Requirements**
Install all required libraries with:
pip install -r requirements.txt
