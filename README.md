# Analysis of Earthquake Prediction using Machine Learning

#### Overview
This project leverages machine learning models to predict potential earthquake events based on historical data. The system utilizes Random Forest, XGBoost, and AdaBoost classifiers to analyze earthquake data and provide real-time predictions, visualized on an interactive map using Django and Google Maps API.

#### Features
-- **Data Preprocessing & Feature Engineering**: Handling raw earthquake data for model input.

-- **Machine Learning Models**: Implementation of Random Forest, XGBoost, and AdaBoost classifiers.

-- **Real-time Prediction**: Predictions visualized on a dynamic map.

-- **Django Web Interface**: A web-based interface for easy interaction with the prediction results.

-- **SQLite Database**: Efficient storage and retrieval of processed data.

#### Technologies
-- **Backend**: Django

-- **Machine Learning**: Scikit-learn, XGBoost, AdBoost, Random Forest

-- **Data Analysis**: Pandas, NumPy

-- **Data Visualization**: Matplotlib, Seaborn

-- **Database**: SQLite

-- **API**: Google Maps API

#### Project Structure
Earthquake-Prediction/
├── data/             # Raw and processed earthquake data (CSV)
│   └── database/     # SQLite database
├── models/           # Trained machine learning models
├── quakeproject/     # Django project
│   └── webapp/       # Django app with views and templates        
└── README.md         # Project documentation

#### Setup

##### Prerequisites
-- Python

-- pip (Python package installer)

##### Installation
Clone the repository:
    git clone https://github.com/shaktisagar14/Earthquake-Prediction.git
    cd Earthquake-Prediction

Set up a virtual environment:
    python -m venv venv

Install dependencies:
    pip install -r requirements.txt

##### Usage
Run the Django server:
    cd quakeproject
    python manage.py runserver

Access the web application at http://127.0.0.1:8000/.

##### Authors
Shakti Sagar Samantaray
MCA (ITER, SOA University)
[GitHub Profile](https://github.com/shaktisagar14)

