# Amazon EC2 Cost Analysis and Prediction

This project analyzes Amazon EC2 instance pricing data and predicts On-Demand costs using Python and Linear Regression.

The project was developed for the INFO49971 Cloud Economics course.

## Project Objectives

The main objectives are to:

- Analyze Amazon EC2 pricing data
- Compare different EC2 pricing options
- Identify potential cost outliers
- Compare T2 and T3 instance families
- Analyze On-Demand and Reserved pricing
- Predict EC2 On-Demand costs using Linear Regression
- Display the analysis through a Streamlit web application

## Dataset

The dataset contains:

- 861 EC2 instances
- 11 columns

The main attributes include:

- Instance Name
- API Name
- Instance Memory
- vCPUs
- Instance Storage
- Network Performance
- Linux On-Demand Cost
- Linux Reserved Cost
- Linux Spot Minimum Cost
- Windows On-Demand Cost
- Windows Reserved Cost

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

## Cost Analysis

The application provides:

- Dataset overview
- Summary statistics
- Cost distribution visualization
- Missing value analysis
- IQR-based cost outlier detection
- Lowest-cost EC2 instance comparison
- Linux On-Demand vs Reserved cost comparison

## EC2 Instance Family Analysis

The application compares the T2 and T3 EC2 instance families.

It includes:

- T2 cost summary
- T3 cost summary
- Cost distribution boxplots
- Lowest-cost T2 and T3 instances

## Cost Prediction

A Linear Regression model is used to predict Linux On-Demand EC2 pricing.

### Predictor Variables

- Instance Memory
- vCPUs

### Target Variable

- On-Demand Cost

The dataset is divided into:

- 80% training data
- 20% testing data

The model is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

The application also allows users to enter memory and vCPU values to estimate the On-Demand hourly cost of a new EC2 configuration.

## Project Files

```text
ec2-cost-analysis/
│
├── Cost_prediction.py
├── ec2dataset.csv
├── requirements.txt
└── README.md
