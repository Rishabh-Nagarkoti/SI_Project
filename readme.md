## Electric Motor Temperature Prediction

**Category:** Machine Learning

**Skills Required:**

Python,Exploratory data Analysis,Numpy,Scikit-Learn

**Project Description:**

The permanent-magnet synchronous machine (PMSM) drive is one of the best choices for a full range of motion control applications. For example, the PMSM is widely used in robotics, machine tools, actuators, and it is being considered in high-power applications such as industrial drives and vehicular propulsion. It is also used for residential/commercial applications. The PMSM is known for having low torque ripple, superior dynamic performance, high efficiency and high power density.

The task is to design a model with appropriate feature engineering that estimates the target temperature of a rotor In this project,we will be using algorithms such as Linear Regression, Decision Tree, Random forest and SVM. We will train and test the data with these algorithms and select the best model. Best algorithm will be selected and saved in pkl format. We will be doing flask integration.

**Author:** Nagarkoti Rishabh

**College:** S. N. Patel Institute Of Technology, Bardoli

**University:** Gujarat Technological University (GTU)

**Date:** August 10, 2025

### Abstract

The Permanent-Magnet Synchronous Machine (PMSM) is a critical component in numerous high-performance applications, from industrial robotics to electric vehicles. The thermal state of the rotor is a key indicator of the machine's health, performance, and operational safety. Direct measurement of rotor temperature is often impractical, creating a need for accurate predictive models. This project details the end-to-end development of a machine learning system to estimate the rotor temperature of a PMSM based on readily available sensor data. The process involved comprehensive data exploration, robust feature engineering, and a comparative analysis of four regression algorithms: Linear Regression, Decision Tree, Support Vector Machine (SVR), and Random Forest. The Random Forest model was identified as the most robust and reliable choice, demonstrating strong predictive power with an R-squared value of 0.9035 on unseen test data, effectively balancing accuracy with generalization. The selected model was then deployed as an interactive web application using the Flask framework, providing a practical tool for real-time temperature estimation.

### 1\. Introduction

#### 1.1 Background

Permanent-Magnet Synchronous Machines (PMSMs) are at the forefront of modern electric motor technology due to their high efficiency, superior power density, and excellent dynamic performance. Their application spans a wide range of industries, including advanced manufacturing, robotics, aerospace, and the rapidly growing electric vehicle market.

#### 1.2 Problem Statement

The operational lifespan and immediate performance of a PMSM are heavily influenced by its thermal condition. Excessive heat, particularly in the rotor's permanent magnets, can lead to demagnetization, reduced efficiency, and catastrophic failure. However, the rotor is an enclosed, rotating component, making direct temperature sensing with physical probes difficult, expensive, and often unreliable in industrial settings. Therefore, the ability to accurately predict the rotor temperature using other, more accessible sensor readings (such as stator temperatures, currents, and voltages) is of immense value for predictive maintenance, operational optimization, and ensuring system safety.

#### 1.3 Project Objectives

The primary objectives of this project were as follows:

1. To perform a thorough exploratory data analysis (EDA) on a PMSM operational dataset to understand the relationships between various physical parameters.
2. To engineer new, physically meaningful features to enhance model performance.
3. To train, test, and critically evaluate four distinct machine learning regression models for their ability to predict rotor temperature.
4. To deploy the final, selected model as a user-friendly web application, demonstrating a practical application of the solution.

### 2\. Methodology

#### 2.1 Dataset Description

The project utilized the "PMSM Temperature Data Set" obtained from the UCI Machine Learning Repository. This dataset contains numerous measurement sessions from a real-world PMSM test bench. The key features include:

1. **Environmental:** ambient, coolant
2. **Stator Voltages:** u_d, u_q (d-q reference frame)
3. **Stator Currents:** i_d, i_q (d-q reference frame)
4. **Operational:** motor_speed, torque
5. **Stator Temperatures:** stator_yoke, stator_tooth, stator_winding

The target variable for prediction was pm, the rotor surface temperature in degrees Celsius.

#### 2.2 Exploratory Data Analysis (EDA)

An initial analysis was conducted using pandas, matplotlib, and seaborn. Key findings included:

1. The dataset was complete with no missing values.
2. A correlation heatmap revealed a very strong positive correlation between the various stator temperatures and the target rotor temperature (pm), as expected from physical principles.
3. Distributions of features like motor_speed and torque indicated varied operational cycles, providing a rich dataset for model training.
4. Box plots were used to identify statistical outliers, particularly in the ambient temperature readings, which were subsequently handled by capping them at the 1st and 99th percentiles to reduce noise.

#### 2.3 Feature Engineering

To better represent the underlying physics of motor heating, three new features were engineered:

1. **i_mag_sq**: Calculated as $i_d^2 + i_q^2$, this feature represents the square of the stator current magnitude, which is directly proportional to copper losses—a primary source of heat.
2. **p_elec**: Calculated as $u_d \\cdot i_d + u_q \\cdot i_q$, this represents the instantaneous electrical power being drawn by the motor.
3. **temp_diff_stator_coolant**: The difference between stator_yoke and coolant temperature, representing the thermal gradient driving heat dissipation.

#### 2.4 Model Training and Evaluation

The dataset was split into an 80% training set and a 20% testing set. Four regression models were trained:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regressor (SVR)

For models sensitive to feature scale (Linear Regression and SVR), the data was normalized using StandardScaler. Model performance was evaluated using three standard metrics:

1. **Mean Absolute Error (MAE):** The average absolute difference between predicted and actual values.
2. **Root Mean Squared Error (RMSE):** A measure of error that penalizes larger mistakes more heavily.
3. **R-squared (R²):** The proportion of the variance in the target variable that is predictable from the features.

### 3\. Results and Final Model Selection

#### 3.1 Model Comparison

The evaluation of the four baseline models yielded the following results:

| **Model** | **MAE** | **RMSE** | **R-squared** |
| --- | --- | --- | --- |
| **Decision Tree** | **0.133** | **0.810** | **0.9982** |
| Random Forest | 4.085 | 5.906 | 0.9035 |
| SVR | 4.260 | 6.289 | 0.8906 |
| Linear Regression | 5.437 | 7.223 | 0.8557 |

#### 3.2 Model Selection Analysis

While the Decision Tree model achieved a near-perfect score on the test set, this result is a strong indicator of **severe overfitting**. An unconstrained Decision Tree can memorize the specific noise and patterns of the training data, failing to generalize to new, unseen data.

Therefore, the **Random Forest Regressor** was selected as the final model. Its R-squared value of 0.9035 represents a more realistic and trustworthy measure of performance. The ensemble nature of Random Forest makes it inherently more robust and less prone to overfitting, providing the best balance between predictive accuracy and model generalization for a real-world application.

### 4\. System Deployment

#### 4.1 Model Persistence

To make the model operational, the trained Random Forest model and the StandardScaler object were serialized and saved to separate files (pmsm_random_forest_model.pkl and pmsm_scaler.pkl) using Python's pickle library. Saving both the model and the scaler is critical to ensure that new, incoming data is preprocessed in the exact same manner as the training data.

#### 4.2 Web Application Architecture

A web application was developed using the **Flask** micro-framework to provide an interactive interface for the model. The application consists of:

1. **app.py:** A Python backend script that handles HTTP requests, loads the pickle model and scaler artifacts, processes user input from the web form, and calls the model's predict function.
2. **templates/index.html:** A front-end HTML page styled with CSS. It provides a user-friendly form for inputting the 11 required motor parameters and a space to display the final prediction.

#### 4.3 User Workflow

The user navigates to the web application's home page, fills in the known operational parameters of the motor, and clicks the "Predict Temperature" button. The form data is sent to the Flask backend, which processes the data, runs the prediction, and re-renders the page with the estimated rotor temperature displayed clearly to the user.

### 5\. Conclusion

This project successfully achieved all its objectives. A highly accurate and robust predictive model for PMSM rotor temperature was developed, validated, and deployed. The selected Random Forest model demonstrated strong performance (R² = 0.9035), proving that machine learning is a viable and effective solution for virtual sensing in complex electromechanical systems. The deployed Flask application serves as a successful proof-of-concept for how such a model can be integrated into a practical tool for engineers and maintenance teams.
