### **Traffic Congestion Prediction using Random Forest Classifier**

This project uses a **Random Forest Classifier** to predict traffic congestion levels in Bangalore based on historical traffic data. The model predicts congestion as one of three categories: **Low**, **Medium**, or **High** based on factors like weather conditions, roadwork, and intersection data.

### **Key Features:**

*   **Traffic Data Preprocessing**: The dataset is preprocessed by encoding categorical variables (e.g., Area Name, Road Name, Weather, Roadwork) using **Label Encoding**. Additionally, traffic congestion levels are categorized into **Low**, **Medium**, and **High**.
    
*   **Model Training**: The model uses a **Random Forest Classifier** trained on various features, including **date**, **area name**, **road/intersection name**, **weather conditions**, and **roadwork activity**.
    
*   **Interactive Web Application**: Built using **Streamlit**, the app allows users to input traffic details (such as area, weather, and date) and get predictions for the expected traffic congestion. The results are displayed visually using **Plotly**:
    
    *   A **gauge chart** showing predicted congestion level (Low, Medium, or High).
        
    *   A **bar chart** displaying the model's confidence in its predictions.
        

### **Workflow:**

1.  **Data Loading**: The dataset is loaded from a specified file path and preprocessed.
    
2.  **Model Prediction**: Upon user input, the trained Random Forest model makes predictions about traffic congestion.
    
3.  **Visualization**: A gauge chart and a bar chart are generated to visualize the predicted congestion level and prediction confidence, respectively.
    

### **Requirements:**

*   Streamlit
    
*   Pandas
    
*   NumPy
    
*   Plotly
    
*   Scikit-learn
    
*   Joblib
    

### **Usage**:

1.  Clone the repository.
    
2.  Install the required dependencies using pip install -r requirements.txt.
    
3.  Run the app using streamlit run app.py.
    
4.  Input the desired traffic parameters and view the predicted congestion level.