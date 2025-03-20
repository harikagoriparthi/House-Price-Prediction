 🏡 House Price Prediction using Linear Regression

 📌 Project Overview
This project builds a **machine learning model** to predict house prices based on area (sq. ft.) and the number of bedrooms using **Linear Regression**. The model is trained using **Python, Pandas, and Scikit-Learn**, with data visualization provided by **Matplotlib and Seaborn**.

 🚀 Tech Stack
- **Programming Language:** Python
- **Libraries Used:**
  - `numpy` (Numerical computations)
  - `pandas` (Data manipulation)
  - `matplotlib` & `seaborn` (Data visualization)
  - `scikit-learn` (Machine Learning model)

 📊 Dataset
The dataset consists of:
- **Area (sq. ft.)**
- **Number of Bedrooms**
- **Price (in Lakhs)**

Example Data:
```
   Area (sq ft)  Bedrooms  Price (Lakhs)
0          750         1             25
1          800         2             30
2          850         2             32
3          900         3             40
4          950         3             45
```

 🔧 How to Run the Code
1. **Clone the Repository**
   ```bash
   git clone https://github.com/harikagoriparthi/House-Price-Prediction.git
   cd House-Price-Prediction
   ```
2. **Install Required Libraries**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. **Run the Python Script**
   ```bash
   python house_price_prediction.py
   ```

 📈 Data Visualization
The project includes **data visualizations** to understand relationships between features:
1. **Pairplot:** Relationship between area, bedrooms, and price.
2. **Heatmap:** Correlation between features.
3. **Scatter Plot:** Price vs. Area.
4. **Actual vs Predicted Prices** visualization.

 🎯 Model Performance
After training the model, it evaluates performance using:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**

 📌 Sample Output
```
📊 Model Evaluation Metrics:
✅ Mean Absolute Error: 0.70 Lakhs
✅ Mean Squared Error: 0.50 Lakhs
✅ Model Coefficients: [0.066, 2.78]
✅ Model Intercept: -27.60
🏡 Predicted Price for 1200 sq ft, 3BHK: 59.99 Lakhs
```

 🤝 Contributing
Feel free to fork the repository and improve the model or dataset.

 📞 Contact
- **GitHub:** [harikagoriparthi](https://github.com/harikagoriparthi)
- **LinkedIn:** [(https://www.linkedin.com/in/goriparthi-harika-4a0a2928b/)]

📢 **If you like this project, don’t forget to star ⭐ the repository!**

