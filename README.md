🚗 Car Price Prediction Project Report
1. Project Objective
The objective of this project is to predict the selling price of used cars based on various factors such as manufacturing year, brand, mileage, engine power, and other key features.
The project applies machine learning algorithms to analyze car data and build models that estimate accurate resale prices.
2. Dataset Information
Column Name	Description
name	Full name of the car model (includes brand and variant). Example: Maruti Swift Dzire VDI. Used to extract the brand.
year	Year of manufacture. Newer cars tend to have higher resale value.
selling_price	The target variable — actual selling price of the car (in INR).
km_driven	Total distance driven in kilometers; higher values reduce price.
fuel	Type of fuel — Petrol, Diesel, CNG, LPG, or Electric.
seller_type	Type of seller — Individual, Dealer, or Trustmark Dealer.
transmission	Transmission type — Manual or Automatic.
owner	Ownership count — First Owner, Second Owner, etc.
mileage(km/ltr/kg)	Fuel efficiency of the car in km per liter or km per kg.
engine	Engine capacity in cubic centimeters (CC).
max_power	Maximum power produced by the engine (in bhp).
seats	Number of seats in the car.
brand	Derived feature from ‘name’, representing the car’s brand.
The dataset used in this project is obtained from CarDekho, a well-known car marketplace website. It contains details of cars sold in India, along with their technical specifications and ownership information.

Dataset Name: cardekho.csv
Total Rows: ~6000
Total Columns: 12

Column Descriptions
 
3. Feature Impact on Selling Price
Feature	Effect on Selling Price
year	Newer cars generally have higher resale value.
km_driven	More kilometers driven → lower price.
fuel	Petrol cars retain better resale value than diesel or CNG in many cases.
seller_type	Dealer cars may have slightly higher price due to warranty/service.
transmission	Automatic cars often priced higher than manual ones.
owner	More previous owners → lower price.
mileage(km/ltr/kg)	Better mileage increases resale value.
engine	Higher engine capacity increases performance and price.
max_power	Higher power usually means higher price.
seats	Slightly affects price depending on car type (SUV, Sedan, etc.).
brand	Premium brands have higher resale prices.
 





4. Data Cleaning
Extracted brand name from name column.
Converted max_power values to numeric (float).
Changed year column to integer type.
Dropped unnecessary name column.

 
 
5. Handling Missing Values
Missing values were filled using mean or mode strategies:
mileage(km/ltr/kg) → Mean
engine → Mean
seats → Mode
max_power → Mode

 
  





6. Removing Duplicates
Duplicate rows were identified and dropped to maintain data integrity.

 
 
7. Encoding Categorical Data
Categorical features (fuel, seller_type, transmission, owner, and brand) were converted into numeric form using Label Encoding.

 

8. Reducing Noise (Grouping Rare Brands)
Car brands with fewer than 30 occurrences were grouped under label 0, representing rare brands, to prevent overfitting and improve model stability.

 
 
 9. Feature Scaling
All numerical features were scaled between 0 and 1 using MinMaxScaler to bring them to the same range and help models converge faster.

 





10. Train-Test Split
The dataset was split into:
Training Set: 80%
Testing Set: 20%
This helps evaluate the model’s generalization ability.

 

11. Model Training and Evaluation
Five regression models were trained and tested:
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor

Evaluation Metrics
R² Score
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)

 

12. Model Comparison
A bar chart was created to compare R² Scores of all models.


 
13. Selling Price Distribution
A histogram with KDE curve shows how selling prices are distributed in the dataset.
 

 14. Correlation Heatmap
A heatmap displays the correlation among all numerical features, helping visualize how strongly each variable relates to selling price.

 
15. Actual vs Predicted (XGBoost)
A scatter plot compares actual vs predicted prices using the XGBoost model.
A red diagonal line indicates the ideal prediction line.

 
.


16. Conclusion
The XGBoost and Random Forest models achieved the highest accuracy and R² score, making them the best-performing models for this dataset.
Data cleaning, encoding, and feature scaling significantly improved model performance.
The project successfully demonstrates how machine learning can predict used car prices with high accuracy using real-world data.
 17. Future Enhancements
Add more real-world features like car condition, location, and service history.
Perform hyperparameter tuning for optimized results.
Deploy the model using Flask or Streamlit for user interaction.
Integrate the system with a live dataset API for real-time predictions.
