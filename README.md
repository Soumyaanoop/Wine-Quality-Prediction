# Wine-Quality-Prediction

## OVERVIEW
This project applies machine learning classification techniques to predict the quality of wine based on its properties. Using Wine Quality Dataset, the project explores data preprocessing, feature selection and model building with algorithms such as KNN, Random Forest Classifier, and Gradient Boosting Classifier. Then evaluate model performance with metrics like accuracy, precision, recall, and F1-score. This project demonstrates the practical application of machine learning in the food and beverage industry for quality assessment and decision support.

## Goals

1. Data Understanding & Preprocessing 

   - Explore the physicochemical properties of wines from the dataset.  

   - Handle missing values, normalize features, and prepare the data for classification.  

3. Exploratory Data Analysis (EDA)

   - Visualize the distribution of wine quality scores.  

   - Identify correlations between features and wine quality.  

   - Highlight the most influential attributes (e.g., alcohol, acidity).  

5. Model Development 

    - Apply multiple machine learning classification algorithms (KNN,Random Forest Classifier,Gradient Boosting Classifier).  

   - Train and validate models using an 80-20 train/test split.  

   - Optimize performance with feature selection and hyperparameter tuning.  

7. Model Evaluation 

    - Assess performance using accuracy, precision, recall, F1-score, and confusion matrix.  

    - Compare models to determine the most effective classifier for wine quality prediction.  

9. Insights & Applications  

   - Demonstrate how machine learning can assist in automated quality assessment for the wine industry.  

   - Provide insights into which physicochemical features most strongly influence wine quality.  

11. Future Enhancements 
   
   - Explore advanced techniques such as deep learning models.  
   
   - Deploy the trained model as a user-friendly web application (Flask/Streamlit).
  
## Tools   

* Programming Language - Python

* Data manipulation & preprocessing - Pandas

* Numerical computations - Numpy

* Data visualization - Matplotlib, Seaborn

* Classification models, preprocessing, evaluation metrics - scikit-learn

* Development Environment - Jupyter notebook

## Dataset 

The Wine Quality Dataset contains physicochemical properties of wines along with a quality score assigned by experts. 
Number of Features: 11 physicochemical properties 

Features Included:  
                 
                 1. Fixed Acidity - Primarily tartaric acid, contributes to wine’s taste and stability. 
                 
                 2. Volatile Acidity -  Acetic acid in wine. Too much gives a vinegar taste (negative impact on quality). 
                 
                 3. Citric Acid - Adds freshness and flavor to wine; moderate levels are good.
                 
                 4. Residual Sugar - Amount of sugar left after fermentation. Some wines are intentionally sweet. 
                 
                 5. Chlorides - Salt content. High values can affect taste negatively.
                 
                 6. Free Sulfur Dioxide - Prevents microbial growth. Too much can affect aroma and taste. 
                 
                 7. Total Sulfur Dioxide - Total amount of SO₂ (free + bound). Excessive amounts are undesirable. 
                 
                 8. Density - Related to sugar/alcohol content. High density may suggest more sugar or lower alcohol.
                 
                 9. pH - Acidity level. Lower pH means more acidic wine. 
                
                10. Sulphates - Adds to wine preservation and may influence taste. Higher values can enhance quality.
                
                11. Alcohol -  Percentage of alcohol. Higher alcohol often improves perception of quality. 

Target Variable : `quality` – Wine quality score (0–10) rated by human tasters. Used as the target variable.

## Methodology

### Import
To begin the project,first import the essential Python libraries( Numpy,Pandas) that will help us handle data and perform computations. To better understand the dataset and identify patterns,import Python’s visualization libraries like Matplotlib,Seaborn. Next import StandardScaler that standardizes features by removing the mean and scaling to unit variance. Then import regression algorithms from the scikit-learn library. To measure how well these machine learning models predict insurance charges, import evaluation metrics like mean_absolute_error,mean_squared_error,r2_score from the scikit-learn library.

<img width="1402" height="144" alt="Screenshot 2025-09-03 at 12 26 58" src="https://github.com/user-attachments/assets/14baaad5-c7ac-4a55-a23e-2279e4bd8c19" />

<img width="1395" height="290" alt="Screenshot 2025-09-03 at 12 27 21" src="https://github.com/user-attachments/assets/287d913c-e849-4704-a64e-f40b72ec5ade" />


### load the dataset into our Python environment

<img width="1577" height="425" alt="Screenshot 2025-09-03 at 12 28 30" src="https://github.com/user-attachments/assets/f16cd380-8eec-45c6-8878-04247dd70ddd" />

### Understanding Dataset

checking how many no of columns and rows in the dataset

<img width="1294" height="101" alt="Screenshot 2025-09-03 at 12 43 30" src="https://github.com/user-attachments/assets/99878854-9b40-4a11-bb65-e837de04e531" />

check the columns names

<img width="1390" height="187" alt="Screenshot 2025-09-03 at 12 43 47" src="https://github.com/user-attachments/assets/0be1ceb2-7698-41c5-a8af-5234cb17a7a6" />


<img width="889" height="642" alt="Screenshot 2025-09-03 at 12 44 02" src="https://github.com/user-attachments/assets/1e3afc52-a3bd-4b39-9c1b-04dbfc401b1f" />


#### Handling Missing values


<img width="420" height="460" alt="Screenshot 2025-09-03 at 12 44 17" src="https://github.com/user-attachments/assets/cce9e361-0f16-44c8-b671-5f58d76f8d23" />

shows no missing values in this dataset

#### Finding Duplicates

<img width="434" height="40" alt="Screenshot 2025-09-03 at 12 44 44" src="https://github.com/user-attachments/assets/e2b5bc77-752e-4458-afae-7affd1f291b7" />

shows 240 rows are duplicates

Removing duplicates

<img width="543" height="44" alt="Screenshot 2025-09-03 at 12 45 18" src="https://github.com/user-attachments/assets/ae222dd7-953c-4629-a57a-145581e86b5c" />

#### check how values are spread for each feature


<img width="686" height="1107" alt="Screenshot 2025-09-03 at 13 22 42" src="https://github.com/user-attachments/assets/1764c822-9cb9-41a1-a61e-f662d17fadc2" />

<img width="657" height="1107" alt="Screenshot 2025-09-03 at 13 23 24" src="https://github.com/user-attachments/assets/d41db1dc-dc6c-4a3f-8b71-1f0479d4dc6b" />

<img width="670" height="1113" alt="Screenshot 2025-09-03 at 13 23 40" src="https://github.com/user-attachments/assets/a4eae245-5a3f-4f2a-9d1f-a0fa0950cbbf" />
<img width="673" height="1119" alt="Screenshot 2025-09-03 at 13 24 00" src="https://github.com/user-attachments/assets/8f927f7d-94df-419a-9456-64ed30cf9e36" />
<img width="667" height="1096" alt="Screenshot 2025-09-03 at 13 24 17" src="https://github.com/user-attachments/assets/bc2e4aa7-7237-449d-9546-d941b96cf2cc" />

<img width="677" height="555" alt="Screenshot 2025-09-03 at 13 24 32" src="https://github.com/user-attachments/assets/dc9f7105-c482-4db3-b4b1-554615e2ca88" />























