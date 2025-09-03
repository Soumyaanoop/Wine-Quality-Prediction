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

Find how many no of columns and rows in the dataset

<img width="1294" height="101" alt="Screenshot 2025-09-03 at 12 43 30" src="https://github.com/user-attachments/assets/99878854-9b40-4a11-bb65-e837de04e531" />

check the columns names

<img width="1390" height="187" alt="Screenshot 2025-09-03 at 12 43 47" src="https://github.com/user-attachments/assets/0be1ceb2-7698-41c5-a8af-5234cb17a7a6" />



Summary of the dataset


<img width="889" height="642" alt="Screenshot 2025-09-03 at 12 44 02" src="https://github.com/user-attachments/assets/1e3afc52-a3bd-4b39-9c1b-04dbfc401b1f" />


### Exploratory Data Analysis


#### Handling Missing values


<img width="420" height="460" alt="Screenshot 2025-09-03 at 12 44 17" src="https://github.com/user-attachments/assets/cce9e361-0f16-44c8-b671-5f58d76f8d23" />


It shows no missing values in this dataset


#### Finding Duplicates

<img width="434" height="40" alt="Screenshot 2025-09-03 at 12 44 44" src="https://github.com/user-attachments/assets/e2b5bc77-752e-4458-afae-7affd1f291b7" />

shows 240 rows are duplicates

Removing duplicates

<img width="543" height="44" alt="Screenshot 2025-09-03 at 12 45 18" src="https://github.com/user-attachments/assets/ae222dd7-953c-4629-a57a-145581e86b5c" />


#### check how values are spread for each feature

Knowing the distribution of features is important because it helps identify skewness or outliers in data. Also it can suggest if scaling or transformations (like log, standardization) are needed.

 <img width="1199" height="225" alt="Screenshot 2025-09-03 at 13 22 14" src="https://github.com/user-attachments/assets/8616697f-36b9-4cf0-ab6b-549fa8f2f0f0" />

loop iterates over through all independent features in the wine dataset. Uses Seaborn’s histplot to plot the distribution of each feature. kde=True → This makes it easier to understand the underlying distribution pattern.


<img width="669" height="520" alt="Screenshot 2025-09-03 at 17 30 54" src="https://github.com/user-attachments/assets/a05b0d3d-da3c-436a-83e9-8680c67ca40c" />    <img width="672" height="551" alt="Screenshot 2025-09-03 at 17 31 17" src="https://github.com/user-attachments/assets/f35df8ed-7337-46ff-9557-c94fc4d9bccc" />


<img width="670" height="541" alt="Screenshot 2025-09-03 at 17 31 39" src="https://github.com/user-attachments/assets/ff422573-8c30-45b1-b9d7-0a3ac98e0978" />     <img width="676" height="534" alt="Screenshot 2025-09-03 at 17 31 57" src="https://github.com/user-attachments/assets/bdef4c10-8136-4748-9123-10c74c9c89a2" />


<img width="677" height="536" alt="Screenshot 2025-09-03 at 17 32 17" src="https://github.com/user-attachments/assets/fe3508b2-df19-498b-84f1-9ba8e1be08ab" />      <img width="676" height="542" alt="Screenshot 2025-09-03 at 17 32 36" src="https://github.com/user-attachments/assets/fb4ebb06-d501-4915-a78c-da301f5839dd" />


<img width="697" height="1087" alt="Screenshot 2025-09-03 at 17 32 58" src="https://github.com/user-attachments/assets/54e6bff2-6348-4245-a262-5f435f3fa978" />

<img width="697" height="1098" alt="Screenshot 2025-09-03 at 17 33 18" src="https://github.com/user-attachments/assets/c2fb412a-6286-4e2a-b56d-ee4278dd0f6e" />

<img width="687" height="551" alt="Screenshot 2025-09-03 at 17 33 33" src="https://github.com/user-attachments/assets/46837ea0-84f4-4885-b8f0-c9885b703318" />

#### Find Relationships Between Features using Correlation Matrix

<img width="1154" height="161" alt="Screenshot 2025-09-03 at 18 03 43" src="https://github.com/user-attachments/assets/9d060f85-31dd-45fe-99b7-f5dc9a458fe8" />

<img width="1345" height="966" alt="Screenshot 2025-09-03 at 18 04 37" src="https://github.com/user-attachments/assets/3f82ee83-fc76-4ff2-b5b8-2fd28d9771ef" />

##### OBSERVATIONS FROM CORRELATION MATRIX

* fixed acidity vs citric acid	+0.67	Wines with more fixed acidity tend to have more citric acid (they’re both acids).

* fixed acidity ↔ density	+0.67	Higher acid levels make the wine slightly denser.

* free SO₂ ↔ total SO₂	+0.67	Total SO₂ is largely driven by the free portion.

* alcohol ↔ quality	+0.48	Higher‐alcohol wines are generally rated better.

* pH ↔ fixed acidity	−0.69	As fixed acidity goes up, pH (acidity scale) goes down—chemically consistent.

* pH ↔ density (r ≈ −0.36): Denser wines (more sugar/solids) tend to be more acidic (lower pH).

Check Redundancy (Multicollinearity)

If two features are very strongly correlated (r ≳ 0.8), consider dropping one:

Here, free SO₂ vs total SO₂ (r ≈ 0.67)—not extreme, but keep an eye on it.

fixed acidity vs citric acid are also strongly linked  so drop one or combine them.


Relate to Your Target (quality) . Look at correlations with quality:

* alcohol	+0.48	Good indicator of perceived quality

* volatile acidity	−0.40	Vinegary taste lowers quality

* sulphates	 +0.25	Somewhat positive (preservative effect)


#### Visualize Relationship 

Visualization supports the numerical correlation values we saw in the heatmap. 

##### Visualize how Alcohol content affects wine Quality

Below scatterplot shows how individual wines are distributed with respect to alcohol content (x-axis) and quality score (y-axis). sns.regplot(..., scatter=False) draws a regression line (best fit line). The red line shows the overall trend in the relationship.

<img width="1155" height="132" alt="Screenshot 2025-09-03 at 20 32 19" src="https://github.com/user-attachments/assets/1539d64a-84fc-46b6-9390-81967d35beb3" />

<img width="701" height="682" alt="Screenshot 2025-09-03 at 20 36 45" src="https://github.com/user-attachments/assets/0c3d6ded-be10-4952-b6b4-3420e38f0d9e" />


##### Visualize how volatile acidity affects wine quality


<img width="693" height="675" alt="Screenshot 2025-09-03 at 22 19 36" src="https://github.com/user-attachments/assets/981e6538-66c5-45c2-b743-98751872a78d" /> 


#### Detect and Visualize outliers

<img width="1136" height="496" alt="Screenshot 2025-09-03 at 22 23 29" src="https://github.com/user-attachments/assets/10bf87a0-259d-442f-944c-75e70dbe6a8c" />


<img width="1201" height="1022" alt="Screenshot 2025-09-03 at 22 25 30" src="https://github.com/user-attachments/assets/2ac955f2-01c5-4f7d-8a99-639396d9f2f2" />



### Data Preprocessessing

#### Transforming highly skewed features

First identify  Features with high skewness 

<img width="1267" height="448" alt="Screenshot 2025-09-03 at 22 33 12" src="https://github.com/user-attachments/assets/eb6cdfab-bb85-4d72-8753-053d5198119f" />


Transforming highly skewed features.(>_70) use np.sqrt when skew is moderate and that column data ≥ 0.


<img width="1064" height="437" alt="Screenshot 2025-09-03 at 22 34 56" src="https://github.com/user-attachments/assets/c1dc5e2d-4516-4e07-af69-c8ca521b9373" />


#### Converting the numeric wine quality scores into categorical labels

The original dataset has wine quality as a numeric rating (0–10). But ML classification projects work better with categorical labels instead of many small numeric classes.
So I group the scores into 3 broader categories:

Low quality → 0–4

Medium quality → 5–7

High quality → 8–10

Here I create a function that takes a wine’s quality numeric score as input and returns one of the three categories.

<img width="707" height="267" alt="Screenshot 2025-09-03 at 22 43 36" src="https://github.com/user-attachments/assets/0c976c05-a18b-43c0-8215-246514b0ddfd" />

apply this function to the dataframe wine_df

<img width="903" height="64" alt="Screenshot 2025-09-03 at 22 44 12" src="https://github.com/user-attachments/assets/eb02cc54-2d2c-4986-889b-e4af8e892444" />



#### Removing low variance feature based on correlation matrix

The feature 'residual sugar' has very low correlation with the target (quality). It means they don’t help in predicting it.


<img width="753" height="54" alt="Screenshot 2025-09-03 at 22 52 14" src="https://github.com/user-attachments/assets/758536e2-6102-4a7f-8016-17692b8a92c9" />







































