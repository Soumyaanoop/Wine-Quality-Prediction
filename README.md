# Wine-Quality-Prediction

## OVERVIEW
This project applies machine learning classification techniques to predict the quality of wine based on its properties. Using Wine Quality Dataset, the project explores data preprocessing, feature selection and model building with algorithms such as KNN, Random Forest Classifier, and Gradient Boosting Classifier. Then evaluate model performance with metrics like accuracy, precision, recall, and F1-score. This project demonstrates the practical application of machine learning in the food and beverage industry for quality assessment and decision support.

## Goals

1. Data Understanding & Preprocessing 

   - Explore the physicochemical properties of wines from the dataset.  

   - Handle missing values, normalize features, and prepare the data for classification.  

2. Exploratory Data Analysis (EDA)

   - Visualize the distribution of wine quality scores.  

   - Identify correlations between features and wine quality.  

   - Highlight the most influential attributes (e.g., alcohol, acidity).  

3. Model Development 

    - Apply multiple machine learning classification algorithms (KNN,Random Forest Classifier,Gradient Boosting Classifier).  

   - Train and validate models using an 80-20 train/test split.  

   - Optimize performance with feature selection and hyperparameter tuning.  

4. Model Evaluation 

    - Assess performance using accuracy, precision, recall, F1-score, and confusion matrix.  

    - Compare models to determine the most effective classifier for wine quality prediction.  

5. Insights & Applications  

   - Demonstrate how machine learning can assist in automated quality assessment for the wine industry.  

   - Provide insights into which physicochemical features most strongly influence wine quality.  


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

<img width="758" height="144" alt="Screenshot 2025-09-04 at 09 00 59" src="https://github.com/user-attachments/assets/8cb3ac0a-865f-4d3d-89b6-3d6505896844" />


<img width="1368" height="301" alt="Screenshot 2025-09-04 at 09 01 09" src="https://github.com/user-attachments/assets/82b0e75d-aa21-4879-8393-97cddb6ee586" />



### load the dataset into our Python environment

<img width="1521" height="421" alt="Screenshot 2025-09-04 at 09 01 26" src="https://github.com/user-attachments/assets/bb93ee3a-f7ee-447f-b81c-c9b38af36ef6" />


### Understanding Dataset

Find how many no of columns and rows in the dataset

<img width="339" height="114" alt="Screenshot 2025-09-04 at 09 01 39" src="https://github.com/user-attachments/assets/25291558-23ec-4477-9253-001bc40dfeb4" />


check the columns names

<img width="1096" height="192" alt="Screenshot 2025-09-04 at 09 02 04" src="https://github.com/user-attachments/assets/17011676-e37e-4f9d-934d-34986b7e66bf" />


Summary of the dataset

<img width="882" height="631" alt="Screenshot 2025-09-04 at 09 02 18" src="https://github.com/user-attachments/assets/4a76d40a-d179-4df1-85dc-2d5913e6b292" />



### Exploratory Data Analysis



#### Handling Missing values


<img width="590" height="470" alt="Screenshot 2025-09-04 at 09 02 28" src="https://github.com/user-attachments/assets/0ff70006-94da-4e9d-9f97-5abecd1bd315" />


It shows no missing values in this dataset


#### Finding Duplicates & Removing duplicates


<img width="458" height="51" alt="Screenshot 2025-09-04 at 09 02 41" src="https://github.com/user-attachments/assets/a9d725d4-aeec-4042-bd8f-152cadf58dc5" />

shows 240 rows are duplicates

Removing duplicates

<img width="566" height="48" alt="Screenshot 2025-09-04 at 09 02 51" src="https://github.com/user-attachments/assets/0dc7b7bf-6575-4322-96ea-e91ea8f4bf2d" />


#### check how values are distributed for each feature

Knowing the distribution of features is important because it helps identify skewness or outliers in data. Also it can suggest if scaling or transformations (like log, standardization) are needed.

 <img width="873" height="189" alt="Screenshot 2025-09-04 at 09 03 25" src="https://github.com/user-attachments/assets/ae392494-cebf-4ef9-b432-214653945519" />


loop iterates over through all independent features in the wine dataset. Uses Seaborn’s histplot to plot the distribution of each feature. kde=True → This makes it easier to understand the underlying distribution pattern.

<img width="456" height="402" alt="Screenshot 2025-09-06 at 21 58 23" src="https://github.com/user-attachments/assets/7a8d1d1c-3afd-4353-83b8-025304286f17" />

Fixed Acidity : The distribution is right-skewed, with most wines having acidity between 6 and 10.This indicates that extreme acidity values are rare.


<img width="485" height="403" alt="Screenshot 2025-09-06 at 21 58 41" src="https://github.com/user-attachments/assets/1264022a-106d-4e3f-9db1-2f742ac8fef6" />

Volatile Acidity : Distribution is also right-skewed. Most values fall between 0.3 and 0.7, with few extreme values above 1.
Since volatile acidity contributes to unpleasant vinegar-like taste, most wines stay within acceptable ranges.


<img width="452" height="412" alt="Screenshot 2025-09-06 at 21 58 58" src="https://github.com/user-attachments/assets/1298b1b2-0aa7-4258-bf9a-c9b85528a19a" />

The distribution shows a high concentration near 0 means many wines contain little or no citric acid. Indicates citric acid is not a dominant feature in most wines.

<img width="484" height="412" alt="Screenshot 2025-09-06 at 21 59 15" src="https://github.com/user-attachments/assets/b77a668f-bb62-4351-b56a-34b4b65c8123" />

Residual Sugar: Strong right-skewness. A few wines representing sweet wines.

<img width="447" height="413" alt="Screenshot 2025-09-06 at 21 59 27" src="https://github.com/user-attachments/assets/3de464e8-fd1e-427f-9f84-6bb769a8b421" />

Chlorides : The distribution is heavily right-skewed. Most wines have chloride content close to 0.04, with rare extreme cases above 0.2.

<img width="530" height="396" alt="Screenshot 2025-09-06 at 21 59 41" src="https://github.com/user-attachments/assets/5c0c69f7-3295-4ff6-a0ac-f12efcb8fc9a" />

Free Sulfur Dioxide : The distribution is skewed to the right. Most values lie between 0 and 30


<img width="522" height="407" alt="Screenshot 2025-09-06 at 22 00 09" src="https://github.com/user-attachments/assets/c83c50f3-c648-47a8-9dba-6120eac85c8e" />

Total Sulfur Dioxide : Similar to free SO₂, but with wider spread. Most values lie between 20 and 100, with a long tail extending beyond 200.

<img width="456" height="405" alt="Screenshot 2025-09-06 at 22 00 25" src="https://github.com/user-attachments/assets/bd3a9cc0-27bc-439f-bf11-78bd8b448b14" />

Density : The distribution is nearly normal (bell-shaped). This reflects the typical density of wine, with only small variations.

<img width="436" height="400" alt="Screenshot 2025-09-06 at 22 00 45" src="https://github.com/user-attachments/assets/57b0a3a3-7c46-4daa-9807-ba9447939cba" />

pH : The pH distribution is close to normal. Indicates most wines have typical acidity levels.

<img width="458" height="418" alt="Screenshot 2025-09-06 at 22 01 00" src="https://github.com/user-attachments/assets/73f34d10-988f-493e-b330-e24610d830ee" />

Sulphates : Right-skewed distribution.

<img width="439" height="410" alt="Screenshot 2025-09-06 at 22 01 13" src="https://github.com/user-attachments/assets/fd8cd6fa-2d98-4eb0-972e-705d8ae9483a" />

The alcohol content ranges from around 8% to 15%, with the majority of wines clustered between 9% and 12%.
The distribution is slightly right-skewed, meaning there are more wines with lower alcohol levels, and fewer wines with very high alcohol content.


#### Find Relationships Between Features using Correlation Matrix

The correlation matrix provides insights into how different chemical properties of wine are related to each other and to wine quality.

<img width="1154" height="161" alt="Screenshot 2025-09-03 at 18 03 43" src="https://github.com/user-attachments/assets/9d060f85-31dd-45fe-99b7-f5dc9a458fe8" />

<img width="1345" height="966" alt="correlation_matrix_wine" src="https://github.com/user-attachments/assets/8fb6ec29-e95e-43bf-b282-1eb6a2656904" />



#### OBSERVATIONS FROM CORRELATION MATRIX

Relate to Target (quality) . Look at correlations with quality:

* alcohol	+0.48	Good indicator of perceived quality

* volatile acidity	−0.40	Vinegary taste lowers quality

* sulphates	 +0.25	Somewhat positive (preservative effect)

Check Redundancy (Multicollinearity)

If two features are very strongly correlated (r ≳ 0.8), consider dropping one:

Here, free SO₂ vs total SO₂ (r ≈ 0.67)—not extreme, but keep an eye on it.

fixed acidity vs citric acid are also strongly linked  so drop one or combine them.

Summary

Most Influential Features for Quality Prediction: Alcohol, Volatile Acidity, Sulphates, Citric Acid.

Redundant/Highly Correlated Features: Free Sulfur Dioxide and Total Sulfur Dioxide; Fixed Acidity and Density. These may require dimensionality reduction or careful feature selection.

Negligible Features: Residual sugar and chlorides may contribute little to quality prediction.



#### Visualize how Alcohol content affects wine Quality

Below scatterplot shows how individual wines are distributed with respect to alcohol content (x-axis) and quality score (y-axis). sns.regplot(..., scatter=False) draws a regression line (best fit line). The red line shows the overall trend in the relationship.

<img width="1155" height="132" alt="Screenshot 2025-09-03 at 20 32 19" src="https://github.com/user-attachments/assets/1539d64a-84fc-46b6-9390-81967d35beb3" />

<img width="701" height="682" alt="Alcoholvsquality" src="https://github.com/user-attachments/assets/59c0ccc0-1f6e-4e12-a5b9-2e6cf413e87a" />


The regression line (in red) indicates a clear upward trend: as the alcohol percentage increases, the wine quality scores also tend to increase. 




#### Visualize how volatile acidity affects wine quality


<img width="693" height="675" alt="Screenshot 2025-09-03 at 22 19 36" src="https://github.com/user-attachments/assets/981e6538-66c5-45c2-b743-98751872a78d" /> 

<img width="693" height="675" alt="acidityvsquality" src="https://github.com/user-attachments/assets/b006cef9-b769-40b4-b355-943fd987a64f" />



#### Detect and Visualize outliers

<img width="1136" height="496" alt="Screenshot 2025-09-03 at 22 23 29" src="https://github.com/user-attachments/assets/10bf87a0-259d-442f-944c-75e70dbe6a8c" />


<img width="1201" height="1022" alt="Screenshot 2025-09-03 at 22 25 30" src="https://github.com/user-attachments/assets/2ac955f2-01c5-4f7d-8a99-639396d9f2f2" />



### Data Preprocessessing


#### Transforming highly skewed features

First identify  Features with high skewness 


<img width="1230" height="498" alt="Screenshot 2025-09-04 at 09 17 22" src="https://github.com/user-attachments/assets/14e68fa6-1269-42de-9049-2f0ddaf0a5ec" />


Transforming highly skewed features.(>_70) use np.sqrt when skew is moderate and that column data ≥ 0.


<img width="1128" height="440" alt="Screenshot 2025-09-04 at 09 18 15" src="https://github.com/user-attachments/assets/b682f641-8373-49ff-9400-d054637e4926" />


#### Converting the numeric wine quality scores into categorical labels

The original dataset has wine quality as a numeric rating (0–10). But ML classification projects work better with categorical labels instead of many small numeric classes.
So I group the scores into 3 broader categories:

Low quality → 0–4

Medium quality → 5–7

High quality → 8–10

Here I create a function that takes a wine’s quality numeric score as input and returns one of the three categories.

<img width="1154" height="283" alt="Screenshot 2025-09-04 at 09 20 29" src="https://github.com/user-attachments/assets/a70efbad-ffd3-456d-85ed-b4493abbb413" />

apply this function to the dataframe wine_df

<img width="979" height="52" alt="Screenshot 2025-09-04 at 09 20 40" src="https://github.com/user-attachments/assets/8a1296bd-4d73-4627-8de6-3a0f715aca52" />



#### Removing low variance feature based on correlation matrix

The feature 'residual sugar' has very low correlation with the target (quality). It means they don’t help in predicting it.


<img width="753" height="54" alt="Screenshot 2025-09-03 at 22 52 14" src="https://github.com/user-attachments/assets/758536e2-6102-4a7f-8016-17692b8a92c9" />


#### In machine learning,always separate independent variables(X) and targets (Y)


<img width="1156" height="80" alt="Screenshot 2025-09-04 at 10 50 42" src="https://github.com/user-attachments/assets/58c03dc5-5269-46c2-b4c3-6c20a12c91b3" />

#### Splitting data

<img width="1207" height="85" alt="Screenshot 2025-09-04 at 10 50 55" src="https://github.com/user-attachments/assets/b6c89ca2-f6f5-4328-aa18-3203ea9aac33" />


#### Feature Scaling

Scaling ensures all features contribute fairly.

<img width="675" height="111" alt="Screenshot 2025-09-04 at 10 52 54" src="https://github.com/user-attachments/assets/39bd1745-40fd-4d8b-9682-532056c0f1a3" />

 
Computes the mean (μ) and standard deviation (σ) of each feature in the training set and applies scaling using those values.


#### Model Training

Model training using K-Nearest Neighbors classifier

<img width="1050" height="91" alt="Screenshot 2025-09-06 at 19 45 22" src="https://github.com/user-attachments/assets/33f3c2b0-afee-49e3-ba26-c990d9acb448" />


The trained K-Nearest Neighbors classifier (Knn) to make predictions on the test dataset (X_test).

<img width="477" height="54" alt="Screenshot 2025-09-06 at 19 49 10" src="https://github.com/user-attachments/assets/5ad3d79b-689b-4d99-ac8f-5ce629e451bc" />


Model training using Random Forest classifier

<img width="896" height="97" alt="Screenshot 2025-09-06 at 19 50 22" src="https://github.com/user-attachments/assets/a5ec0225-dbd4-4c68-810e-8a76cd0fd979" />

The trained Random Forest classifier to make predictions on the test dataset (X_test).

<img width="562" height="75" alt="Screenshot 2025-09-06 at 19 50 46" src="https://github.com/user-attachments/assets/f48b8b65-8462-4d46-bb40-4146463a4e26" />


Model training using GradientBoosting classifier

<img width="1356" height="86" alt="Screenshot 2025-09-06 at 19 54 25" src="https://github.com/user-attachments/assets/98106039-6758-463e-8b9b-cd6732538158" />


The trained GradientBoosting classifier to make predictions on the test dataset (X_test).

<img width="488" height="74" alt="Screenshot 2025-09-06 at 19 55 04" src="https://github.com/user-attachments/assets/2dedc03a-4a22-4f6b-bc58-da365c0657a7" />


#### Evaluating Model

##### To evaluate the performance of the K-Nearest Neighbors (KNN) Classifier

First,used a confusion matrix. The matrix compares the actual wine quality categories (low, medium, high) with the model’s predictions. The diagonal cells 
represent correct classifications, while off-diagonal cells represent misclassifications.


<img width="542" height="533" alt="confusion_matrix_wine" src="https://github.com/user-attachments/assets/2a7154fa-9938-4eab-bbc0-c792beeede21" />


The model has a bias toward predicting the "High-quality" class (Class 2). Out of 254 actual high-quality wines, all were correctly predicted as High.

It achieves perfect accuracy for high-quality wines but completely fails for low- and medium-quality wines.

While the overall accuracy may appear high, the model’s performance is poor for minority classes, making it unsuitable for real-world scenarios where accurate classification across all quality levels is essential.



<img width="953" height="360" alt="Accuracy" src="https://github.com/user-attachments/assets/4510e0f1-6255-42f3-bb09-9252446b964c" />


The KNN model achieved a high overall accuracy of 93%, which at first glance suggests good performance. However, when I examine precision (31%) and recall (33%), it becomes clear that the model performs poorly on minority classes (low and medium wine quality).


##### To evaluate the performance of the Random Forest Classifier


<img width="621" height="549" alt="Confusion_matrix_wine2" src="https://github.com/user-attachments/assets/bbcc9067-3ac6-43e7-a419-5914e4eb868d" />


The model achieves perfect accuracy for class 2 but performs poorly on classes 0 and 1. This indicates that the model is heavily biased toward predicting class 2, regardless of whether the true class is 0, 1, or 2.


<img width="977" height="383" alt="Accuracy2" src="https://github.com/user-attachments/assets/c72cd96e-8ace-4bfd-b665-f0d93a895577" />


Although the model performs extremely well in identifying class 2, it fails to correctly classify any samples from classes 0 and 1. This imbalance leads to deceptively high accuracy, but low precision and recall when considering all classes equally.



##### To evaluate the performance of the Gradient Boosting Classifier


<img width="593" height="551" alt="confusion_matrix_wine3" src="https://github.com/user-attachments/assets/4138cdeb-ef51-4587-88d7-f8c644e402ed" />

Since class 2 dominates the dataset, the model achieves a high number of correct predictions overall. The model consistently predicts class 2 for most inputs, including many belonging to classes 0 and 1.



<img width="902" height="376" alt="Accuracy3" src="https://github.com/user-attachments/assets/98987771-0688-4fb0-bc0c-19d3194d25f7" />

GBC has slightly lower accuracy (91%) but better precision (0.38) and recall (0.35).

This shows that GBC is slightly better at handling minority classes (0 and 1), even though the dataset imbalance still strongly affects the results.


Recommendations : Apply techniques such as SMOTE (Synthetic Minority Over-sampling Technique), ADASYN, or class-weight adjustments in the classifier to give more importance to minority classes.


