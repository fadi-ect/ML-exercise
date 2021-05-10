# By Fadi EL CHEIKH TAHA

# 1 - Data preprocessing
# ----------------------
# In a dataset, there is a feature named "Server/Machine Type", how would you
# transform/prepare this feature so that it can be used in a regression model
# (one only accepting float/bool as value), you don't have to code a solution,
# just write an answer on what you would do.

# Some example of values on this feature:
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 2950 MHz, 385570 MB RAM,  12079 MB swap
# Intel(R) Xeon(R) CPU E5-2670 v2 @ 2.50GHz (x86_64), 2500 MHz,  95717 MB RAM, 149012 MB swap
# Intel(R) Xeon(R) CPU E5-2697A v4 @ 2.60GHz (x86_64), 1300 MHz, 257868 MB RAM,  12027 MB swap
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 3138 MHz, 772642 MB RAM,   9042 MB swap
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 2214 MHz, 385570 MB RAM,  12090 MB swap
# Core(TM) i7-6700HQ CPU @ 2.60GHz (x86_64), 2600 MHz,  40078 MB RAM,  75183 MB swap
# Intel(R) Xeon(R) CPU E5-2697A v4 @ 2.60GHz (x86_64), 1199 MHz, 257868 MB RAM,  12247 MB swap
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 3246 MHz, 514658 MB RAM,  10770 MB swap
# Intel(R) Xeon(R) Gold 6142 CPU @ 2.60GHz (x86_64), 2483 MHz, 772642 MB RAM,   8266 MB swap

# Your full text (not python) answer:
"""
1. Place this features as it is in a table
2. Split with the delimiters (',' here)
3. Extract every columns and name it like this : Name, RAM speed, RAM, swap
4. Delete all parentheses and what it contains in the column 'Name', because they serve no purpose
5. Split column 'Name' at the '@', to create the column 'Name' and the column 'Power'
6. With the column 'Name' create booleans features for if it's a Xeon(1=True, 0=False), for the edition (Gold=1, CPU=0), name them 'Xeon' and 'Gold edition'
"""

# 2 - Regression
# --------------
# You are given a dataset (providing as additional file to this exercise) with 34 features
# and 1 target to predict, your task is to train a regression model on this dataset.
# Code it in python3, provide as well a requirements.txt containing the version you use
# I should be able to run directly in my linux terminal:
# -> pip install -r requirements.txt && python3 exercise_ml_internship_altair.py
# You are free to use every library you think is appropriate
"""
Here’s how I solved this ML exercise. I’m going straight to the point.
The Data Preparation part was very important, many options were available to me,
but finally I preferred to delete all missing values, I know that this 
is not necessarily the right decision but seeing that my missing values, for some features,
was up to 1.04% and that by removing them I had certain features that became unique,
so that’s the decision I’ve decided to make. I also decided to standardize my data, 
because that my variables seems to be measured at different scales do not contribute equally 
to the model fitting & model learned function and might end up creating a bias.
With this data preparation I was able to train my models, with a score of 0.95, and display my error which is 2.77
"""

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def missing_values_table(df):

    mis_val = df.isnull().sum()

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})
    
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(2)
    
    # Print some summary information
    print("Your selected dataframe has {} columns.".format(df.shape[1]) + '\n' + 
    "There are {} columns that have missing values.".format(mis_val_table_ren_columns.shape[0]))
    
    # Return the dataframe with missing information
    return mis_val_table_ren_columns

def remove_constant_value_features(df):
    return [e for e in df.columns if df[e].nunique() == 1]

def fit_regression(X, y):
    # Reshape input for sklearn
    if len(X.shape) < 2:
        X = X.reshape(-1, 1)
    
    # Fit regression
    model = LinearRegression()
    reg = model.fit(X, y)
    
    # Compute score
    print(f"Regression score: {reg.score(X, y)}")
        
    return reg

###########################################################################################################################
# 2.1 loading dataset
print("\n-----------------")
print("|Loading Dataset|")
print("-----------------\n")
df = pd.read_csv('dataset_ml_exo.csv').drop(columns=['Unnamed: 0'])\
    .apply(pd.to_numeric, errors='coerce')
#print(df)
#print(df.dtypes)
#print(df.describe().transpose())
print(df.info())
###########################################################################################################################
# 2.2 data preparation
print("\n------------------")
print("|Data Preparation|")
print("------------------\n")


print('This is a before/after for my missing values, we can see that i have no more\n')
print(missing_values_table(df))
print('\n')

df_without_null = df.dropna()
missing_values_table(df_without_null)

# I delete my constant features because they will explain nothing for my model
drop_col = remove_constant_value_features(df_without_null)#these are the features that i have to delete
#print(drop_col) 

df_without_null_and_constant = df_without_null.drop(['feature_12', 'feature_16', 'feature_19', 'feature_21'], axis = 1)
#print(df_without_null_and_constant)

# Also can be deleted
df_without_null_and_constant['feature_1'].value_counts()

df_without_null_and_constant = df_without_null_and_constant.drop(['feature_1'], axis = 1)
#print(df_without_null_and_constant.dtypes)

# These are the features that I keep
# I keep them in 'keepCols' to be sure that my Standardization will not change their names
keepCols = ['feature_0', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
            'feature_7','feature_8', 'feature_9', 'feature_10', 'feature_11', 'feature_13',
            'feature_14', 'feature_15', 'feature_17', 'feature_18', 'feature_20','feature_22',
            'feature_23', 'feature_24', 'feature_25','feature_26', 'feature_27', 'feature_28',
            'feature_29', 'feature_30', 'feature_31', 'feature_32','feature_33', 'target']

scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_without_null_and_constant[keepCols])
scaled_df = pd.DataFrame(scaled_df, columns=keepCols)

# Correlation with output variable
corr_matrix = scaled_df.corr()
cor_target = abs(corr_matrix["target"])
relevant_features = cor_target[(cor_target>0.5) & (cor_target<1)]#Selecting highly correlated features
relevant_features

df_final = scaled_df[['feature_0', 'feature_3', 'feature_4', 'feature_5', 'feature_6',
                      'feature_7', 'feature_9', 'feature_10','feature_11', 'feature_13',
                      'feature_15', 'feature_20', 'feature_25', 'feature_26', 'feature_27',
                      'feature_33', 'target']]
#print(df_final)
print("\nThis is my final df after I deleted the constant values and make my Standardization\n")
print(df_final.info())
###########################################################################################################################
# 2.3 model training
print("\n----------------")
print("|model training|")
print("----------------\n")

X = np.array(df_final.iloc[:, :-1])
y = np.array(df_final.iloc[:, -1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the regressor to the training data
reg = fit_regression(X_train, y_train)
print('A resultat like this can be explained by the fact that my standardization has maybe biased, a little, my results, let\'s see without it')
###########################################################################################################################
# 2.4 model evaluation (evaluate model perf and display metrics)
print("\n------------------")
print("|model evaluation|")
print("------------------\n")

# Predict on the test data: y_pred
y_pred = reg.predict(X_test) 

print('This is the RMSE after the standardization')
# Compute and print RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))# RMSE allow me to measure the differences between the values predicted by my model and the observed values (2.77 here, it's a good result)

print('\nThis is the regression score and RMSE before the standardization')
X_2 = np.array(df_without_null_and_constant.iloc[:, :-1])
y_2 = np.array(df_without_null_and_constant.iloc[:, -1])

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X_2, y_2, test_size=0.2, random_state=42)

reg_2 = fit_regression(X_train_2, y_train_2)

y_pred_2 = reg_2.predict(X_test_2)
rmse_2 = np.sqrt(mean_squared_error(y_test_2, y_pred_2))
print("Root Mean Squared Error before Standardization: {}".format(rmse_2))

print('\nRegressions scores are almost the same and this proves that standardization has no influence')
print('However, RMSE is really higher due to excessive fluctuations frome one features to another, and we can see how important it is')
###########################################################################################################################

# This part goes to the part 'Data preparation' but I put it in the end otherwise it will block the execution of the rest of the code
# Plots the before/after standardization
plt.rcParams["figure.figsize"]=[15,10]
plt.subplot(211)
plt.title('Distribution of Data before Standardization')
sns.kdeplot(data=df_without_null_and_constant[keepCols], legend=None, palette="Paired")
plt.subplot(212)
plt.title('Distribution of Data after Standardization')
sns.kdeplot(data=scaled_df, legend=None, palette="Paired")
plt.show(block=True)
