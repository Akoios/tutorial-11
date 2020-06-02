import numpy as np 
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://storage.googleapis.com/tutorial-datasets/telco.csv")

# (1) Lowercase transformation
for item in df.columns:
    try:
        df[item] = df[item].str.lower()
    except:
        print(item, "Unable to convert")

# (2) Binary conversion of relevant features so we can use them for the classification
columns_to_convert = ['Partner', 
                      'Dependents', 
                      'PhoneService', 
                      'PaperlessBilling', 
                      'Churn']

for item in columns_to_convert:
    df[item].replace(to_replace='yes', value=1, inplace=True)
    df[item].replace(to_replace='no',  value=0, inplace=True)

# (3) Transform Total Charges to Float
df['TotalCharges'] = df['TotalCharges'].replace(r'\s+', np.nan, regex=True)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# (4) Check for null data points
df.isnull().sum(axis = 0)
df = df.fillna(value=0)

# (5) Balance the labels
churners_number = len(df[df['Churn'] == 1])
print("Number of churners", churners_number)

churners = (df[df['Churn'] == 1])

non_churners = df[df['Churn'] == 0].sample(n=churners_number)
df2 = churners.append(non_churners)

try:
    customer_id = df2['customerID'] # Store this as customer_id variable
    del df2['customerID'] # Not needed
except:
    print("already removed customerID")
    
# (6) One-hot encoding
ml_dummies = pd.get_dummies(df2)
ml_dummies.fillna(value=0, inplace=True)

df2.head()

# (7) Remove labels
try:
    label = ml_dummies['Churn'] # We remove labels before training
    del ml_dummies['Churn']
except:
    print("label already removed.")
    
# Training

feature_train, feature_test, label_train, label_test = train_test_split(ml_dummies, label, test_size=0.3)
    
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(feature_train, label_train)
pred = clf.predict(feature_test)
score = clf.score(feature_test, label_test)
print (round(score,3),"\n", "- - - - - ", "\n")

# Preprocessing original dataframe
def preprocess_df(dataframe):
    x = dataframe.copy()
    try:
        customer_id = x['customerID']
        del x['customerID'] # Not needed
    except:
        print("already removed customerID")
    ml_dummies = pd.get_dummies(x)
    ml_dummies.fillna(value=0, inplace=True)

    try:
        label = ml_dummies['Churn']
        del ml_dummies['Churn']
    except:
        print("label already removed.")
    return ml_dummies, customer_id, label

# Output preparation

original_df = preprocess_df(df)
output_df = original_df[0].copy()

output_df = original_df[0].copy()
output_df['prediction'] = clf.predict_proba(output_df)[:,1]
output_df['churn'] = original_df[2]
output_df['customerID'] = original_df[1]

activate = output_df[output_df['churn'] == 0]
output = activate[['customerID','churn','prediction']]
output = output.sort_values(by=['prediction'], ascending=False)

print(output.to_string())


