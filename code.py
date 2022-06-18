import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
from warnings import filterwarnings
filterwarnings("ignore")
stress = pd.read_csv("/var/SaYoPillow.csv")
df = stress.copy()
df.dropna()
df.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', \
             'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']
df.head()

snoring_rate	respiration_rate	body_temperature	limb_movement	blood_oxygen	eye_movement	sleeping_hours	heart_rate	stress_level
0	93.80	25.680	91.840	16.600	89.840	99.60	1.840	74.20	3
1	91.64	25.104	91.552	15.880	89.552	98.88	1.552	72.76	3
2	60.00	20.000	96.000	10.000	95.000	85.00	7.000	60.00	1
3	85.76	23.536	90.768	13.920	88.768	96.92	0.768	68.84	3
4	48.12	17.248	97.872	6.496	96.248	72.48	8.248	53.12	0
df.describe().T

count	mean	std	min	25%	50%	75%	max
snoring_rate	630.0	71.6	19.372833	45.0	52.50	70.0	91.25	100.0
respiration_rate	630.0	21.8	3.966111	16.0	18.50	21.0	25.00	30.0
body_temperature	630.0	92.8	3.529690	85.0	90.50	93.0	95.50	99.0
limb_movement	630.0	11.7	4.299629	4.0	8.50	11.0	15.75	19.0
blood_oxygen	630.0	90.9	3.902483	82.0	88.50	91.0	94.25	97.0
eye_movement	630.0	88.5	11.893747	60.0	81.25	90.0	98.75	105.0
sleeping_hours	630.0	3.7	3.054572	0.0	0.50	3.5	6.50	9.0
heart_rate	630.0	64.5	9.915277	50.0	56.25	62.5	72.50	85.0
stress_level	630.0	2.0	1.415337	0.0	1.00	2.0	3.00	4.0
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 630 entries, 0 to 629
Data columns (total 9 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   snoring_rate      630 non-null    float64
 1   respiration_rate  630 non-null    float64
 2   body_temperature  630 non-null    float64
 3   limb_movement     630 non-null    float64
 4   blood_oxygen      630 non-null    float64
 5   eye_movement      630 non-null    float64
 6   sleeping_hours    630 non-null    float64
 7   heart_rate        630 non-null    float64
 8   stress_level      630 non-null    int64  
dtypes: float64(8), int64(1)
memory usage: 44.4 KB
df.isnull().values.any()
False
df["stress_level"].value_counts()
3    126
1    126
0    126
2    126
4    126
Name: stress_level, dtype: int64
sbn.countplot(data = df, x = "stress_level")
df.groupby('stress_level').mean()

snoring_rate	respiration_rate	body_temperature	limb_movement	blood_oxygen	eye_movement	sleeping_hours	heart_rate
stress_level								
0	47.5	17.0	97.5	6.0	96.0	70.0	8.0	52.5
1	55.0	19.0	95.0	9.0	93.5	82.5	6.0	57.5
2	70.0	21.0	93.0	11.0	91.0	90.0	3.5	62.5
3	87.5	24.0	91.0	14.5	89.0	97.5	1.0	70.0
4	98.0	28.0	87.5	18.0	85.0	102.5	0.0	80.0
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

 	snoring_rate	respiration_rate	body_temperature	limb_movement	blood_oxygen	eye_movement	sleeping_hours	heart_rate	stress_level
snoring_rate	1.000000	0.976268	-0.902475	0.981078	-0.903140	0.950600	-0.920554	0.976268	0.975322
respiration_rate	0.976268	1.000000	-0.889237	0.991738	-0.889210	0.935572	-0.891855	1.000000	0.963516
body_temperature	-0.902475	-0.889237	1.000000	-0.896412	0.998108	-0.857299	0.954860	-0.889237	-0.962354
limb_movement	0.981078	0.991738	-0.896412	1.000000	-0.898527	0.964703	-0.901102	0.991738	0.971071
blood_oxygen	-0.903140	-0.889210	0.998108	-0.898527	1.000000	-0.862136	0.950189	-0.889210	-0.961092
eye_movement	0.950600	0.935572	-0.857299	0.964703	-0.862136	1.000000	-0.893952	0.935572	0.951988
sleeping_hours	-0.920554	-0.891855	0.954860	-0.901102	0.950189	-0.893952	1.000000	-0.891855	-0.973036
heart_rate	0.976268	1.000000	-0.889237	0.991738	-0.889210	0.935572	-0.891855	1.000000	0.963516
stress_level	0.975322	0.963516	-0.962354	0.971071	-0.961092	0.951988	-0.973036	0.963516	1.000000
def comparisonplot(data, column1, column2, target):
    figure = px.scatter(data_frame = data, x=column1,
                    y=column2, color= target)
    figure.show()
cols = []
for col1 in df.columns:
    cols.append(col1)
    for col2 in df.columns:
        if col2 not in cols:
            comparisonplot(df,col1, col2, 'stress_level')
y = df["stress_level"]
X = df.drop(["stress_level"], axis= 1)
y[0:5]
0    3
1    3
2    1
3    3
4    0
Name: stress_level, dtype: int64
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 42)
knn = KNeighborsClassifier()
knn_model = knn.fit(X_train, y_train)
knn_model
KNeighborsClassifier()
y_pred = knn_model.predict(X_test)
accuracy_score(y_test, y_pred)
1.0
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        34
           1       1.00      1.00      1.00        31
           2       1.00      1.00      1.00        33
           3       1.00      1.00      1.00        31
           4       1.00      1.00      1.00        29

    accuracy                           1.00       158
   macro avg       1.00      1.00      1.00       158
weighted avg       1.00      1.00      1.00       158
mat = confusion_matrix(y_test, y_pred)
p = sbn.heatmap(pd.DataFrame(mat), annot=True, cmap="YlGnBu" ,fmt='g')
df = stress.copy()
df = df.dropna()
df.columns=['snoring_rate', 'respiration_rate', 'body_temperature', 'limb_movement', 'blood_oxygen', \
             'eye_movement', 'sleeping_hours', 'heart_rate', 'stress_level']
y = df["stress_level"]
X = df.drop(["stress_level"], axis= 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.25, random_state= 42)
rf_model = RandomForestClassifier().fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_test, y_pred)
newData = [[93.80,	25.680,	91.840,	16.600,	89.840,	99.60,	1.840,	74.20]]
rf_model.predict(newData)
Importance = pd.DataFrame({"Importance": rf_model.feature_importances_*100},
                          index = X_train.columns)

Importance.sort_values(by = "Importance",
                       axis = 0,
                       ascending = True).plot(kind = "barh", color = "r")

plt.xlabel("Variables Importance Ratio")
print(classification_report(y_test, y_pred))
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        34
           1       1.00      0.97      0.98        31
           2       0.97      1.00      0.99        33
           3       1.00      1.00      1.00        31
           4       1.00      1.00      1.00        29

    accuracy                           0.99       158
   macro avg       0.99      0.99      0.99       158
weighted avg       0.99      0.99      0.99       158
mat = confusion_matrix(y_test, y_pred)
p = sbn.heatmap(pd.DataFrame(mat), annot=True, cmap="YlGnBu" ,fmt='g')
