import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

train = pd.read_csv('D:/Users/sefa.erkan/Desktop/Mobile_Prediciton/train.csv')

test = pd.read_csv('D:/Users/sefa.erkan/Desktop/Mobile_Prediciton/test.csv')

# Preview the dataset (train)
df_train = pd.DataFrame(train)
# Preview the dataset (test)
df_test = pd.DataFrame(test)

df_train.describe()

def check(df):
    l=[]
    columns=df.columns
    for col in columns:
        dtypes=df[col].dtypes
        nunique=df[col].nunique()
        sum_null=df[col].isnull().sum()
        l.append([col,dtypes,nunique,sum_null])
    df_check=pd.DataFrame(l)
    df_check.columns=['column','dtypes','nunique','sum_null']
    return df_check 

check(df_train)
check(df_test)

# Pre-Processing
df_test = df_test.drop('id', axis =1)
df_test

cat_cols = pd.DataFrame (df_train, columns= ['blue', 'dual_sim', 'four_g', 'n_cores', 'three_g', 'touch_screen', 'wifi'])
num_cols = pd.DataFrame (df_train, columns= ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'])

df = df_train

X = df.drop('price_range',axis=1)
y = df['price_range'].values.reshape(-1,1)

test_size = np.arange(start=0.2, stop=0.35, step=0.05)

score = []

for size in test_size:
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=size, random_state=0)
    RF_model = RandomForestClassifier()
    RF_model.fit(X_train, y_train)
    score.append(RF_model.score(X_test, y_test))

r = pd.DataFrame({'Test size': test_size, 'Score':score})   
r.sort_values(by = ['Score'],ascending=False, inplace=True)
r

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=0)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

RF = RandomForestClassifier()
RF.fit(X_train,y_train)

y_pred = RF.predict(X_test)

print('Model Accuracy Score: {0:0.4f}'.format(accuracy_score(y_test,y_pred)))

print(classification_report(y_test, y_pred))

def metrics_calculator(y_test, y_pred, model_name):
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, average='macro'),
                                recall_score(y_test, y_pred, average='macro'),
                                f1_score(y_test, y_pred, average='macro')],
                          index=['Accuracy','Precision','Recall','F1-score'],
                          columns = [model_name])
    return result

RF_result = metrics_calculator(y_test,y_pred, 'Random Forest')
RF_result