import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

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

# for col in cat_cols:
#     fig2 = px.histogram(cat_cols,x=col,color=col)
#     fig2.show()

# sns.pairplot(data=df_train[['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'pc', 'px_height',
#                             'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time', 'price_range']], 
#              hue='price_range')

# for i, col in enumerate(cat_cols):

#     fig, axes = plt.subplots(1,2,figsize=(10,5))
# # palette = 'hls'
#     # count of col (countplot)
#     sns.countplot(data=df_train, x=col, ax=axes[0] )
#     for container in axes[0].containers:
#         axes[0].bar_label(container)
#     # count of col (pie chart)
#     slices = df_train[col].value_counts().sort_index().values
#     activities = [var for var in df_train[col].value_counts().sort_index().index]
#     axes[1].pie(slices, labels=activities, shadow=True, autopct='%1.2f%%' ,startangle=90 )
    
    
 
#     plt.suptitle(f'Count of Unique Value in {col} (Fig {i+1})',fontsize=15)
#     plt.show()

# def dist_box(df_train):
#  # function plots a combined graph for univariate analysis of continous variable 
#  #to check spread, central tendency , dispersion and outliers  
#     Name=df_train.name.upper()
#     fig,(ax_box,ax_dis)  =plt.subplots(nrows=2,sharex=True,gridspec_kw = {"height_ratios": (.25, .75)},figsize=(8, 5))
#     mean=df_train.mean()
#     median=df_train.median()
#     mode=df_train.mode().tolist()[0]
#     sns.set_theme(style="white")
#     fig.suptitle("Distribution of "+ Name  , fontsize=18, fontweight='bold')
#     sns.boxplot(x=df_train,showmeans=True, orient='h',color="yellow",ax=ax_box)
#     ax_box.set(xlabel='')
#      # just trying to make visualisation better. This will set background to white
#     sns.despine(top=True,right=True,left=True) # to remove side line from graph
    
   
#     ax_dis.axvline(mean, color='r', linestyle='--',linewidth=2)
#     ax_dis.axvline(median, color='g', linestyle='-',linewidth=2)
#     ax_dis.axvline(mode, color='y', linestyle='-',linewidth=2)
#     plt.legend({'Mean':mean,'Median':median,'Mode':mode})
#     sns.set_style('darkgrid')
#     plt.grid()
#     sns.histplot(df_train,kde=True,color='blue',ax=ax_dis)
# #select all quantitative columns for checking the spread
# num_cols=['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 
#            'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time']

# for i in range(len(num_cols)):
#     dist_box(df_train[num_cols[i]])

num_cols = pd.DataFrame (df_train, columns= ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w', 'talk_time'])

# plt.figure(figsize = (13,6))
# sns.countplot(x = df_train['clock_speed'], hue ="price_range", data=df_train) 
# plt.show()

# # fc based on price_range
# plt.figure(figsize = (13,6))
# sns.countplot(x = df_train['fc'], hue ="price_range", data=df_train) 
# plt.show()

# # count of int_memory
# plt.figure(figsize = (13,6))
# sns.countplot(x = df_train['int_memory'], data=df_train) 
# plt.show()

# # m_dep based on price_range
# plt.figure(figsize = (13,6))
# sns.countplot(x = df_train['m_dep'], hue ="price_range", data=df_train) 
# plt.show()

# # pc based on price_range
# plt.figure(figsize = (13,6))
# sns.countplot(x = df_train['pc'], hue ="price_range", data=df_train) 
# plt.show()

# # sc_h based on price_range
# plt.figure(figsize = (13,6))
# sns.countplot(x = df_train['sc_h'], hue ="price_range", data=df_train) 
# plt.show()

# # sc_w based on price_range
# plt.figure(figsize = (13,6))
# sns.countplot(x = df_train['sc_w'], hue ="price_range", data=df_train) 
# plt.show()

# # talk_time based on price_range
# plt.figure(figsize = (13,6))
# sns.countplot(x = df_train['talk_time'], hue ="price_range", data=df_train) 
# plt.show()

# fig=plt.gcf()
# fig.set_size_inches(18, 12)
# plt.title('Correlation Between The Features', size=15)
# a = sns.heatmap(df_train.corr(), annot = True, cmap = 'GnBu', fmt='.2f', linewidths=0.2)
# plt.show()

## Model Building
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from statistics import stdev

# Decision Tree
df = df_train
X = df.drop('price_range', axis=1)
y = df['price_range'].values.reshape(-1, 1)

print ('X:', X.shape,'\ny:', y.shape)

# Hangi size'da daha iyi sonuç verdiğini görmek istedik.
test_size = np.arange(start=0.2, stop=0.35, step= 0.05)
score =[]
for size in test_size:
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=size, random_state=0)
    DT_Clf = DecisionTreeClassifier(max_depth=3)
    DT_Clf.fit(X_train1, y_train1)
    score.append(DT_Clf.score(X_test1, y_test1))

# Create a dataframe to display the results 
r= pd.DataFrame({'Test size': test_size , 'Score': score})
r.sort_values(by = ['Score'], ascending = False, inplace = True)

# stratify parametresi, veri setini bölerken belirli bir sınıf dağılımını korumak için kullanılır. 
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y,stratify=y, test_size=0.25, random_state=0)

print('X_train shape: ', X_train1.shape)
print('X_test shape: ', X_test1.shape)
print('y_train shape: ', y_train1.shape)
print('y_test shape: ', y_test1.shape)

scaler = StandardScaler()

X_train1 = scaler.fit_transform(X_train1)

X_test1 = scaler.fit_transform(X_test1)

DT = DecisionTreeClassifier(max_depth=3)

DT.fit(X_train1,y_train1)

y_pred = DT.predict(X_test1)

print("Model accuracy score: {0:0.4f}".format(accuracy_score(y_test1,y_pred)))

print(classification_report(y_test1, y_pred))

def metrics_calculator(y_test, y_pred, model_name):
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, average='macro'),
                                recall_score(y_test, y_pred, average='macro'),
                                f1_score(y_test, y_pred, average='macro')],
                          index=['Accuracy','Precision','Recall','F1-score'],
                          columns = [model_name])
    return result

DT_result = metrics_calculator(y_test1, y_pred, 'Decision Tree')
DT_result
