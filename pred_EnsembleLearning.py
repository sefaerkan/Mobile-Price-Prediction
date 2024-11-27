import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier

train = pd.read_csv('D:/Users/sefa.erkan/Desktop/Mobile_Prediciton/train.csv')
test = pd.read_csv('D:/Users/sefa.erkan/Desktop/Mobile_Prediciton/test.csv')

df_train = pd.DataFrame(train)
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

X = df.drop('price_range', axis=1)
y = df['price_range'].values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(X,y,stratify=y,test_size=0.2, random_state=0)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# "criterion" parametresi, modelin karar verme sürecinde hangi değerlendirme kriterini kullanacağını belirtir.
DT = DecisionTreeClassifier(max_depth=3)
improved_DT = DecisionTreeClassifier(criterion='entropy', max_depth=8)

RF = RandomForestClassifier()
improved_RF = RandomForestClassifier(criterion='gini',max_depth=None, max_features='sqrt',n_estimators=500)

SVM = SVC(probability=True)
improved_SVM = SVC(probability=True, C=10, gamma='scale',kernel='linear')

ensemble_model = VotingClassifier(estimators=[('DT',DT),('IDT',improved_DT),('RF',RF),('IRF',improved_RF),('SVM',SVM),('ISVM',improved_SVM)],voting='soft')

ensemble_model.fit(X_train, y_train)

prediction = ensemble_model.predict(X_test)

accuracy = metrics.accuracy_score(prediction,y_test)
print('Accuracy of the Ensemble Model on the test set:',accuracy)

cross_val_scores = cross_val_score(ensemble_model, X_train, y_train, cv=10, scoring='accuracy')
print('Cross-validated Score of the Ensemble Model:', cross_val_scores.mean())

def metrics_calculator(y_test, y_pred, model_name):
    result = pd.DataFrame(data=[accuracy_score(y_test, y_pred),
                                precision_score(y_test, y_pred, average='macro'),
                                recall_score(y_test, y_pred, average='macro'),
                                f1_score(y_test, y_pred, average='macro')],
                          index=['Accuracy','Precision','Recall','F1-score'],
                          columns = [model_name])
    return result

Voting_Classifier_r = metrics_calculator(y_test, prediction, 'Voting Classifier')
Voting_Classifier_r


df_test_predict = scaler.transform(df_test)
# Tahmin yapın
df_test['predicted_price_range'] = ensemble_model.predict(df_test_predict)
# Tahminleri kaydedin veya analiz edin
print(df_test.head())