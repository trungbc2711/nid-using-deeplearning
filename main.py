import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from joblib import dump, load

df_test = pd.read_csv('dataset/test.csv')

categorical_features = ['fin_flag_number','syn_flag_number','rst_flag_number','psh_flag_number','ack_flag_number',
                        'ece_flag_number','cwr_flag_number','HTTP','HTTPS','DNS','Telnet','SMTP','SSH','IRC','TCP',
                        'UDP','DHCP','ARP','ICMP','IPv','LLC']
numberical_features = ['flow_duration','Header_Length','Protocol type','Duration','Rate','Srate','Drate',
                       'ack_count','syn_count','fin_count','urg_count','rst_count','Tot sum','Min','Max',
                       'AVG','Std','Tot size','IAT','Number','Magnitue','Radius','Covariance','Variance']
imputer = SimpleImputer(strategy='most_frequent')
df_test[categorical_features] = imputer.transform(df_test[categorical_features])
imputer = SimpleImputer(strategy='median')
df_test[numberical_features] = imputer.transform(df_test[numberical_features])

X_test = df_test.drop(['Weight'], axis=1)

scaler = MinMaxScaler()
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

rfe_selected_features = ['ID','flow_duration','Header_Length','Protocol type','Duration',
                         'fin_flag_number','syn_flag_number','rst_flag_number','psh_flag_number',
                         'ack_flag_number','ack_count','syn_count','fin_count','urg_count',
                         'rst_count','HTTP','HTTPS','TCP','UDP','ICMP','IPv','Min','Max',
                         'AVG','Std','Tot size','Number','Magnitue','Radius','Covariance','Variance']
X_test = X_test[rfe_selected_features]
X_test = X_test.to_numpy()

encoder = load('config/label_encoder.joblib')
y_test = encoder.transform(y_test)

loaded_model = keras.models.load_model("models/final_model_LSTM.keras")

f1 = f1_score(y_test, y_pred, average="micro")
print(f"F1-score: {f1:.4f}")

y_pred = model.predict(X_test).argmax(axis=1)
print(classification_report(y_test, y_pred))
