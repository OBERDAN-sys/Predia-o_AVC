import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import pickle


df_avc = pd.read_csv('C:/Users/oberd/OneDrive/Documentos/..BANCO EDA/cardio_avc_idade.csv', encoding='ISO-8859-1',
                                 engine='python')
# PREPROCESSAMENTO
# # substituir os  0 por valores ausentes NaN
df_avc = df_avc.copy(deep=True)
df_avc[['hipertensao', 'doença_do_coracao', 'ja_se_casou', 'tipo_trabalho', 'tipo_residencia', 'nivel_medio_glicose',
        'imc', 'status_tabagismo']] = df_avc[['hipertensao', 'doença_do_coracao', 'ja_se_casou', 'tipo_trabalho',
                                              'tipo_residencia', 'nivel_medio_glicose', 'imc',
                                              'status_tabagismo']].replace(0, np.NaN)

# Substitui valores ausentes por mean,medain
df_avc['imc'] = df_avc['imc'].fillna(df_avc['imc'].mean())

# Limpeza e adequação do dados
df_avc.fillna(value=0, inplace=True)
df_avc["idade"] = df_avc["idade"].astype(int)
df_avc["imc"] = df_avc["imc"].astype(int)
df_avc["nivel_medio_glicose"] = df_avc["nivel_medio_glicose"].astype(int)
df_avc["hipertensao"] = df_avc["hipertensao"].astype(int)
df_avc["doença_do_coracao"] = df_avc["doença_do_coracao"].astype(int)
df_avc = df_avc.drop('id', axis=1)

# Contagem de classes de recurso categorico
# print(df_avc['genero'].value_counts())
# print(df_avc['ja_se_casou'].value_counts())
# print(df_avc['tipo_residencia'].value_counts())
# print(df_avc['tipo_trabalho'].value_counts())
# print (df_avc['status_tabagismo'].value_counts())

# Engenharia de dados (nivel_medio_glicose
df_avc['nivel_medio_glicose'] = df_avc['nivel_medio_glicose'].apply(lambda x: 1 if x <= 100 else 2 if x <= 125 else 3)
df_avc.rename(columns={"nivel_medio_glicose": "glicose"}, inplace=True)


# Transformação de dados: Conversão
df_avc['genero'] = df_avc['genero'].map({'Male': 0, 'Female': 1, 'Other': 2})
df_avc['ja_se_casou'] = df_avc['ja_se_casou'].map({'No': 0, 'Yes': 1})
df_avc['tipo_residencia'] = df_avc['tipo_residencia'].map({'Rural': 0, 'Urban': 1})
df_avc['tipo_trabalho'] = df_avc['tipo_trabalho'].map({'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3,
                                                       'Never_worked': 4})
df_avc['status_tabagismo'] = df_avc['status_tabagismo'].map({'never smoked': 0, 'Unknown': 1,
                                                             'formerly smoked': 2, 'smokes': 3})

print(df_avc.head())
print(df_avc.describe())
# CONSTRUÇÃO DOS MODELO E SELEÇÃO DO MELHOR
# Padronização dos dados
std_list = ["idade", "imc"]
def standartization(x):
    x_std = x.copy(deep=True)
    for column in std_list:
        x_std[column] = (x_std[column] - x_std[column].mean()) / x_std[column].std()
    return x_std

df_avc = standartization(df_avc)

# Separar  variaveis independentes/entradas da dependente/saida
X = df_avc.iloc[:, :-1]
y = df_avc.iloc[:, -1]
# Divisão dos dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
# Construção do modelo (Logistic Regression)
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
# Avaliando o modelo
log_reg = log_reg.score(X_test, y_test)
# Construção do modelo (Random forest classifier)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
# Avaliando o modelo
clf = clf.score(X_test, y_test)
# Construção do modelo (XGBClassifier)
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
gbc.fit(X_train, y_train)
# Avaliando o modelo
gbc = gbc.score(X_test, y_test)
# Construção do modelo (LGBM Classifier)
lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
# Avaliando o modelo
lgbm = lgbm.score(X_test, y_test)
model_compare = pd.DataFrame({"Logistic Regression": log_reg, "Random Forest Classifier": clf,
                               "GradientBoostingClassifier": gbc, "LGBM Classifier": lgbm}, index=["accuracy"])
print(model_compare)

log_reg1 = log_reg
clf1 = clf
gbc1 = gbc
lgbm1 = lgbm

# Verificando quem é maior resulyado
maior = log_reg1
if clf1 > log_reg1 and clf1 > gbc1 or clf1 > lgbm1:
    maior = clf1
if gbc1 > log_reg1 and gbc1 > clf1 or gbc1 > lgbm1:
    maior = gbc1
if lgbm1 > log_reg1 and lgbm1 > clf1 or lgbm1 > gbc1:
    maior = lgbm1

print('O  melhor resultado do classificador é {}  '.format(maior))

if maior == log_reg1:
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    pickle.dump(log_reg, open('pre_log_reg.pkl', 'wb'))

if maior == clf1:
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pickle.dump(clf, open('pred_clf.pkl', 'wb'))

if maior == gbc1:
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    gbc.fit(X_train, y_train)
    pickle.dump(gbc, open('pred_gbc.pkl', 'wb'))

else:
    lgbm = LGBMClassifier()
    lgbm.fit(X_train, y_train)
    pickle_out = open("pre_lgbm.pkl", mode="wb")
    pickle.dump(lgbm, pickle_out)
    pickle_out.close()


