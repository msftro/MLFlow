# %%

# Importa as bibliotecas necessárias
from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import pipeline

from feature_engine import imputation

import mlflow
import mlflow.sklearn

import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import scikitplot as skplot

# Carrega os dados do arquivo CSV
df = pd.read_csv('../data/dados_pontos.csv', sep=';')
df.head()  # Exibe as primeiras linhas do DataFrame para visualização


# %%

# Define as features (colunas) e a target (variável alvo) do modelo
features = df.columns[3:-1]  # Seleciona as colunas de interesse para features
target = 'flActive'  # Define a coluna alvo

# Cria os dataframes X e y
X = df[features]  # Features
y = df[target]  # Target

# Divide os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y,
                                                                    train_size=0.8,
                                                                    random_state=42,
                                                                    stratify=y)

# Imprime a proporção de valores positivos na variável alvo nos conjuntos de treino e teste
print('Acurácia Train: ', y_train.mean())
print('Acurária Teste: ', y_test.mean())


# %%

# Verifica a quantidade de valores nulos em cada feature no conjunto de treino
print(X_train.isna().sum())


# %%

# Configura o URI de rastreamento do MLflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Define o experimento do MLflow
mlflow.set_experiment(experiment_id=570199247161289105)

# Ativa a autologação do MLflow para capturar automaticamente métricas e parâmetros do modelo
mlflow.autolog()


# %%

# Cria os imputadores de valores arbitrários para preencher valores nulos
imput_recorrencia = imputation.ArbitraryNumberImputer(variables=['avgRecorrencia'], arbitrary_number=X_train['avgRecorrencia'].max())
imput_0 = imputation.ArbitraryNumberImputer(variables=list(set(features) - set(imput_recorrencia.variables)))

# Define o classificador
clf_rf = ensemble.RandomForestClassifier(random_state=42)
clf_gb = ensemble.GradientBoostingClassifier(random_state=42)
clf_ab = ensemble.AdaBoostClassifier(random_state=42)
clf_xgb = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Define os parâmetros para a busca em grid

params_xgb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, 15]
}

params_ab = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
}

params_gb = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [3, 5, 10, 15]
}

params_rf = {
    'max_depth': [3, 5, 10, 15, 20],
    'n_estimators': [50, 100, 200, 500, 1000],
    'min_samples_leaf': [10, 15, 20, 50, 100]
}

# Configura o GridSearchCV para otimizar os hiperparâmetros dos modelos

grid_xgb = model_selection.GridSearchCV(clf_xgb,
                                       param_grid=params_xgb,
                                       scoring='roc_auc',
                                       n_jobs=-1,
                                       verbose=3
                                       )

grid_rf = model_selection.GridSearchCV(clf_rf,
                                       param_grid=params_rf,
                                       scoring='roc_auc',
                                       n_jobs=-1,
                                       verbose=3
                                       )

grid_gb = model_selection.GridSearchCV(clf_gb,
                                       param_grid=params_gb,
                                       scoring='roc_auc',
                                       n_jobs=-1,
                                       verbose=3
                                       )

grid_ab = model_selection.GridSearchCV(clf_ab,
                                       param_grid=params_ab,
                                       scoring='roc_auc',
                                       n_jobs=-1,
                                       verbose=3
                                       )

# %%

# Ajusta o pipeline aos dados de treino
def train_and_evaluate(model, model_name):

    with mlflow.start_run():

        # Cria um pipeline para imputação de valores nulos e ajuste do modelo
        model = pipeline.Pipeline([
        ('imput 0', imput_0),
        ('imput recorrencia', imput_recorrencia),
        ('model', model)]
        )

        model.fit(X_train, y_train)

        # Faz previsões no conjunto de teste
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)

        # Calcula e imprime a AUC (Área sob a Curva ROC) e ACC para o conjunto de teste
        test_auc = metrics.roc_auc_score(y_test, y_test_proba[:,1])
        test_acc = metrics.accuracy_score(y_test, y_test_pred)

        # Registra a métrica AUC no MLflow
        mlflow.log_metric(f'{model_name}_test_roc_auc', test_auc)
        mlflow.log_metric(f'{model_name}_test_accuracy_score', test_acc)

        # Plota a estatística KS e registra no MLflow
        skplot.metrics.plot_ks_statistic(y_test, y_test_proba)
        plt.savefig(f"{model_name}_ks_statistic.png")
        mlflow.log_artifact(f"{model_name}_ks_statistic.png")
        plt.show()

        # Plota a curva de lift e registra no MLflow
        skplot.metrics.plot_lift_curve(y_test, y_test_proba)
        plt.savefig(f"{model_name}_lift_curve.png")
        mlflow.log_artifact(f"{model_name}_lift_curve.png")
        plt.show()

        # Plota o ganho cumulativo e registra no MLflow
        skplot.metrics.plot_cumulative_gain(y_test, y_test_proba)
        plt.savefig(f"{model_name}_cumulative_gain.png")
        mlflow.log_artifact(f"{model_name}_cumulative_gain.png")
        plt.show()


# %%

train_and_evaluate(grid_rf, 'RandomForest')
train_and_evaluate(grid_gb, 'GradientBoosting')
train_and_evaluate(grid_ab, 'AdaBoost')
train_and_evaluate(grid_xgb, 'XGBoost')

# %%
