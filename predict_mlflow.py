# %%
# Importando as bibliotecas necessárias
import pandas as pd
import mlflow
import mlflow.sklearn


# %%
# Configurando o URI de rastreamento do MLflow para a URL local especificada
mlflow.set_tracking_uri('http://127.0.0.1:8080/')

# Carregando o modelo registrado no MLflow com o nome 'Churn-Estudos-ML' na fase de produção
model = mlflow.sklearn.load_model('models:/Churn-Estudos-ML/production')

# %%
# Obtendo as informações do modelo carregado, incluindo a assinatura (inputs e outputs)
model_info = mlflow.models.get_model_info('models:/Churn-Estudos-ML/production')

# Extraindo os nomes das features (entradas) do modelo a partir da assinatura
features = [input_.name for input_ in model_info.signature.inputs]
features


# %%

# MODIFICAR OS DADOS DE TESTE (ESTÃO IGUAIS AOS DE TREINO)


# Carregando um arquivo CSV contendo os dados que serão usados para a previsão
df = pd.read_csv(r'..\MLFlow\data\dados_pontos.csv', sep=';')

# %%
# Utilizando o modelo carregado para prever as probabilidades de churn para os dados carregados
pred = model.predict_proba(df[features])

# Extraindo a probabilidade de churn (segunda coluna das previsões)
proba_churn = pred[:,1]
proba_churn

# %%
# Criando um novo DataFrame para armazenar os resultados das previsões
df_predict = df[['dtRef', 'idCustomer']].copy()  # Copiando as colunas de data de referência e ID do cliente
df_predict['probaChurn'] = proba_churn.copy()  # Adicionando a coluna com a probabilidade de churn

# Ordenando os resultados pela probabilidade de churn em ordem decrescente e reiniciando o índice
df_predict = (df_predict.sort_values('probaChurn', ascending=False)
                        .reset_index(drop=True))

# Exibindo as primeiras linhas do DataFrame com os resultados das previsões
df_predict.head()

# %%
