import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)


def gerar_dataset_tickets(qtd=1000):
    categorias = ['Incidente', 'Requisicao', 'Problema']
    prioridades = ['Baixa', 'Media', 'Alta', 'Critica']
    equipes = ['Infra', 'Sistemas', 'Seguranca', 'Suporte']
    turnos = ['Manha', 'Tarde', 'Noite']

    data = []

    for _ in range(qtd):
        categoria = np.random.choice(categorias)
        prioridade = np.random.choice(prioridades, p=[0.3, 0.4, 0.2, 0.1])
        equipe = np.random.choice(equipes)
        turno = np.random.choice(turnos)

        tempo_resolucao = np.random.normal(loc=6, scale=3)
        tempo_resolucao = max(0.5, tempo_resolucao)

        prioridade_peso = {'Baixa':1, 'Media':2, 'Alta':3, 'Critica':4}[prioridade]
        chance_sla = tempo_resolucao * prioridade_peso

        estourou_sla = 1 if chance_sla > 10 else 0

        data.append([
            categoria,
            prioridade,
            equipe,
            turno,
            round(tempo_resolucao,2),
            estourou_sla
        ])

    df = pd.DataFrame(data, columns=[
        'categoria',
        'prioridade',
        'equipe',
        'turno',
        'tempo_resolucao_horas',
        'estourou_sla'
    ])

    for col in ['categoria', 'prioridade']:
        df.loc[df.sample(frac=0.03).index, col] = np.nan

    return df


df = gerar_dataset_tickets(1000)
df.head()

print(df.info())
print(df.describe())

plt.hist(df['estourou_sla'])
plt.title('Distribuição de Estouro de SLA')
plt.show()

for col in ['categoria', 'prioridade']:
    df[col].fillna(df[col].mode()[0], inplace=True)

X = df.drop('estourou_sla', axis=1)
y = df['estourou_sla']

cat_features = ['categoria', 'prioridade', 'equipe', 'turno']
num_features = ['tempo_resolucao_horas']

preprocessador = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

modelos = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'DecisionTree': DecisionTreeClassifier(max_depth=5),
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=6)
}

for nome, modelo in modelos.items():
    pipeline = Pipeline([
        ('preprocess', preprocessador),
        ('model', modelo)
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    print(f"\nModelo: {nome}")
    print(classification_report(y_test, preds))

pipeline = Pipeline([
    ('preprocess', preprocessador),
    ('model', RandomForestClassifier(n_estimators=100, max_depth=6))
])

pipeline.fit(X_train, y_train)
preds = pipeline.predict(X_test)

cm = confusion_matrix(y_test, preds)
print('Matriz de Confusão:\n', cm)
