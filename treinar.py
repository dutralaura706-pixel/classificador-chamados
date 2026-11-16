import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
from sklearn.pipeline import make_pipeline

#ler e limpar dados
df=pd.read_csv("chamados.csv")
df["categoria"]=df["categoria"].str.strip().str.lower()

print("Total chamados.",len(df))
print(df["categoria"].value_counts())

#separar treino (80%) e teste(20%)
X_treino, X_teste, y_treino, y_teste = train_test_split(
    df["texto"],
    df["categoria"],
    test_size=0.2,
    random_state=42,
    stratify=df["categoria"],
)

# 3. Montar o modelo
modelo = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000),
)

# 4. Treinar
modelo.fit(X_treino, y_treino)

# 5. Avaliar no conjunto de teste
preditos = modelo.predict(X_teste)
print()
print("Acurácia:", accuracy_score(y_teste, preditos))
print()
print(classification_report(y_teste, preditos))

print()
print("Chamados que o modelo errou")
for texto,certo,previsto,in zip(X_teste,y_teste,preditos):
    if certo!=previsto:
        print("-",texto)
        print("categoria certa:",certo,"| modelo disse.",previsto)

print()
print("Testando frases novas.") 
frases_novas=[
      "meu notebook não liga mais",
    "não consigo entrar no sistema com a minha senha",
    "a internet do prédio está caindo toda hora",
    "o aplicativo fecha sozinho quando eu abro",
]
previsoes = modelo.predict(frases_novas)
for frase, categoria in zip(frases_novas, previsoes):
    print(categoria, "<-", frase)

