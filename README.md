# Classificador de sentimento de avaliações (português)

Modelo de Machine Learning clássico que classifica avaliações de produtos como **positivas** ou **negativas**, usando TF-IDF + Regressão Logística (scikit-learn). Gera relatório de métricas, matriz de confusão e permite testar frases novas pelo terminal.

## Como rodar

```bash
git clone <URL-DO-SEU-REPOSITORIO>
cd sentimento-avaliacoes
python -m venv .venv && source .venv/bin/activate   # no Windows: .venv\Scripts\activate
pip install -r requirements.txt

python treinar.py                                    # treina e avalia
python treinar.py --prever "Adorei o produto, chegou rápido"
```

## Sobre os dados

O arquivo `data/avaliacoes.csv` é apenas uma **amostra pequena (30 linhas) para o projeto rodar de imediato**. Para resultados de verdade, use um dataset maior em português, por exemplo avaliações de e-commerce do Kaggle, ajustando as colunas para `texto` e `sentimento`:

```bash
python treinar.py --dados data/meu_dataset.csv
```

## Resultados

Com a amostra de 30 linhas, o conjunto de teste tem só 8 avaliações e a acurácia fica perto de 50%, o que é esperado com tão pouco dado. Treine com um dataset real (milhares de linhas) antes de publicar os números.

Cole aqui a saída do `classification_report` e a imagem `matriz_confusao.png` do seu treino final, para mostrar que você sabe interpretar acurácia, precisão e recall.

## Como funciona

1. Lê o CSV e divide em treino (75%) e teste (25%) de forma estratificada.
2. `TfidfVectorizer` com unigramas e bigramas transforma o texto em números.
3. `LogisticRegression` aprende a separar as classes.
4. Avalia no conjunto de teste, salva o modelo com `joblib` e gera a matriz de confusão.

## Tecnologias

Python, pandas, scikit-learn, matplotlib, joblib.

## Próximos passos

- Incluir a classe "neutro".
- Comparar com Naive Bayes e SVM.
- Testar embeddings de um modelo em português (BERTimbau).
