#!/usr/bin/env python3
"""Classificador de sentimento de avaliações em português.

Treinar:   python treinar.py
Prever:    python treinar.py --prever "Adorei o produto, chegou rápido"
Interativo: python treinar.py --interativo
Outro CSV: python treinar.py --dados data/meu_arquivo.csv
(o CSV precisa das colunas: texto, sentimento)
"""
import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # salva a figura sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DADOS_PADRAO = Path("data/avaliacoes.csv")
ARQUIVO_MODELO = Path("modelo.joblib")
ARQUIVO_IMAGEM = Path("matriz_confusao.png")


def treinar(caminho_csv: Path) -> None:
    df = pd.read_csv(caminho_csv).dropna(subset=["texto", "sentimento"])
    print(f"{len(df)} avaliações carregadas.")
    print(df["sentimento"].value_counts().to_string(), end="\n\n")

    x_treino, x_teste, y_treino, y_teste = train_test_split(
        df["texto"],
        df["sentimento"],
        test_size=0.25,
        random_state=42,
        stratify=df["sentimento"],
    )

    modelo = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode", lowercase=True),
        LogisticRegression(max_iter=1000),
    )
    modelo.fit(x_treino, y_treino)

    previsoes = modelo.predict(x_teste)
    print(classification_report(y_teste, previsoes, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y_teste, previsoes, cmap="Blues")
    plt.title("Matriz de confusão (conjunto de teste)")
    plt.savefig(ARQUIVO_IMAGEM, dpi=150, bbox_inches="tight")

    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f"Modelo salvo em {ARQUIVO_MODELO} e gráfico em {ARQUIVO_IMAGEM}.")


def prever(texto: str) -> None:
    if not ARQUIVO_MODELO.exists():
        sys.exit("Treine primeiro com: python treinar.py")
    modelo = joblib.load(ARQUIVO_MODELO)
    classe = modelo.predict([texto])[0]
    probabilidades = modelo.predict_proba([texto])[0]
    confianca = max(probabilidades)
    print(f"Sentimento: {classe} (confiança {confianca:.0%})")


def interativo() -> None:
    """Fica pedindo frases e mostrando o sentimento de cada uma."""
    if not ARQUIVO_MODELO.exists():
        sys.exit("Treine primeiro com: python treinar.py")
    modelo = joblib.load(ARQUIVO_MODELO)
    print("Modo interativo. Digite uma avaliação e aperte Enter (ou 'sair').")
    while True:
        try:
            texto = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if texto.lower() in {"sair", "exit", "q"}:
            break
        if not texto:
            continue
        classe = modelo.predict([texto])[0]
        confianca = max(modelo.predict_proba([texto])[0])
        print(f"  Sentimento: {classe} (confiança {confianca:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dados", type=Path, default=DADOS_PADRAO)
    parser.add_argument("--prever", metavar="TEXTO", help="classifica um texto novo")
    parser.add_argument(
        "--interativo", action="store_true", help="classifica várias frases em sequência"
    )
    args = parser.parse_args()

    if args.interativo:
        interativo()
    elif args.prever:
        prever(args.prever)
    else:
        treinar(args.dados)


if __name__ == "__main__":
    main()
