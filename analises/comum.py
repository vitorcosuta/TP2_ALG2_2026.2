"""Caminhos e entrada/saída compartilhados pelas análises.

Regra da camada de análise: **lê só ``resultados/`` e escreve só em
``relatorio/``**. Nenhuma análise importa ``tsp`` ou ``experimentos``, nem roda
algoritmo: tudo vem dos CSVs da coleta (ver ``ESTRUTURA.md``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "resultados"
RELATORIO = RAIZ / "relatorio"
FIGURAS = RELATORIO / "figuras"
TABELAS = RELATORIO / "tabelas"


def ler_resultado(nome: str, pasta: Path | None = None) -> pd.DataFrame:
    """Lê ``resultados/<nome>``, com uma mensagem que diz qual coleta falta."""
    caminho = (pasta or RESULTADOS) / nome
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe: rode antes a coleta que o gera "
            "(ver a tabela de scripts em ESTRUTURA.md)."
        )
    return pd.read_csv(caminho)


def salvar_tabela(tabela: pd.DataFrame, nome: str, pasta: Path | None = None) -> Path:
    """Grava ``relatorio/tabelas/<nome>`` (CSV) e devolve o caminho."""
    destino = (pasta or TABELAS) / nome
    destino.parent.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(destino, index=False)
    return destino


def salvar_figura(figura: Any, nome: str, pasta: Path | None = None) -> Path:
    """Grava ``relatorio/figuras/<nome>`` e fecha a figura do matplotlib."""
    import matplotlib.pyplot as plt

    destino = (pasta or FIGURAS) / nome
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)
    return destino
