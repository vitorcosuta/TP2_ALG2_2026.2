"""Instâncias usadas nos experimentos.

- ``SELECIONADAS``: conjunto **principal** (12; 75% com N ≤ 40). Usado no E1,
  no E2 e, quando o B&B prova o ótimo, no E3.
- ``EXTRAS_REGRESSAO``: instâncias da TSPLIB fora do conjunto principal que o
  B&B resolve em menos de 30 min. Estendem a faixa de N na regressão do B&B
  (E1) e, por terem ótimo provado, também entram no E3. Não entram no E2.
- Instâncias aleatórias (``instancia_aleatoria``): pontos uniformes no plano,
  geradas de forma determinística a partir de (N, semente). Usadas no E1.

O E3 usa só instâncias com ótimo provado pelo B&B (``otimo_provado`` em
``otimos_bnb.csv``): espera-se 9 principais + 4 extras = 13.
"""

from __future__ import annotations

import numpy as np

from tsp.instancia import Instancia, fechamento_metrico

SELECIONADAS = [
    "burma14",
    "ulysses16",
    "gr17",
    "gr21",
    "ulysses22",
    "gr24",
    "fri26",
    "bayg29",
    "bays29",  # faixa principal (N ≤ 40)
    "att48",
    "berlin52",
    "st70",  # faixa limite (45 ≤ N ≤ 75)
]

EXTRAS_REGRESSAO = ["swiss42", "dantzig42", "hk48", "eil51"]

# Valores da coluna ``conjunto`` nos CSVs do B&B.
PRINCIPAL, EXTRA, ALEATORIO = "principal", "extra", "aleatorio"


def conjunto_da_instancia(nome: str) -> str:
    """``principal`` se está em SELECIONADAS; ``extra`` para outras da TSPLIB."""
    return PRINCIPAL if nome in SELECIONADAS else EXTRA


def nome_aleatoria(n: int, semente: int) -> str:
    return f"rand_n{n:03d}_s{semente}"


def instancia_aleatoria(n: int, semente: int) -> Instancia:
    """Instância euclidiana determinística para (n, semente).

    N pontos uniformes em [0, 1000]², custo = distância euclidiana arredondada
    (como EUC_2D da TSPLIB), com fecho métrico, porque o arredondamento pode
    violar a desigualdade triangular por 1. A mesma (n, semente) gera a mesma
    instância em qualquer máquina.
    """
    rng = np.random.default_rng([n, semente])
    pontos = rng.uniform(0.0, 1000.0, size=(n, 2))
    dif = pontos[:, None, :] - pontos[None, :, :]
    originais = np.rint(np.hypot(dif[..., 0], dif[..., 1])).astype(np.int64)
    return Instancia(
        nome=nome_aleatoria(n, semente),
        dimensao=n,
        tipo_peso="EUC_2D",
        custos=fechamento_metrico(originais),
        custos_originais=originais,
        coordenadas=pontos,
        origem_coordenadas="ALEATORIA",
    )
