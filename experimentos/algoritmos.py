"""Os 5 algoritmos construtivos (heurísticas e aproximados) com interface única.

Usados nos scripts de experimento: como algoritmos medidos
(``aproximados_tsplib``) e como estratégias de warm-start do B&B
(``branch_and_bound``). Todos recebem a instância inteira porque a inserção
pelo fecho convexo precisa das coordenadas (ou do MDS, quando não há).
"""

from __future__ import annotations

from collections.abc import Callable

from tsp.christofides import christofides
from tsp.heuristicas import insercao_fecho_convexo, trivial, vizinho_mais_proximo
from tsp.instancia import Instancia
from tsp.medicao import Medidor
from tsp.tour import Tour
from tsp.twice_around_the_tree import twice_around_the_tree

Algoritmo = Callable[[Instancia, Medidor], tuple[int, Tour]]

ALGORITMOS: dict[str, Algoritmo] = {
    "trivial": lambda i, m: trivial(i.custos, medidor=m),
    "vizinho_mais_proximo": lambda i, m: vizinho_mais_proximo(i.custos, medidor=m),
    "insercao_fecho_convexo": lambda i, m: insercao_fecho_convexo(
        i.custos, i.coordenadas, medidor=m
    ),
    "twice_around_the_tree": lambda i, m: twice_around_the_tree(i.custos, medidor=m),
    "christofides": lambda i, m: christofides(i.custos, medidor=m),
}

TIPO = {
    "trivial": "heuristica",
    "vizinho_mais_proximo": "heuristica",
    "insercao_fecho_convexo": "heuristica",
    "twice_around_the_tree": "aproximativo",
    "christofides": "aproximativo",
}


def origem_pontos(instancia: Instancia, algoritmo: str) -> str:
    """De onde vieram os pontos usados pelo algoritmo ("" se não usa pontos)."""
    if algoritmo != "insercao_fecho_convexo":
        return ""
    return instancia.origem_coordenadas or "MDS"
