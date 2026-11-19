"""Algoritmo aproximado Twice-around-the-tree (fator 2 em instâncias métricas).

1. Calcula a AGM da instância (Prim, O(n²)).
2. Faz o caminhamento em pré-ordem na AGM anotando cada vértice sempre que ele
   é visitado, inclusive na volta de um filho. É o passeio fechado que percorre
   cada aresta da árvore duas vezes ("dar duas voltas na árvore"), com 2n − 1
   vértices e custo 2 · custo(AGM).
3. Remove os vértices repetidos, mantendo só a primeira ocorrência. Cada
   remoção troca o trecho ``u → (repetidos) → w`` pela aresta direta ``(u, w)``;
   pela desigualdade triangular, esse atalho não aumenta o custo. Ex.: em
   a-b-c-b-d, o segundo b sai e a aresta (c, d) substitui c → b → d.

Custo final ≤ 2 · custo(AGM) ≤ 2 · ótimo.
"""

from __future__ import annotations

from tsp.agm import VetorPais, agm_prim, filhos_agm
from tsp.instancia import MatrizCustos
from tsp.medicao import MEDIDOR_NULO, Medidor
from tsp.tour import Tour, custo_tour


def caminhamento_pre_ordem(pai: VetorPais, raiz: int = 0) -> list[int]:
    """Passeio fechado pela árvore, anotando vértices na ida e na volta.

    Devolve 2n − 1 vértices, começando e terminando em ``raiz``. É iterativo
    para não depender do limite de recursão do Python.
    """
    filhos = filhos_agm(pai)
    passeio = [raiz]
    pilha = [(raiz, iter(filhos[raiz]))]
    while pilha:
        _, proximos = pilha[-1]
        filho = next(proximos, None)
        if filho is None:
            pilha.pop()
            if pilha:
                passeio.append(pilha[-1][0])  # volta ao pai
        else:
            passeio.append(filho)
            pilha.append((filho, iter(filhos[filho])))
    return passeio


def remover_repetidos(passeio: list[int]) -> Tour:
    """Mantém a primeira ocorrência de cada vértice (atalhos pela métrica)."""
    vistos: set[int] = set()
    tour: Tour = []
    for v in passeio:
        if v not in vistos:
            vistos.add(v)
            tour.append(v)
    return tour


def twice_around_the_tree(
    custos: MatrizCustos, raiz: int = 0, medidor: Medidor = MEDIDOR_NULO
) -> tuple[int, Tour]:
    """Devolve ``(custo, tour)``; o tour começa em ``raiz``.

    Etapas medidas: ``agm``, ``pre_ordem``, ``atalhos``, ``custo``.
    """
    with medidor.etapa("agm"):
        pai = agm_prim(custos, raiz)
    with medidor.etapa("pre_ordem"):
        passeio = caminhamento_pre_ordem(pai, raiz)
    with medidor.etapa("atalhos"):
        tour = remover_repetidos(passeio)
    with medidor.etapa("custo"):
        custo = custo_tour(custos, tour)
    return custo, tour
