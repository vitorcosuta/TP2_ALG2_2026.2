"""Algoritmo aproximado de Christofides (fator 1,5 em instâncias métricas).

1. Calcula a AGM T (Prim, O(n²)).
2. Separa os vértices de grau ímpar em T (são sempre em número par).
3. Calcula o emparelhamento perfeito de custo mínimo M no subgrafo completo
   desses vértices (Blossom de Edmonds, O(k³)).
4. Une as arestas de T e M num multigrafo em que todo vértice tem grau par
   (euleriano) e encontra um circuito euleriano (Hierholzer, O(n)).
5. Remove os vértices repetidos do circuito, mantendo a primeira ocorrência
   (mesmo atalho do Twice-around-the-tree).

Custo final ≤ custo(T) + custo(M) ≤ ótimo + ótimo/2.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from tsp.agm import VetorPais, agm_prim, arestas_agm, graus_agm
from tsp.emparelhamento import emparelhamento_perfeito_minimo
from tsp.instancia import MatrizCustos
from tsp.medicao import MEDIDOR_NULO, Medidor
from tsp.tour import Tour, custo_tour
from tsp.twice_around_the_tree import remover_repetidos


def vertices_grau_impar(pai: VetorPais) -> list[int]:
    """Vértices com grau ímpar na árvore, em ordem crescente."""
    return [int(v) for v in np.flatnonzero(graus_agm(pai) % 2 == 1)]


def circuito_euleriano(
    n: int, arestas: Sequence[tuple[int, int]], inicio: int = 0
) -> list[int]:
    """Circuito euleriano de um multigrafo conexo com todos os graus pares.

    Algoritmo de Hierholzer iterativo, O(n + m). Arestas repetidas (presentes
    na AGM e no emparelhamento) são tratadas como arestas paralelas distintas.
    Devolve m + 1 vértices, começando e terminando em ``inicio``.
    """
    adjacencia: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (u, v) in enumerate(arestas):
        adjacencia[u].append((v, k))
        adjacencia[v].append((u, k))
    usada = [False] * len(arestas)
    proxima = [0] * n  # próxima posição a examinar na lista de cada vértice
    pilha = [inicio]
    circuito: list[int] = []
    while pilha:
        u = pilha[-1]
        lista = adjacencia[u]
        while proxima[u] < len(lista) and usada[lista[proxima[u]][1]]:
            proxima[u] += 1
        if proxima[u] == len(lista):
            circuito.append(pilha.pop())
        else:
            v, k = lista[proxima[u]]
            usada[k] = True
            pilha.append(v)
    circuito.reverse()
    return circuito


def christofides(
    custos: MatrizCustos, raiz: int = 0, medidor: Medidor = MEDIDOR_NULO
) -> tuple[int, Tour]:
    """Devolve ``(custo, tour)``; o tour começa em ``raiz``.

    Etapas medidas: ``agm``, ``impares``, ``emparelhamento``, ``euleriano``,
    ``atalhos``, ``custo``.
    """
    n = len(custos)
    with medidor.etapa("agm"):
        pai = agm_prim(custos, raiz)
    with medidor.etapa("impares"):
        impares = vertices_grau_impar(pai)
    with medidor.etapa("emparelhamento"):
        emparelhamento = emparelhamento_perfeito_minimo(custos, impares)
    with medidor.etapa("euleriano"):
        multigrafo = arestas_agm(pai) + emparelhamento
        circuito = circuito_euleriano(n, multigrafo, raiz)
    with medidor.etapa("atalhos"):
        tour = remover_repetidos(circuito)
    with medidor.etapa("custo"):
        custo = custo_tour(custos, tour)
    return custo, tour
