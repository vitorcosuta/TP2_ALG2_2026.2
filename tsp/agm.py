"""Árvore geradora mínima (AGM) pelo algoritmo de Prim.

As instâncias do TSP são grafos completos, então usamos a versão de Prim com
vetor de distâncias (sem heap), em O(n²): cada um dos n passos escolhe o vértice
mais próximo da árvore e atualiza as distâncias com operações vetorizadas sobre
uma linha da matriz de custos. Em grafos densos isso é assintoticamente melhor
que Kruskal ou Prim com heap, ambos O(n² log n).

A árvore é representada pelo vetor ``pai``: ``pai[v]`` é o vizinho de ``v`` no
caminho até a raiz, e ``pai[raiz] == -1``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tsp.instancia import MatrizCustos

VetorPais = npt.NDArray[np.int64]

_INFINITO = np.iinfo(np.int64).max


def agm_prim(custos: MatrizCustos, raiz: int = 0) -> VetorPais:
    """Calcula a AGM do grafo completo dado por ``custos``, em O(n²).

    Devolve o vetor de pais com ``pai[raiz] == -1``. A matriz não é alterada.
    """
    n = len(custos)
    if not 0 <= raiz < n:
        raise ValueError(f"raiz {raiz} fora do intervalo [0, {n})")

    na_arvore = np.zeros(n, dtype=bool)
    pai = np.full(n, raiz, dtype=np.int64)
    distancia = custos[raiz].copy()

    na_arvore[raiz] = True
    pai[raiz] = -1
    distancia[raiz] = _INFINITO

    for _ in range(n - 1):
        v = int(np.argmin(distancia))
        na_arvore[v] = True
        distancia[v] = _INFINITO
        melhora = ~na_arvore & (custos[v] < distancia)
        distancia[melhora] = custos[v, melhora]
        pai[melhora] = v
    return pai


def arestas_agm(pai: VetorPais) -> list[tuple[int, int]]:
    """Lista as n − 1 arestas ``(pai[v], v)`` da árvore."""
    return [(int(p), v) for v, p in enumerate(pai) if p >= 0]


def custo_agm(custos: MatrizCustos, pai: VetorPais) -> int:
    """Soma dos custos das arestas da árvore."""
    filhos = np.flatnonzero(pai >= 0)
    return int(custos[pai[filhos], filhos].sum())


def graus_agm(pai: VetorPais) -> npt.NDArray[np.int64]:
    """Grau de cada vértice na árvore (cada aresta conta para os dois extremos)."""
    n = len(pai)
    filhos = np.flatnonzero(pai >= 0)
    graus = np.bincount(pai[filhos], minlength=n)
    graus[filhos] += 1
    return graus.astype(np.int64)


def filhos_agm(pai: VetorPais) -> list[list[int]]:
    """Lista de filhos de cada vértice, útil para percorrer a árvore em pré-ordem."""
    filhos: list[list[int]] = [[] for _ in range(len(pai))]
    for v, p in enumerate(pai):
        if p >= 0:
            filhos[int(p)].append(v)
    return filhos
