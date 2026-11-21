"""Heurísticas construtivas usadas como warm-start do Branch-and-Bound.

Todas devolvem ``(custo, tour)`` com o tour começando no vértice 0, no mesmo
formato do Twice-around-the-tree e do Christofides.

- **Trivial** (0 → 1 → … → n−1 → 0): não olha os custos. Serve de referência
  "a frio": o limite superior inicial é ruim.
- **Vizinho mais próximo**: a partir do 0, vai sempre para o vértice não
  visitado mais barato. O(n²).
- **Inserção a partir do fecho convexo**: começa com o ciclo formado pelos
  vértices do fecho convexo (na ordem do fecho) e insere os demais um a um.
  O(n³) no total, mas cada passo é vetorizado com NumPy.

Inserção pelo fecho convexo (Stewart, 1977; Golden et al., 1980). A cada
passo, para cada vértice k fora do ciclo:

1. acha a aresta (i, j) do ciclo que minimiza o acréscimo
   ``c(i,k) + c(k,j) − c(i,j)``;
2. entre todos os k, escolhe o que minimiza a razão
   ``(c(i,k) + c(k,j)) / c(i,j)`` na sua melhor aresta, e o insere entre i e j.

**Instâncias sem coordenadas.** As coordenadas só são usadas para o ciclo
inicial (o fecho); a inserção usa a matriz de custos real. Quando a instância
não tem pontos, eles são reconstruídos da matriz por MDS clássico
(``tsp.features.mds_classico``), e o tempo do MDS entra na medição (etapa
``mds``), porque faz parte do custo de obter o warm-start. O tour é sempre
válido e o seu custo é sempre o custo real; um MDS ruim só piora o ciclo
inicial.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull, QhullError

from tsp.features import mds_classico
from tsp.instancia import Coordenadas, MatrizCustos
from tsp.medicao import MEDIDOR_NULO, Medidor
from tsp.tour import Tour, custo_tour


def _comecar_em_zero(tour: Tour) -> Tour:
    i = tour.index(0)
    return tour[i:] + tour[:i]


# Trivial
def trivial(custos: MatrizCustos, medidor: Medidor = MEDIDOR_NULO) -> tuple[int, Tour]:
    """Tour 0 → 1 → … → n−1. Etapas medidas: ``construcao``, ``custo``."""
    with medidor.etapa("construcao"):
        tour = list(range(len(custos)))
    with medidor.etapa("custo"):
        custo = custo_tour(custos, tour)
    return custo, tour


# Vizinho mais próximo
def tour_vizinho_mais_proximo(custos: MatrizCustos, inicio: int = 0) -> Tour:
    """Vai sempre ao vértice não visitado mais barato (empate: menor índice)."""
    n = len(custos)
    if n == 0:
        return []
    linhas = custos.astype(np.float64)
    visitado = np.zeros(n, dtype=bool)
    tour = [inicio]
    visitado[inicio] = True
    atual = inicio
    for _ in range(n - 1):
        candidatos = np.where(visitado, np.inf, linhas[atual])
        atual = int(np.argmin(candidatos))
        visitado[atual] = True
        tour.append(atual)
    return tour


def vizinho_mais_proximo(
    custos: MatrizCustos, medidor: Medidor = MEDIDOR_NULO
) -> tuple[int, Tour]:
    """Etapas medidas: ``construcao``, ``custo``."""
    with medidor.etapa("construcao"):
        tour = tour_vizinho_mais_proximo(custos)
    with medidor.etapa("custo"):
        custo = custo_tour(custos, tour)
    return custo, tour


# Inserção pelo fecho
def ciclo_fecho_convexo(pontos: Coordenadas, custos: MatrizCustos) -> list[int]:
    """Vértices do fecho convexo na ordem do fecho (anti-horária).

    Se os pontos forem degenerados (todos colineares ou repetidos), o
    ``ConvexHull`` falha; nesse caso o ciclo inicial é o par de vértices mais
    distante segundo os custos, o que equivale ao "fecho" de uma reta.
    """
    n = len(pontos)
    if n <= 2:
        return list(range(n))
    try:
        return [int(v) for v in ConvexHull(pontos).vertices]
    except (QhullError, ValueError):
        i, j = np.unravel_index(int(np.argmax(custos)), custos.shape)
        return [int(i), int(j)] if i != j else [0]


def _inserir_restantes(custos: MatrizCustos, ciclo: list[int]) -> Tour:
    """Insere os vértices fora de ``ciclo`` pelo critério da razão."""
    c = custos.astype(np.float64)
    n = len(c)
    tour = list(ciclo)
    fora: npt.NDArray[np.intp] = np.setdiff1d(np.arange(n), tour)
    while len(fora):
        i = np.array(tour)
        j = np.roll(i, -1)  # aresta (i[e], j[e]) para cada posição e
        base = c[i, j]  # (arestas,)
        ida_volta = c[i][:, fora] + c[fora][:, j].T  # (arestas, fora)
        acrescimo = ida_volta - base[:, None]
        melhor_aresta = np.argmin(acrescimo, axis=0)  # para cada k fora
        colunas = np.arange(len(fora))
        # Arestas de custo 0 (pontos coincidentes): divide por 1 para não
        # gerar infinito; com custos inteiros, qualquer aresta real vale ≥ 1.
        razao = ida_volta[melhor_aresta, colunas] / np.maximum(base[melhor_aresta], 1.0)
        escolhido = int(np.argmin(razao))
        posicao = int(melhor_aresta[escolhido])
        tour.insert(posicao + 1, int(fora[escolhido]))
        fora = np.delete(fora, escolhido)
    return tour


def insercao_fecho_convexo(
    custos: MatrizCustos,
    pontos: Coordenadas | None = None,
    medidor: Medidor = MEDIDOR_NULO,
) -> tuple[int, Tour]:
    """Inserção a partir do fecho convexo de ``pontos``.

    Sem ``pontos`` (instância sem coordenadas), eles são reconstruídos por MDS
    clássico. Etapas medidas: ``mds`` (só quando roda), ``fecho``,
    ``insercao``, ``custo``.
    """
    if len(custos) == 0:
        return 0, []
    if pontos is None:
        with medidor.etapa("mds"):
            pontos, _ = mds_classico(custos)
    with medidor.etapa("fecho"):
        ciclo = ciclo_fecho_convexo(pontos, custos)
    with medidor.etapa("insercao"):
        tour = _comecar_em_zero(_inserir_restantes(custos, ciclo))
    with medidor.etapa("custo"):
        custo = custo_tour(custos, tour)
    return custo, tour
