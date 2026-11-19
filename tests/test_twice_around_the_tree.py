from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from tsp.agm import agm_prim, custo_agm
from tsp.instancia import carregar_instancia
from tsp.tour import custo_tour, eh_tour_valido
from tsp.twice_around_the_tree import (
    caminhamento_pre_ordem,
    remover_repetidos,
    twice_around_the_tree,
)

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
INSTANCIAS = sorted(PASTA.glob("*.tsp"))

# Árvore:   0
#          / \
#         1   4
#        / \
#       2   3
PAI_EXEMPLO = np.array([-1, 0, 1, 1, 0], dtype=np.int64)


def test_caminhamento_anota_ida_e_volta():
    assert caminhamento_pre_ordem(PAI_EXEMPLO) == [0, 1, 2, 1, 3, 1, 0, 4, 0]


def test_remover_repetidos_mantem_primeira_ocorrencia():
    # Exemplo do enunciado: a-b-c-b-d vira a-b-c-d.
    assert remover_repetidos([0, 1, 2, 1, 3]) == [0, 1, 2, 3]
    assert remover_repetidos([0, 1, 2, 1, 3, 1, 0, 4, 0]) == [0, 1, 2, 3, 4]


def test_custo_tour_inclui_volta():
    custos = np.array([[0, 1, 4], [1, 0, 2], [4, 2, 0]], dtype=np.int64)
    assert custo_tour(custos, [0, 1, 2]) == 1 + 2 + 4


def test_grafos_minimos():
    assert twice_around_the_tree(np.zeros((1, 1), dtype=np.int64)) == (0, [0])
    custos = np.array([[0, 3], [3, 0]], dtype=np.int64)
    assert twice_around_the_tree(custos) == (6, [0, 1])


@pytest.mark.parametrize("caminho", INSTANCIAS, ids=lambda p: p.stem)
def test_instancias(caminho: Path):
    custos = carregar_instancia(caminho).custos
    n = len(custos)
    pai = agm_prim(custos)
    passeio = caminhamento_pre_ordem(pai)
    assert len(passeio) == 2 * n - 1
    assert passeio[0] == passeio[-1] == 0
    # O passeio percorre cada aresta da AGM duas vezes.
    custo_passeio = sum(int(custos[u, v]) for u, v in pairwise(passeio))
    assert custo_passeio == 2 * custo_agm(custos, pai)

    custo, tour = twice_around_the_tree(custos)
    assert eh_tour_valido(tour, n)
    assert tour[0] == 0
    assert custo == custo_tour(custos, tour)
    # Garantias: AGM ≤ ótimo ≤ custo ≤ 2·AGM.
    assert custo_agm(custos, pai) <= custo <= custo_passeio
