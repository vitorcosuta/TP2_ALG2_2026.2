import itertools
from pathlib import Path

import numpy as np
import pytest

from tsp.branch_and_bound import (
    BranchAndBound,
    branch_and_bound,
    penalidades_held_karp,
)
from tsp.christofides import christofides
from tsp.instancia import carregar_instancia
from tsp.medicao import medir
from tsp.tour import custo_tour, eh_tour_valido

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
LIMITES = ["dois_menores", "agm", "held_karp"]

# Instâncias pequenas em que o fecho métrico não alterou nenhum custo, logo o
# ótimo da TSPLIB continua valendo.
OTIMOS = {"burma14": 3323, "ulysses16": 6859, "ulysses22": 7013}


def _aleatoria(semente: int) -> np.ndarray:
    rng = np.random.default_rng(semente)
    n = int(rng.integers(1, 10))
    if semente % 2:  # euclidiana
        p = rng.random((n, 2)) * 100
        c = np.rint(np.hypot(*(p[:, None] - p[None]).transpose(2, 0, 1)))
    else:  # pesos arbitrários (não necessariamente métricos)
        m = np.triu(rng.integers(1, 50, (n, n)), 1)
        c = m + m.T
    c = c.astype(np.int64)
    if n > 2:
        c[c == 0] = 1
    np.fill_diagonal(c, 0)
    return c


def _forca_bruta(c: np.ndarray) -> int:
    n = len(c)
    if n == 1:
        return 0
    return min(custo_tour(c, [0, *p]) for p in itertools.permutations(range(1, n)))


@pytest.mark.parametrize("semente", range(60))
@pytest.mark.parametrize("limite", LIMITES)
def test_contra_forca_bruta(semente: int, limite):
    c = _aleatoria(semente)
    otimo = _forca_bruta(c)
    for inicial in (None, list(range(len(c)))[::-1]):
        r = branch_and_bound(c, inicial, limite_inferior=limite)
        assert r.status == "OK" and r.otimo_provado
        assert r.custo == otimo == custo_tour(c, r.tour)
        assert eh_tour_valido(r.tour, len(c)) and r.tour[0] == 0
        assert r.limite_raiz <= otimo


@pytest.mark.parametrize("nome", sorted(OTIMOS))
def test_otimos_conhecidos(nome: str):
    c = carregar_instancia(PASTA / f"{nome}.tsp").custos
    _, inicial = christofides(c)
    r = branch_and_bound(c, inicial, limite_tempo_s=60)
    assert r.status == "OK"
    assert r.custo == OTIMOS[nome] == custo_tour(c, r.tour)
    assert r.limite_raiz <= OTIMOS[nome] <= r.custo_inicial


def test_tempo_esgotado_preserva_contadores_e_melhor_tour():
    c = carregar_instancia(PASTA / "st70.tsp").custos
    _, inicial = christofides(c)
    r = branch_and_bound(c, inicial, limite_tempo_s=0, limite_inferior="dois_menores")
    assert r.status == "NA" and not r.otimo_provado
    assert r.nos_expandidos == 4096  # primeira verificação do relógio
    assert r.nos_podados > 0 and r.profundidade_maxima > 1
    assert eh_tour_valido(r.tour, len(c))
    assert r.custo == custo_tour(c, r.tour) <= r.custo_inicial


def test_sem_warm_start_e_sem_tempo_nao_tem_tour():
    c = carregar_instancia(PASTA / "st70.tsp").custos
    r = branch_and_bound(c, None, limite_tempo_s=0, limite_inferior="dois_menores")
    # A busca pode ou não ter chegado a uma folha antes da primeira verificação.
    assert r.status == "NA"
    assert r.custo_inicial == float("inf")
    assert (r.custo is None) == (r.tour == [])


def test_warm_start_e_rotacionado_para_comecar_em_zero():
    c = carregar_instancia(PASTA / "burma14.tsp").custos
    inicial = list(range(5, 14)) + list(range(5))
    r = branch_and_bound(c, inicial, limite_tempo_s=0, limite_inferior="dois_menores")
    assert r.custo_inicial == custo_tour(c, inicial)
    assert r.tour[0] == 0


def test_warm_start_otimo_nao_gera_melhorias():
    c = carregar_instancia(PASTA / "burma14.tsp").custos
    otimo = branch_and_bound(c).tour
    r = branch_and_bound(c, otimo)
    assert r.custo == OTIMOS["burma14"] and r.melhorias == 0


def test_entradas_invalidas():
    c = carregar_instancia(PASTA / "burma14.tsp").custos
    with pytest.raises(ValueError):
        branch_and_bound(c, [0, 1, 2])
    with pytest.raises(ValueError):
        BranchAndBound(c, limite_inferior="inexistente")  # type: ignore[arg-type]


def test_held_karp_e_limite_valido_e_justo():
    c = carregar_instancia(PASTA / "burma14.tsp").custos
    pi, limite = penalidades_held_karp(c)
    assert len(pi) == len(c)
    assert 0.99 * OTIMOS["burma14"] <= limite <= OTIMOS["burma14"] + 1e-6


def test_cronometro_do_limite_e_medidor():
    c = carregar_instancia(PASTA / "ulysses16.tsp").custos
    bnb = BranchAndBound(c, cronometrar_limite=True)
    medicao = medir(lambda m: bnb.resolver(None, m), repeticoes=1)
    r = medicao.resultado
    assert r.tempo_limite_inferior_s is not None and r.tempo_limite_inferior_s > 0
    assert r.chamadas_limite_inferior == r.nos_podados + r.nos_expandidos - 1
    assert {"total/penalidades", "total/busca"} <= set(medicao.tempos_s)
