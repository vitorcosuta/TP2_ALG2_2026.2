import gc
import time
import tracemalloc
from pathlib import Path

import pytest

from tsp.christofides import christofides
from tsp.instancia import carregar_instancia
from tsp.medicao import MEDIDOR_NULO, Medidor, medir
from tsp.twice_around_the_tree import twice_around_the_tree

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"
MEGA = 1_000_000


def _aloca(nbytes: int) -> int:
    dados = bytearray(nbytes)
    return len(dados)


def test_etapas_aninhadas_e_acumuladas():
    medidor = Medidor()
    with medidor.etapa("externa"):
        for _ in range(3):
            with medidor.etapa("interna"):
                time.sleep(0.002)
    assert set(medidor.etapas) == {"externa", "externa/interna"}
    interna, externa = medidor.etapas["externa/interna"], medidor.etapas["externa"]
    assert interna.chamadas == 3 and externa.chamadas == 1
    assert interna.tempo_s >= 0.006
    assert externa.tempo_s >= interna.tempo_s
    assert interna.pico_memoria_bytes is None  # memória não foi pedida


def test_etapa_registra_mesmo_com_excecao():
    medidor = Medidor()
    with pytest.raises(RuntimeError), medidor.etapa("falha"):
        raise RuntimeError
    assert medidor.etapas["falha"].chamadas == 1
    assert medidor._pilha_nomes == []


def test_pico_de_memoria_por_etapa_aninhada():
    tracemalloc.start()
    try:
        medidor = Medidor(medir_memoria=True)
        with medidor.etapa("pai"):
            _aloca(1 * MEGA)
            with medidor.etapa("grande"):
                _aloca(5 * MEGA)
            with medidor.etapa("pequena"):
                _aloca(2 * MEGA)
    finally:
        tracemalloc.stop()
    picos = {k: e.pico_memoria_bytes for k, e in medidor.etapas.items()}
    assert 5 * MEGA <= picos["pai/grande"] < 5.5 * MEGA
    assert 2 * MEGA <= picos["pai/pequena"] < 2.5 * MEGA
    # O pico do pai inclui o da filha maior, mesmo com reset_peak nas filhas.
    assert 5 * MEGA <= picos["pai"] < 5.5 * MEGA


def test_medidor_nulo_nao_registra():
    with MEDIDOR_NULO.etapa("qualquer"):
        pass
    assert MEDIDOR_NULO.etapas == {}


def test_medir_separa_tempo_e_memoria():
    medicao = medir(lambda m: _medida_simples(m), repeticoes=4)
    assert medicao.resultado == 2 * MEGA
    assert len(medicao.tempos_s["total/aloca"]) == 4
    assert medicao.picos_memoria_bytes["total/aloca"] >= 2 * MEGA
    linhas = {linha["etapa"]: linha for linha in medicao.linhas()}
    assert set(linhas) == {"total", "total/aloca"}
    assert linhas["total/aloca"]["repeticoes"] == 4
    assert not tracemalloc.is_tracing()


def _medida_simples(medidor: Medidor) -> int:
    with medidor.etapa("aloca"):
        return _aloca(2 * MEGA)


def test_medir_sem_memoria_e_com_gc_desativado():
    assert gc.isenabled()
    medicao = medir(
        lambda m: gc.isenabled(), repeticoes=2, medir_memoria=False, desativar_gc=True
    )
    assert medicao.resultado is False
    assert medicao.picos_memoria_bytes == {}
    assert gc.isenabled()


def test_medir_repeticoes_invalidas():
    with pytest.raises(ValueError):
        medir(lambda m: None, repeticoes=0)


@pytest.mark.parametrize(
    ("algoritmo", "etapas"),
    [
        (twice_around_the_tree, ["agm", "pre_ordem", "atalhos", "custo"]),
        (
            christofides,
            ["agm", "impares", "emparelhamento", "euleriano", "atalhos", "custo"],
        ),
    ],
    ids=["twice_around_the_tree", "christofides"],
)
def test_algoritmos_instrumentados(algoritmo, etapas):
    custos = carregar_instancia(PASTA / "berlin52.tsp").custos
    sem_medidor = algoritmo(custos)
    medicao = medir(lambda m: algoritmo(custos, medidor=m), repeticoes=3)
    assert medicao.resultado == sem_medidor  # instrumentar não muda o resultado
    esperadas = {"total"} | {f"total/{e}" for e in etapas}
    assert set(medicao.tempos_s) == esperadas
    assert set(medicao.picos_memoria_bytes) == esperadas
    soma = sum(medicao.tempos_s[f"total/{e}"][0] for e in etapas)
    assert soma <= medicao.tempos_s["total"][0]
