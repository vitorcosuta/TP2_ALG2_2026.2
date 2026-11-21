import csv
import shutil
from pathlib import Path

import experimentos.aproximados_tsplib as script
from experimentos.algoritmos import ALGORITMOS

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"


def _ler(caminho):
    with caminho.open(newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def test_execucoes_e_subrotinas_com_os_5_algoritmos(tmp_path, monkeypatch):
    instancias = tmp_path / "instancias"
    instancias.mkdir()
    for nome in ("gr17.tsp", "ulysses16.tsp", "ulysses16.opt.tour"):
        shutil.copy(PASTA / nome, instancias)
    monkeypatch.setattr(script, "PASTA_SAIDA", tmp_path / "saida")
    monkeypatch.setattr(script, "REPETICOES", 2)

    script.rodar_tsplib(instancias)
    execucoes = _ler(tmp_path / "saida" / "execucoes.csv")
    assert {(x["instancia"], x["algoritmo"]) for x in execucoes} == {
        (i, a) for i in ("gr17", "ulysses16") for a in ALGORITMOS
    }
    por_chave = {(x["instancia"], x["algoritmo"]): x for x in execucoes}
    assert por_chave["gr17", "insercao_fecho_convexo"]["origem_pontos"] == "MDS"
    assert por_chave["gr17", "trivial"]["tipo"] == "heuristica"
    assert por_chave["gr17", "christofides"]["tipo"] == "aproximativo"
    assert float(por_chave["ulysses16", "christofides"]["gap_tsplib"]) >= 0

    etapas = {
        x["etapa"]
        for x in _ler(tmp_path / "saida" / "subrotinas.csv")
        if x["instancia"] == "gr17" and x["algoritmo"] == "insercao_fecho_convexo"
    }
    assert etapas == {
        "total",
        "total/mds",
        "total/fecho",
        "total/insercao",
        "total/custo",
    }
