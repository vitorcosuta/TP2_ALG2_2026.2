import csv
import shutil
from pathlib import Path

import pytest

from experimentos import (
    aproximados_aleatorios,
    aproximados_tsplib,
    bnb_aleatorias,
    branch_and_bound,
    features,
    verificar,
)

PASTA = Path(__file__).resolve().parent.parent / "TSP_Instances"


@pytest.fixture(scope="module")
def coleta_completa(tmp_path_factory):
    """Coleta em miniatura, com o plano reduzido a burma14 (principal),
    swiss42 (extra), N = 10 e 20 nas aleatórias do E1 e N = 8 no B&B."""
    raiz = tmp_path_factory.mktemp("coleta")
    instancias, res = raiz / "instancias", raiz / "resultados"
    instancias.mkdir()
    for nome in ("burma14", "swiss42"):
        shutil.copy(PASTA / f"{nome}.tsp", instancias)

    mp = pytest.MonkeyPatch()
    mp.setattr(verificar, "_instancias_tsplib", lambda: ["burma14", "swiss42"])
    mp.setattr(verificar, "SELECIONADAS", ["burma14"])
    mp.setattr(verificar, "EXTRAS_REGRESSAO", ["swiss42"])
    mp.setattr(aproximados_aleatorios, "TAMANHOS", [10, 20])
    mp.setattr(bnb_aleatorias, "TAMANHOS", [8])
    mp.setattr(bnb_aleatorias, "SEMENTES", 1)
    mp.setattr(aproximados_tsplib, "PASTA_SAIDA", res)
    mp.setattr(aproximados_tsplib, "REPETICOES", 1)

    features.main(
        ["--mds", "--pasta", str(instancias), "--saida", str(res / "features.csv")]
    )
    aproximados_tsplib.rodar_tsplib(instancias)
    aproximados_tsplib.rodar_tsplib(instancias, desativar_gc=True)
    aproximados_aleatorios.coletar_dados_aleatorios(res, [10, 20], repeticoes=1)
    branch_and_bound.main(
        ["--instancias", "burma14", "--saida", str(res / "bnb_p.csv")]
    )
    branch_and_bound.main(
        ["--instancias", "swiss42", "--estrategias", "christofides"]
        + ["--limite-memoria", "0"]  # pula a execução de memória (lenta)
        + ["--saida", str(res / "bnb_extras.csv")]
    )
    bnb_aleatorias.main(
        [
            "--tamanhos",
            "8",
            "--sementes",
            "1",
            "--saida",
            str(res / "bnb_aleatorias.csv"),
        ]
    )
    branch_and_bound.main(["--consolidar", "--saida", str(res / "bnb_p.csv")])
    yield res
    mp.undo()


def test_coleta_completa_nao_tem_problemas(coleta_completa):
    vazio = {nome: [] for nome in verificar.VERIFICACOES}
    assert verificar.verificar(coleta_completa) == vazio


def test_detecta_falta_duplicata_e_otimos_desatualizados(coleta_completa, tmp_path):
    copia = tmp_path / "resultados"
    shutil.copytree(coleta_completa, copia)
    (copia / "execucoes_sem_gc.csv").unlink()
    with (copia / "bnb_p.csv").open() as f:
        linhas = list(csv.DictReader(f))
    with (copia / "bnb_p.csv").open("a", newline="") as f:  # duplica uma linha
        csv.DictWriter(f, list(linhas[0])).writerow(linhas[0])

    problemas = verificar.verificar(copia)
    assert problemas["execucoes_sem_gc.csv (passo 2)"] == [
        "execucoes_sem_gc.csv não existe (passo 2)"
    ]
    bnb = problemas["B&B: principal, extras e aleatórias (passos 3, 3b, 4)"]
    assert any(p.startswith("duplicado ('burma14'") for p in bnb)
    assert problemas["otimos_bnb.csv (passo 5)"] == [
        "otimos_bnb.csv desatualizado: rode --consolidar de novo"
    ]
    assert verificar.main(["--pasta", str(copia)]) == 1


def test_detecta_parametros_fora_do_oficial(coleta_completa, tmp_path):
    copia = tmp_path / "resultados"
    shutil.copytree(coleta_completa, copia)
    branch_and_bound.main(
        ["--instancias", "burma14", "--estrategias", "christofides"]
        + ["--limite-tempo", "5", "--saida", str(copia / "bnb_teste.csv")]
    )
    bnb = verificar.verificar(copia)[
        "B&B: principal, extras e aleatórias (passos 3, 3b, 4)"
    ]
    assert "burma14/christofides: limite 5.0 s, held_karp" in bnb
