import csv

import pytest

from experimentos.branch_and_bound import COLUNAS, LimiteDeTempoMisturado, main


def _ler(caminho):
    with caminho.open(newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def test_gera_csv_retoma_e_so_consolida_quando_pedido(tmp_path, capsys):
    saida = tmp_path / "bnb_teste.csv"
    base = ["--instancias", "burma14", "--saida", str(saida), "--limite-tempo", "30"]

    main([*base, "--estrategias", "trivial", "christofides"])
    linhas = _ler(saida)
    assert [x["estrategia"] for x in sorted(linhas, key=lambda x: x["estrategia"])] == [
        "christofides",
        "trivial",
    ]
    assert list(linhas[0]) == COLUNAS
    for linha in linhas:
        assert linha["status"] == "OK" and linha["custo"] == "3323"
        assert int(linha["pico_memoria_bytes"]) > 0  # terminou rápido: mede memória
        assert len(linha["tour"].split("-")) == 14
        assert linha["processador"] and int(linha["nucleos_logicos"]) >= 1

    # Rodar de novo pula o que já existe e só acrescenta o que falta.
    main([*base, "--estrategias", "trivial", "sem_warm_start"])
    assert "1 combinações já estão" in capsys.readouterr().out
    linhas = _ler(saida)
    assert len(linhas) == 3
    sem = next(x for x in linhas if x["estrategia"] == "sem_warm_start")
    assert sem["custo_inicial"] == "" and sem["tempo_heuristica_s"] == "0.0"

    # Sem --consolidar, nenhum otimos_bnb.csv é gerado (evita versões parciais).
    assert not (tmp_path / "otimos_bnb.csv").exists()
    main(["--consolidar", "--saida", str(saida)])
    (otimo,) = _ler(tmp_path / "otimos_bnb.csv")
    assert otimo["otimo_provado"] == "True" and otimo["melhor_custo"] == "3323"
    assert otimo["execucoes"] == otimo["execucoes_ok"] == "3"


def test_teste_rapido_nao_se_mistura_com_execucao_oficial(tmp_path):
    saida = tmp_path / "bnb_st70.csv"
    rapido = ["--instancias", "burma14", "--estrategias", "christofides"]
    main([*rapido, "--limite-tempo", "5", "--saida", str(saida)])
    # A execução "oficial" no mesmo arquivo para com erro em vez de pular.
    with pytest.raises(LimiteDeTempoMisturado, match="outra --saida"):
        main([*rapido, "--limite-tempo", "30", "--saida", str(saida)])
    assert len(_ler(saida)) == 1


def test_tempo_esgotado_nao_mede_memoria(tmp_path):
    saida = tmp_path / "bnb.csv"
    main(
        ["--instancias", "st70", "--estrategias", "christofides"]
        + ["--limite-tempo", "0", "--limite-inferior", "dois_menores"]
        + ["--saida", str(saida)]
    )
    (linha,) = _ler(saida)
    assert linha["status"] == "NA" and linha["pico_memoria_bytes"] == ""
    assert int(linha["custo"]) <= int(linha["custo_inicial"])
    main(["--consolidar", "--saida", str(saida)])
    (otimo,) = _ler(tmp_path / "otimos_bnb.csv")
    assert otimo["otimo_provado"] == "False"


def test_coluna_conjunto_separa_principal_extra_e_aleatorio(tmp_path):
    from experimentos import bnb_aleatorias

    saida = tmp_path / "bnb.csv"
    comum = ["--estrategias", "christofides", "--limite-tempo", "30"]
    main(["--instancias", "burma14", "--saida", str(saida), *comum])
    # swiss42 só para ver o rótulo: sem tempo e com o limite barato, é rápido.
    # (em outro arquivo: um arquivo só aceita um limite de tempo)
    main(
        ["--instancias", "swiss42", "--saida", str(tmp_path / "bnb_extra.csv")]
        + ["--limite-tempo", "0", "--estrategias", "christofides"]
        + ["--limite-inferior", "dois_menores"]
    )
    bnb_aleatorias.main(
        ["--tamanhos", "8", "10", "--sementes", "2"]
        + ["--saida", str(tmp_path / "bnb_aleatorias.csv"), *comum]
    )
    tsplib = {x["instancia"]: x for x in _ler(saida) + _ler(tmp_path / "bnb_extra.csv")}
    assert tsplib["burma14"]["conjunto"] == "principal"
    assert tsplib["swiss42"]["conjunto"] == "extra"
    assert tsplib["burma14"]["semente"] == ""

    aleatorias = _ler(tmp_path / "bnb_aleatorias.csv")
    assert {x["conjunto"] for x in aleatorias} == {"aleatorio"}
    assert sorted((x["n"], x["semente"]) for x in aleatorias) == [
        ("10", "0"), ("10", "1"), ("8", "0"), ("8", "1"),
    ]  # fmt: skip
    assert all(x["status"] == "OK" for x in aleatorias)

    # Aleatórias não entram nos ótimos de referência.
    main(["--consolidar", "--saida", str(saida)])
    otimos = {x["instancia"]: x["conjunto"] for x in _ler(tmp_path / "otimos_bnb.csv")}
    assert otimos == {"burma14": "principal", "swiss42": "extra"}


def test_instancia_aleatoria_e_deterministica_e_metrica():
    import numpy as np

    from experimentos.instancias import instancia_aleatoria

    a, b = instancia_aleatoria(12, 3), instancia_aleatoria(12, 3)
    assert np.array_equal(a.custos, b.custos)
    assert not np.array_equal(a.custos, instancia_aleatoria(12, 4).custos)
    c = a.custos
    assert (c[:, :, None] <= c[:, None, :] + c[None, :, :]).all()  # triangular
