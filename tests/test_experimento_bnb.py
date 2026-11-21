import csv

from experimentos.branch_and_bound import COLUNAS, main


def _ler(caminho):
    with caminho.open(newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def test_gera_csv_retoma_e_consolida(tmp_path, capsys):
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

    (otimo,) = _ler(tmp_path / "otimos_bnb.csv")
    assert otimo["otimo_provado"] == "True" and otimo["melhor_custo"] == "3323"
    assert otimo["execucoes"] == otimo["execucoes_ok"] == "3"


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
    (otimo,) = _ler(tmp_path / "otimos_bnb.csv")
    assert otimo["otimo_provado"] == "False"
