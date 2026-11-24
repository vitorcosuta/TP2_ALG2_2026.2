import numpy as np
import pandas as pd
import pytest

from analises import e2

# Nós expandidos por estratégia: quanto melhor o limite, menos nós.
GAPS = {
    "trivial": 1.0,
    "vizinho_mais_proximo": 0.25,
    "twice_around_the_tree": 0.30,
    "christofides": 0.10,
    "insercao_fecho_convexo": 0.02,
}


def _bnb(n: int, instancia: str, otimo: int, na_trivial: bool = False) -> list[dict]:
    linhas = []
    for est, gap in GAPS.items():
        nos = int(1000 * n * gap) + 1
        linhas.append(
            {
                "instancia": instancia,
                "n": n,
                "estrategia": est,
                "custo_inicial": round(otimo * (1 + gap)),
                "limite_raiz": otimo - 5,
                "tempo_heuristica_s": 1e-4,
                "tempo_bnb_s": nos / 1000,
                "nos_expandidos": nos,
                "nos_podados": 2 * nos,
                "profundidade_maxima": n,
                "melhorias": 1,
                "status": "NA" if (na_trivial and est == "trivial") else "OK",
                "custo": otimo,
            }
        )
    return linhas


def _dados():
    bnb = pd.DataFrame(_bnb(14, "pequena", 1000) + _bnb(70, "grande", 5000, True))
    otimos = pd.DataFrame(
        {"instancia": ["pequena", "grande"], "melhor_custo": [1000, 5000]}
    )
    return bnb, otimos


def test_tabela_calcula_gaps_e_razoes():
    tabela = e2.montar_tabela(*_dados())
    assert len(tabela) == 10
    linha = tabela[
        (tabela["instancia"] == "pequena") & (tabela["estrategia"] == "christofides")
    ].iloc[0]
    assert linha["gap_limite_inicial"] == pytest.approx(0.10)
    assert linha["gap_limite_raiz"] == pytest.approx(0.005)
    assert linha["nos_rel_trivial"] == pytest.approx(1401 / 14001)
    assert linha["aceleracao_vs_trivial"] > 1
    # Ordem das estratégias segue ESTRATEGIAS dentro de cada instância.
    assert tabela["estrategia"].tolist()[:5] == e2.ESTRATEGIAS
    grande_trivial = tabela[
        (tabela["instancia"] == "grande") & (tabela["estrategia"] == "trivial")
    ]
    assert grande_trivial["truncado"].all()


def test_correlacao_por_instancia_e_geral():
    corr = e2.correlacoes(e2.montar_tabela(*_dados())).set_index("instancia")
    assert corr.loc["pequena", "spearman"] == pytest.approx(1.0)
    assert corr.loc["pequena", "melhor_estrategia"] == "insercao_fecho_convexo"
    # Na instância com NA, o trivial recebe o pior posto e a correlação continua 1.
    assert corr.loc["grande", "spearman"] == pytest.approx(1.0)
    # A linha geral exclui as instâncias cuja referência estourou o tempo.
    assert corr.loc["(todas)", "execucoes"] == 5


def test_postos_poem_na_no_fim():
    g = pd.DataFrame(
        {"nos_expandidos": [10, 500, 100], "truncado": [False, False, True]}
    )
    assert e2._postos_esforco(g).tolist() == [1.0, 2.0, 3.0]


def test_tradeoff_e_colapso():
    tabela = e2.montar_tabela(*_dados())
    trade = e2.tradeoff(tabela).set_index("estrategia")
    assert e2.REFERENCIA not in trade.index
    assert (trade["aceleracao_min"] > 1).all()
    assert trade.loc["insercao_fecho_convexo", "nao_compensou"] == ""
    col = e2.colapso(tabela).set_index("instancia")
    assert col.loc["grande", "estrategias_na"] == 1
    assert col.loc["grande", "quais_na"] == "trivial"
    assert col.loc["pequena", "estrategias_na"] == 0


def test_main_escreve_tabelas_e_figuras(tmp_path):
    resultados = tmp_path / "resultados"
    resultados.mkdir()
    bnb, otimos = _dados()
    bnb[bnb["n"] < 40].to_csv(resultados / "bnb_pequenas.csv", index=False)
    bnb[bnb["n"] >= 40].to_csv(resultados / "bnb_grandes.csv", index=False)
    otimos.to_csv(resultados / e2.ENTRADA_OTIMOS, index=False)
    tabelas, figuras = tmp_path / "tabelas", tmp_path / "figuras"

    saidas = e2.main(resultados, tabelas, figuras)

    for nome in saidas:
        assert (tabelas / nome).exists()
    pngs = {p.name for p in (figuras / e2.PASTA_FIGURAS).glob("*.png")}
    assert pngs == {"gap_vs_nos.png", "mapa_nos.png", "tempos_maiores.png"}
    assert np.isfinite(saidas["e2_tabela.csv"]["aceleracao_vs_trivial"]).all()
