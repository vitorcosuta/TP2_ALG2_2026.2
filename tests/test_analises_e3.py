import numpy as np
import pandas as pd
import pytest

from analises import e3

INSTANCIAS = [f"inst{i:02d}" for i in range(13)]


def _features(rng: np.random.Generator) -> pd.DataFrame:
    n = np.arange(14, 14 + 3 * len(INSTANCIAS), 3)
    dados = {f: rng.random(len(INSTANCIAS)) for f in e3.FEATURES}
    dados["n"] = n
    # Feature que acompanha N (como a fração no fecho convexo): ρ alto, parcial ~0.
    dados["fracao_fecho_convexo"] = 1.0 / n
    return pd.DataFrame(
        {"instancia": INSTANCIAS, "origem_coordenadas": "NODE_COORD_SECTION", **dados}
    )


def _dados_sinteticos(semente: int = 0):
    """O gap do Christofides depende só de cv_arestas_agm; o resto é ruído."""
    rng = np.random.default_rng(semente)
    features = _features(rng)
    otimos = pd.DataFrame(
        {
            "instancia": INSTANCIAS + ["grande"],
            "otimo_provado": [True] * len(INSTANCIAS) + [False],
            "melhor_custo": 1000,
        }
    )
    linhas = []
    for alg in [*e3.ALGORITMOS, "trivial"]:
        for _, f in features.iterrows():
            if alg == "christofides":
                gap = 0.05 + 0.2 * f["cv_arestas_agm"]
            else:
                gap = 0.1 + 0.3 * rng.random()
            linhas.append(
                {
                    "algoritmo": alg,
                    "instancia": f["instancia"],
                    "custo": 1000 * (1 + gap),
                }
            )
        linhas.append({"algoritmo": alg, "instancia": "grande", "custo": 9999})
    return pd.DataFrame(linhas), otimos, features


def test_base_filtra_otimo_nao_provado_e_trivial():
    execucoes, otimos, features = _dados_sinteticos()
    base = e3.montar_base(execucoes, otimos, features)
    assert "grande" not in set(base["instancia"])
    assert "trivial" not in set(base["algoritmo"])
    assert len(base) == len(e3.ALGORITMOS) * len(INSTANCIAS)
    christofides = base[base["algoritmo"] == "christofides"]
    esperado = 0.05 + 0.2 * christofides["cv_arestas_agm"]
    assert christofides["gap"].to_numpy() == pytest.approx(esperado.to_numpy())


def test_correlacao_parcial_desconta_o_controle():
    z = np.arange(20, dtype=float)
    # x tem os mesmos postos de z: nada sobra depois de descontar z.
    assert np.isnan(e3.correlacao_parcial(2 * z + 1, z**2, z))
    rng = np.random.default_rng(3)
    ruido = rng.random(20)
    assert e3.correlacao_parcial(ruido, ruido + 0.01 * z, z) > 0.9


def test_correlacoes_acham_a_feature_certa():
    base = e3.montar_base(*_dados_sinteticos())
    corr = e3.correlacoes(base).set_index(["algoritmo", "feature"])
    assert corr.loc[("christofides", "cv_arestas_agm"), "spearman"] == pytest.approx(
        1.0
    )
    # Acompanha N: correlação bruta alta, parcial não definida para N.
    assert np.isnan(corr.loc[("christofides", "n"), "parcial_dado_n"])


def test_modelos_preferem_a_feature_verdadeira():
    base = e3.montar_base(*_dados_sinteticos())
    mods = e3.modelos(base)
    christofides = mods[mods["algoritmo"] == "christofides"]
    melhor = christofides.iloc[0]
    assert melhor["features"] == "n + cv_arestas_agm"
    assert melhor["q2_loo"] > 0.95
    media = christofides[christofides["modelo"] == "media"].iloc[0]
    assert media["q2_loo"] <= 0  # a média nunca prevê melhor que a média
    # Nos algoritmos com gap aleatório, nenhum modelo deve generalizar bem.
    ruido = mods[mods["algoritmo"] == "vizinho_mais_proximo"]
    assert ruido["q2_loo"].max() < 0.5


def test_pesos_tem_uma_linha_por_feature():
    base = e3.montar_base(*_dados_sinteticos())
    tabela = e3.pesos(base)
    assert len(tabela) == len(e3.ALGORITMOS) * len(e3.FEATURES)
    christofides = tabela[tabela["algoritmo"] == "christofides"].set_index("feature")
    assert christofides["importancia_arvore"].idxmax() == "cv_arestas_agm"
    assert christofides["coef_ridge_padronizado"].abs().idxmax() == "cv_arestas_agm"


def test_features_com_nan_sao_ignoradas():
    execucoes, otimos, features = _dados_sinteticos()
    features["razao_aspecto"] = np.nan  # features.csv gerado sem --mds
    base = e3.montar_base(execucoes, otimos, features)
    assert "razao_aspecto" not in e3.features_disponiveis(base)
    assert "razao_aspecto" not in set(e3.correlacoes(base)["feature"])


def test_main_le_resultados_e_escreve_no_relatorio(tmp_path):
    resultados = tmp_path / "resultados"
    resultados.mkdir()
    execucoes, otimos, features = _dados_sinteticos()
    execucoes.to_csv(resultados / e3.ENTRADA_EXECUCOES, index=False)
    otimos.to_csv(resultados / e3.ENTRADA_OTIMOS, index=False)
    features.to_csv(resultados / e3.ENTRADA_FEATURES, index=False)
    tabelas, figuras = tmp_path / "tabelas", tmp_path / "figuras"

    e3.main(resultados, tabelas, figuras)

    for nome in ("e3_base", "e3_correlacoes", "e3_modelos", "e3_pesos"):
        assert (tabelas / f"{nome}.csv").exists()
    pngs = {p.name for p in (figuras / e3.PASTA_FIGURAS).glob("*.png")}
    assert {"correlacoes.png", "arvore_christofides.png"} <= pngs
    assert "gap_features_christofides.png" in pngs


def test_sem_otimo_provado_explica_o_que_falta(tmp_path):
    resultados = tmp_path / "resultados"
    resultados.mkdir()
    execucoes, otimos, features = _dados_sinteticos()
    otimos["otimo_provado"] = False
    execucoes.to_csv(resultados / e3.ENTRADA_EXECUCOES, index=False)
    otimos.to_csv(resultados / e3.ENTRADA_OTIMOS, index=False)
    features.to_csv(resultados / e3.ENTRADA_FEATURES, index=False)
    with pytest.raises(ValueError, match="--consolidar"):
        e3.main(resultados, tmp_path / "t", tmp_path / "f")