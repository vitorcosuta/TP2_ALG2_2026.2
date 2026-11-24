import numpy as np
import pandas as pd
import pytest

from analises import e1
from analises.comum import ler_resultado


def _dados_sinteticos() -> pd.DataFrame:
    # agm cresce com n², emparelhamento com k³ (k = n/2), total com n².
    linhas = []
    for n in (10, 20, 50, 100, 200, 300, 500):
        k = n // 2
        for etapa, tempo in (
            ("total", 3e-7 * n**2),
            ("total/agm", 2e-7 * n**2 + 1e-4),
            ("total/emparelhamento", 1e-8 * k**3),
        ):
            for alg in ("twice_around_the_tree", "christofides"):
                linhas.append(
                    {
                        "algoritmo": alg,
                        "etapa": etapa,
                        "n": n,
                        "k_impares": k,
                        "tempo_mediana_s": tempo,
                        "pico_memoria_bytes": 8 * n**2,
                    }
                )
    return pd.DataFrame(linhas)


def test_regressao_escolhe_o_modelo_certo():
    regressao = e1.regressoes(_dados_sinteticos())
    melhores = e1.melhores_ajustes(regressao, "tempo_mediana_s").set_index(
        ["algoritmo", "etapa"]
    )
    assert melhores.loc[("christofides", "total/agm"), "modelo"] == "O(x^2)"
    emparelhamento = melhores.loc[("christofides", "total/emparelhamento")]
    assert emparelhamento["modelo"] == "O(x^3)"
    assert emparelhamento["variavel_x"] == "k_impares"
    assert emparelhamento["r2"] == pytest.approx(1.0)


def test_main_le_resultados_e_escreve_no_relatorio(tmp_path):
    resultados = tmp_path / "resultados"
    resultados.mkdir()
    dados = _dados_sinteticos()
    dados.to_csv(resultados / e1.ENTRADA, index=False)
    # Repetições individuais: o e1 também lê tempos_brutos_e1.csv.
    brutos = dados.assign(repeticao=1, tempo_s=dados["tempo_mediana_s"])
    brutos.to_csv(resultados / e1.ENTRADA_BRUTOS, index=False)
    tabelas, figuras = tmp_path / "tabelas", tmp_path / "figuras"

    e1.main(resultados, tabelas, figuras)

    regressao = pd.read_csv(tabelas / e1.SAIDA_REGRESSAO)
    assert set(regressao["modelo"]) == set(e1.MODELOS)
    assert np.isfinite(regressao["rmse"]).all()
    pngs = sorted(p.name for p in (figuras / e1.PASTA_FIGURAS).glob("*.png"))
    assert "christofides_total_emparelhamento.png" in pngs
    # Os dados sintéticos só têm os 2 aproximados: os demais alvos são pulados.
    presentes = [a for a in e1.ALVOS_GRAFICOS if a[0] != "vizinho_mais_proximo"]
    presentes = [a for a in presentes if a[0] != "insercao_fecho_convexo"]
    assert len(pngs) == len(presentes)


def test_resultado_ausente_explica_qual_coleta_falta(tmp_path):
    with pytest.raises(FileNotFoundError, match="rode antes a coleta"):
        ler_resultado("inexistente.csv", tmp_path)
