import math

import numpy as np
import pandas as pd
import pytest

from analises import e1_bnb


def _aleatorias(c: float = 1.3) -> pd.DataFrame:
    """Tempo e nós crescem exatamente como a·cⁿ (com ruído pequeno)."""
    rng = np.random.default_rng(0)
    linhas = []
    for n in range(10, 55, 5):
        for semente in range(5):
            ruido = 10 ** rng.normal(0, 0.05)
            linhas.append(
                {
                    "instancia": f"rand_n{n:03d}_s{semente}",
                    "n": n,
                    "estrategia": "christofides",
                    "status": "OK",
                    "tempo_bnb_s": 1e-4 * c**n * ruido,
                    "nos_expandidos": 2 * c**n * ruido,
                    "pico_memoria_bytes": 30 * n**2 + 5000.0,
                    "pico_memoria_busca_bytes": 80 * n + 3000.0,
                }
            )
    return pd.DataFrame(linhas)


def test_exponencial_vence_e_recupera_a_base():
    reg = e1_bnb.regressoes(_aleatorias(c=1.3))
    tempo = reg[reg["metrica"] == "tempo_bnb_s"]
    melhor = tempo.iloc[0]
    assert melhor["modelo"] == "a·c^n"
    assert melhor["parametro"] == pytest.approx(1.3, rel=0.02)
    assert melhor["r2_log"] > 0.99
    # Os modelos de 1 parâmetro crescem rápido demais para esses dados.
    assert tempo.set_index("modelo").loc["a·n!", "r2_log"] < 0


def test_ajuste_ignora_valores_nao_positivos():
    n = np.array([10, 20, 30, 40.0])
    y = np.array([0.0, 2.0, 4.0, 8.0])  # o 0 não tem log
    linhas = e1_bnb.ajustar_log(n, y)
    assert all(math.isfinite(linha["r2_log"]) for linha in linhas)


def test_memoria_total_quadratica_e_busca_linear():
    mem = e1_bnb.regressoes_memoria(_aleatorias()).groupby("metrica").head(1)
    melhores = dict(zip(mem["metrica"], mem["modelo"], strict=True))
    assert melhores["pico_memoria_bytes"] == "O(n^2)"
    assert melhores["pico_memoria_busca_bytes"] == "O(n)"


def test_memoria_com_valores_faltando():
    dados = _aleatorias()
    dados.loc[dados["n"] >= 45, "pico_memoria_bytes"] = np.nan
    mem = e1_bnb.regressoes_memoria(dados)
    assert np.isfinite(mem["r2"]).all()


def test_main_escreve_tabelas_e_figuras(tmp_path):
    resultados = tmp_path / "resultados"
    resultados.mkdir()
    _aleatorias().to_csv(resultados / e1_bnb.ENTRADA_ALEATORIAS, index=False)
    tsplib = _aleatorias().head(3).assign(estrategia="christofides")
    for nome in e1_bnb.ENTRADAS_TSPLIB:
        tsplib.to_csv(resultados / nome, index=False)
    tabelas, figuras = tmp_path / "tabelas", tmp_path / "figuras"

    e1_bnb.main(resultados, tabelas, figuras)

    assert (tabelas / e1_bnb.SAIDA).exists()
    assert (tabelas / e1_bnb.SAIDA_MEMORIA).exists()
    pngs = {p.name for p in (figuras / e1_bnb.PASTA_FIGURAS).glob("*.png")}
    assert pngs == {"bnb_tempo.png", "bnb_nos.png", "bnb_memoria.png"}
