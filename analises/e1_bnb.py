

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import gammaln

from analises.comum import FIGURAS, TABELAS, ler_resultado, salvar_figura, salvar_tabela

ENTRADA_ALEATORIAS = "bnb_aleatorias.csv"
ENTRADAS_TSPLIB = ["bnb_pequenas.csv", "bnb_grandes.csv"]
ESTRATEGIA_TSPLIB = "christofides"  # mesma das aleatórias
SAIDA = "regressao_e1_bnb.csv"
SAIDA_MEMORIA = "regressao_e1_bnb_memoria.csv"
PASTA_FIGURAS = "e1"
METRICAS = {"tempo_bnb_s": "Tempo do B&B (s)", "nos_expandidos": "Nós expandidos"}

LN10 = math.log(10)


# log10 de f(n) para os modelos de um parâmetro (y ≈ a·f(n)).
def _log10_n2_2n(n: np.ndarray) -> np.ndarray:
    return 2 * np.log10(n) + n * math.log10(2)


def _log10_fatorial(n: np.ndarray) -> np.ndarray:
    return gammaln(n + 1) / LN10


# Modelos com 2 parâmetros: log10 y = log10 a + b · g(n).
MODELOS_2P: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "a·c^n": lambda n: n,
    "a·n^p (polinomial)": np.log10,
}
MODELOS_1P: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "a·n²·2^n": _log10_n2_2n,
    "a·n!": _log10_fatorial,
}
MODELOS_MEMORIA: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "O(n)": lambda n: n,
    "O(n log n)": lambda n: n * np.log2(n),
    "O(n^2)": lambda n: n**2,
    "O(n^3)": lambda n: n**3,
}


#  ajustes
def _metricas(logy: np.ndarray, previsto: np.ndarray) -> tuple[float, float]:
    residuos = logy - previsto
    ss_tot = float(np.sum((logy - logy.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residuos**2)) / ss_tot if ss_tot > 0 else float("nan")
    return r2, float(np.sqrt(np.mean(residuos**2)))


def ajustar_log(n: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Ajusta os modelos em log10(y). Pontos com y ≤ 0 são descartados."""
    validos = y > 0
    n, logy = n[validos].astype(float), np.log10(y[validos].astype(float))
    linhas: list[dict[str, Any]] = []
    for nome, g in MODELOS_2P.items():
        b, log_a = np.polyfit(g(n), logy, 1)
        r2, rmse = _metricas(logy, log_a + b * g(n))
        parametro = 10**b if nome == "a·c^n" else b  # c ou p
        linhas.append(
            {"modelo": nome, "a": 10**log_a, "parametro": parametro, "r2_log": r2,
             "rmse_log10": rmse}
        )  # fmt: skip
    for nome, f in MODELOS_1P.items():
        log_a = float(np.mean(logy - f(n)))
        r2, rmse = _metricas(logy, log_a + f(n))
        linhas.append(
            {"modelo": nome, "a": 10**log_a, "parametro": np.nan, "r2_log": r2,
             "rmse_log10": rmse}
        )  # fmt: skip
    return linhas


def regressoes(dados: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (métrica, modelo), ordenada pelo R² dentro da métrica."""
    linhas = []
    n = dados["n"].to_numpy()
    for metrica in METRICAS:
        for res in ajustar_log(n, dados[metrica].to_numpy()):
            linhas.append({"metrica": metrica, "pontos": len(dados), **res})
    tabela = pd.DataFrame(linhas)
    return tabela.sort_values(["metrica", "r2_log"], ascending=[False, False])


def regressoes_memoria(dados: pd.DataFrame) -> pd.DataFrame:
    """Pico de memória × N em escala linear (y ≈ a·f(n) + b)."""
    linhas = []
    for coluna in ("pico_memoria_bytes", "pico_memoria_busca_bytes"):
        if coluna not in dados:
            continue
        com_valor = dados[dados[coluna].notna()]
        if len(com_valor) < 3:
            continue
        n = com_valor["n"].to_numpy(dtype=float)
        y = com_valor[coluna].to_numpy(dtype=float)
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        for nome, f in MODELOS_MEMORIA.items():
            a, b = np.polyfit(f(n), y, 1)
            residuos = y - (a * f(n) + b)
            linhas.append(
                {
                    "metrica": coluna,
                    "modelo": nome,
                    "a": a,
                    "b": b,
                    "r2": 1.0 - float(np.sum(residuos**2)) / ss_tot,
                    "rmse": float(np.sqrt(np.mean(residuos**2))),
                }
            )
    tabela = pd.DataFrame(linhas)
    return tabela.sort_values(["metrica", "r2"], ascending=[True, False])


def _prever(linha: pd.Series, n: np.ndarray) -> np.ndarray:
    modelo = linha["modelo"]
    if modelo in MODELOS_2P:
        expoente = (
            math.log10(linha["parametro"]) if modelo == "a·c^n" else linha["parametro"]
        )
        return linha["a"] * 10 ** (expoente * MODELOS_2P[modelo](n))
    return linha["a"] * 10 ** MODELOS_1P[modelo](n)


#  figuras
def figura_metrica(
    aleatorias: pd.DataFrame,
    tsplib: pd.DataFrame,
    regressao: pd.DataFrame,
    metrica: str,
    pasta: Path,
) -> Path:
    import matplotlib.pyplot as plt

    figura, eixo = plt.subplots(figsize=(7, 4.5), layout="constrained")
    eixo.scatter(
        aleatorias["n"], aleatorias[metrica], color="#1f77b4", alpha=0.4, s=22,
        label="aleatórias (5 sementes por N)",
    )  # fmt: skip
    medianas = aleatorias.groupby("n")[metrica].median()
    eixo.scatter(
        medianas.index, medianas, color="#d62728", s=40, label="mediana", zorder=4
    )
    if not tsplib.empty:
        eixo.scatter(
            tsplib["n"], tsplib[metrica], marker="s", facecolor="none",
            edgecolor="black", s=40, label="TSPLIB (Christofides)", zorder=5,
        )  # fmt: skip
    n_suave = np.linspace(aleatorias["n"].min(), aleatorias["n"].max(), 200)
    estilos = ["--", ":"]
    for estilo, (_, linha) in zip(
        estilos,
        regressao[regressao["metrica"] == metrica].head(2).iterrows(),
        strict=False,
    ):
        eixo.plot(
            n_suave, _prever(linha, n_suave), estilo, color="#222222",
            label=f"{linha['modelo']} ($R^2_{{log}}={linha['r2_log']:.3f}$)",
        )  # fmt: skip
    eixo.set_yscale("log")
    eixo.set(xlabel="N", ylabel=METRICAS[metrica], title=f"B&B — {METRICAS[metrica]}")
    eixo.grid(True, alpha=0.3, which="both")
    eixo.legend(fontsize=8)
    nome = "bnb_tempo.png" if metrica == "tempo_bnb_s" else "bnb_nos.png"
    return salvar_figura(figura, nome, pasta)


def figura_memoria(
    aleatorias: pd.DataFrame, regressao: pd.DataFrame, pasta: Path
) -> Path:
    import matplotlib.pyplot as plt

    figura, eixo = plt.subplots(figsize=(7, 4.5), layout="constrained")
    medidas = aleatorias[aleatorias["pico_memoria_bytes"].notna()]
    eixo.scatter(medidas["n"], medidas["pico_memoria_bytes"] / 1024, alpha=0.5, s=22)
    melhor = regressao[regressao["metrica"] == "pico_memoria_bytes"].iloc[0]
    n = np.linspace(medidas["n"].min(), medidas["n"].max(), 200)
    y = melhor["a"] * MODELOS_MEMORIA[melhor["modelo"]](n) + melhor["b"]
    eixo.plot(n, y / 1024, "--", color="#222222",
              label=f"{melhor['modelo']} ($R^2={melhor['r2']:.3f}$)")  # fmt: skip
    eixo.set(xlabel="N", ylabel="pico de memória (KiB)", title="B&B — pico de memória")
    eixo.grid(True, alpha=0.3)
    eixo.legend()
    return salvar_figura(figura, "bnb_memoria.png", pasta)


#  main
def main(
    pasta_resultados: Path | None = None,
    pasta_tabelas: Path = TABELAS,
    pasta_figuras: Path = FIGURAS,
) -> dict[str, pd.DataFrame]:
    aleatorias = ler_resultado(ENTRADA_ALEATORIAS, pasta_resultados)
    aleatorias = aleatorias[aleatorias["status"] == "OK"]
    tsplib = pd.concat(
        [ler_resultado(nome, pasta_resultados) for nome in ENTRADAS_TSPLIB],
        ignore_index=True,
    )
    tsplib = tsplib[
        (tsplib["estrategia"] == ESTRATEGIA_TSPLIB) & (tsplib["status"] == "OK")
    ]
    regressao = regressoes(aleatorias)
    memoria = regressoes_memoria(aleatorias)
    salvar_tabela(regressao, SAIDA, pasta_tabelas)
    salvar_tabela(memoria, SAIDA_MEMORIA, pasta_tabelas)

    destino = pasta_figuras / PASTA_FIGURAS
    for metrica in METRICAS:
        figura_metrica(aleatorias, tsplib, regressao, metrica, destino)
    figura_memoria(aleatorias, memoria, destino)

    print("=== B&B: ajuste em log10 (todas as repetições) ===")
    print(regressao.round(4).to_string(index=False))
    print("\n=== B&B: pico de memória ===")
    print(memoria.round(4).to_string(index=False))
    return {SAIDA: regressao, SAIDA_MEMORIA: memoria}


if __name__ == "__main__":
    main()
