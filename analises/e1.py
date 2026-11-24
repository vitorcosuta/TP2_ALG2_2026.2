"""E1: complexidade empírica dos algoritmos construtivos (tempo e memória).

Cobre as 3 heurísticas e os 2 aproximados medidos em instâncias aleatórias.

Lê ``resultados/subrotinas_aleatorias_e1.csv`` (gerado por
``experimentos.aproximados_aleatorios``) e grava:

- ``relatorio/tabelas/regressao_e1_aproximados.csv``: para cada (algoritmo,
  etapa, métrica), o ajuste de cada modelo candidato com R² e RMSE;
- ``relatorio/figuras/e1/<algoritmo>_<etapa>.png``: medição e melhor ajuste.

Uso (a partir da raiz do repositório)::

    python -m analises.e1
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from analises.comum import FIGURAS, TABELAS, ler_resultado, salvar_figura, salvar_tabela

ENTRADA = "subrotinas_aleatorias_e1.csv"
ENTRADA_BRUTOS = "tempos_brutos_e1.csv"
SAIDA_REGRESSAO = "regressao_e1_aproximados.csv"
PASTA_FIGURAS = "e1"

# Modelos candidatos: tempo ≈ a · f(x) + b.
MODELOS: dict[str, Callable[[np.ndarray, float, float], np.ndarray]] = {
    "O(x)": lambda x, a, b: a * x + b,
    "O(x log x)": lambda x, a, b: a * x * np.log2(x) + b,
    "O(x^2)": lambda x, a, b: a * (x**2) + b,
    "O(x^3)": lambda x, a, b: a * (x**3) + b,
}

# Etapas com gráfico no relatório: (algoritmo, etapa, variável do eixo x).
ALVOS_GRAFICOS = [
    ("vizinho_mais_proximo", "total", "n"),
    ("insercao_fecho_convexo", "total", "n"),
    ("twice_around_the_tree", "total", "n"),
    ("twice_around_the_tree", "total/agm", "n"),
    ("christofides", "total", "n"),
    ("christofides", "total/emparelhamento", "k_impares"),
]


def variavel_x(etapa: str) -> str:
    """O Blossom (emparelhamento) é O(k³): ajusta-se contra k, não contra N."""
    return "k_impares" if etapa == "total/emparelhamento" else "n"


def ajustar_modelos(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Ajusta todos os modelos candidatos e calcula R² e RMSE."""
    resultados = []
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    for nome_modelo, func in MODELOS.items():
        with warnings.catch_warnings():
            # A covariância não é usada; com ajuste exato ela não é estimável.
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(func, x, y, maxfev=10_000)
        residuos = y - func(x, *popt)
        ss_res = float(np.sum(residuos**2))
        resultados.append(
            {
                "modelo": nome_modelo,
                "a": float(popt[0]),
                "b": float(popt[1]),
                "r2": round(1.0 - ss_res / ss_tot, 6) if ss_tot > 0 else 1.0,
                "rmse": float(np.sqrt(np.mean(residuos**2))),
            }
        )
    return resultados


def regressoes(dados: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (algoritmo, etapa, métrica, modelo)."""
    linhas: list[dict[str, Any]] = []
    for (alg, etapa), grupo in dados.groupby(["algoritmo", "etapa"]):
        var_x = variavel_x(str(etapa))
        grupo_ord = grupo.sort_values(var_x)
        x = grupo_ord[var_x].to_numpy(dtype=float)
        for metrica in ("tempo_mediana_s", "pico_memoria_bytes"):
            y = grupo_ord[metrica].to_numpy(dtype=float)
            for res in ajustar_modelos(x, y):
                linhas.append(
                    {
                        "algoritmo": alg,
                        "etapa": etapa,
                        "variavel_x": var_x,
                        "metrica": metrica,
                        **res,
                    }
                )
    return pd.DataFrame(linhas)


def melhores_ajustes(regressao: pd.DataFrame, metrica: str) -> pd.DataFrame:
    """O modelo de maior R² para cada (algoritmo, etapa)."""
    sub = regressao[regressao["metrica"] == metrica]
    indices = sub.groupby(["algoritmo", "etapa"])["r2"].idxmax()
    return sub.loc[indices].sort_values(["algoritmo", "etapa"])


def gerar_graficos(
    dados: pd.DataFrame,
    regressao: pd.DataFrame,
    pasta: Path,
    brutos: pd.DataFrame | None = None,
) -> list[Path]:
    """Repetições individuais (azul), mediana (vermelho) e curva do melhor modelo."""
    import matplotlib.pyplot as plt

    melhores = melhores_ajustes(regressao, "tempo_mediana_s").set_index(
        ["algoritmo", "etapa"]
    )
    caminhos = []
    for alg, etapa, var_x in ALVOS_GRAFICOS:
        sub = dados[(dados["algoritmo"] == alg) & (dados["etapa"] == etapa)]
        sub = sub.sort_values(var_x)
        if sub.empty:
            continue
        x = sub[var_x].to_numpy(dtype=float)
        y = sub["tempo_mediana_s"].to_numpy(dtype=float)
        melhor = melhores.loc[(alg, etapa)]
        x_suave = np.linspace(x.min(), x.max(), 200)
        y_suave = MODELOS[melhor["modelo"]](x_suave, melhor["a"], melhor["b"])

        figura, eixo = plt.subplots(figsize=(7, 4.5))
        if brutos is not None:
            sub_brutos = brutos[
                (brutos["algoritmo"] == alg) & (brutos["etapa"] == etapa)
            ]
            eixo.scatter(
                sub_brutos[var_x],
                sub_brutos["tempo_s"],
                label="Repetições individuais",
                color="#1f77b4",
                alpha=0.35,
                s=22,
                zorder=2,
            )
        eixo.scatter(
            x,
            y,
            label="Mediana",
            color="#d62728",
            s=40,
            zorder=4,
        )
        eixo.plot(
            x_suave,
            y_suave,
            label=f"Ajuste {melhor['modelo']} ($R^2={melhor['r2']:.4f}$)",
            color="#222222",
            linestyle="--",
            linewidth=1.8,
            zorder=3,
        )
        eixo.set(title=f"{alg} — {etapa}", xlabel=var_x, ylabel="Tempo (s)")
        eixo.legend()
        eixo.grid(True, alpha=0.3)
        nome = f"{alg}_{etapa.replace('/', '_')}.png"
        caminhos.append(salvar_figura(figura, nome, pasta))
    return caminhos


def main(
    pasta_resultados: Path | None = None,
    pasta_tabelas: Path = TABELAS,
    pasta_figuras: Path = FIGURAS,
) -> pd.DataFrame:
    dados = ler_resultado(ENTRADA, pasta_resultados)
    brutos = ler_resultado(ENTRADA_BRUTOS, pasta_resultados)
    regressao = regressoes(dados)
    destino = salvar_tabela(regressao, SAIDA_REGRESSAO, pasta_tabelas)
    figuras = gerar_graficos(dados, regressao, pasta_figuras / PASTA_FIGURAS, brutos=brutos)

    print("=== Melhor ajuste de TEMPO por etapa (maior R²) ===")
    colunas = ["algoritmo", "etapa", "variavel_x", "modelo", "r2", "rmse"]
    print(
        melhores_ajustes(regressao, "tempo_mediana_s")[colunas].to_string(index=False)
    )
    print(f"\nTabela: {destino}\nFiguras: {len(figuras)} em {pasta_figuras}")
    return regressao


if __name__ == "__main__":
    # O E1 tem três partes; cada uma fica no seu módulo para evitar conflito
    from analises import e1_bnb, e1_perfil

    main()  # aproximados
    e1_bnb.main()  # B&B
    e1_perfil.main()  # etapa GC
