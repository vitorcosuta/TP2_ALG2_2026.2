"""Complexidade empírica de Christofides e Twice-around-the-tree."""
from __future__ import annotations
import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from tsp.agm import agm_prim
from tsp.christofides import christofides, vertices_grau_impar
from tsp.medicao import medir
from tsp.twice_around_the_tree import twice_around_the_tree

PASTA_SAIDA = Path("resultados")
TAMANHOS = [10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 1000]

ALGORITMOS = {
    "twice_around_the_tree": twice_around_the_tree,
    "christofides": christofides,
}

# Modelos candidatos para regressão
MODELOS: dict[str, Callable[[np.ndarray, float, float], np.ndarray]] = {
    "O(x)": lambda x, a, b: a * x + b,
    "O(x log x)": lambda x, a, b: a * x * np.log2(x) + b,
    "O(x^2)": lambda x, a, b: a * (x**2) + b,
    "O(x^3)": lambda x, a, b: a * (x**3) + b,
}


def _instancia_aleatoria(n: int, seed: int) -> np.ndarray:
    """Matriz de custos métrica para n pontos uniformes em [0, 1000]²."""
    rng = np.random.default_rng(seed)
    pontos = rng.uniform(0.0, 1000.0, size=(n, 2))
    dif = pontos[:, None, :] - pontos[None, :, :]
    dist = np.rint(np.hypot(dif[:, :, 0], dif[:, :, 1])).astype(np.int64)
    np.fill_diagonal(dist, 0)
    return dist


def _salvar_csv(caminho: Path, linhas: list[dict[str, Any]]) -> None:
    
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)


def coletar_dados_aleatorios() -> Path:
    """Executa ambos os algoritmos e salva o CSV de sub-rotinas."""
    subrotinas: list[dict[str, Any]] = []

    for i, n in enumerate(TAMANHOS):
        custos = _instancia_aleatoria(n, seed = i)
        pai = agm_prim(custos)
        k_impares = len(vertices_grau_impar(pai))
        reps = 15

        for nome_alg, func in ALGORITMOS.items():
            med = medir(lambda m, f=func: f(custos, medidor=m),repeticoes=reps,medir_memoria=True,desativar_gc=False,)
            for linha in med.linhas():
                subrotinas.append( {"algoritmo": nome_alg,"instancia": f"rand_{n}","n": n,"k_impares": k_impares,**linha,} )
        print(f"Coletado: N={n:3d} | k_impares={k_impares:3d} | reps={reps}")

    caminho_csv = PASTA_SAIDA / "subrotinas_aleatorias_e1.csv"
    _salvar_csv(caminho_csv, subrotinas)
    return caminho_csv


def _ajustar_modelos(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    """Ajusta todos os modelos candidatos e calcula R² e RMSE."""
    resultados = []
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    for nome_modelo, func in MODELOS.items():
        popt, _ = curve_fit(func, x, y, maxfev=10_000)
        y_pred = func(x, *popt)
        residuos = y - y_pred
        ss_res = float(np.sum(residuos**2))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
        rmse = float(np.sqrt(np.mean(residuos**2)))
        resultados.append({"modelo": nome_modelo,"a": float(popt[0]),"b": float(popt[1]),"r2": round(r2, 6),"rmse": rmse,})
    return resultados


def ajustar_e_salvar_regressoes(caminho_csv: Path) -> pd.DataFrame:
    """Roda a regressão para tempo e memória de cada tupla (algoritmo, etapa)."""
    df = pd.read_csv(caminho_csv)
    linhas_reg: list[dict[str, Any]] = []

    for (alg, etapa), grupo in df.groupby(["algoritmo", "etapa"]):
        # Blossom (emparelhamento) é ajustado contra k; demais etapas contra N
        variavel_x = "k_impares" if etapa == "total/emparelhamento" else "n"
        grupo_ord = grupo.sort_values(variavel_x)
        x = grupo_ord[variavel_x].to_numpy(dtype=float)

        for metrica in ("tempo_mediana_s", "pico_memoria_bytes"):
            y = grupo_ord[metrica].to_numpy(dtype=float)
            for res in _ajustar_modelos(x, y):
                linhas_reg.append({"algoritmo": alg,"etapa": etapa,"variavel_x": variavel_x,"metrica": metrica,**res,})

    df_reg = pd.DataFrame(linhas_reg)
    saida_reg = PASTA_SAIDA / "regressao_e1_aproximados.csv"
    df_reg.to_csv(saida_reg, index=False)

    # Mostra o melhor modelo (maior R²) para tempo de cada etapa
    print("\n=== Melhor ajuste de TEMPO por etapa (maior R²) ===")
    tempo_df = df_reg[df_reg["metrica"] == "tempo_mediana_s"]
    melhores = tempo_df.loc[tempo_df.groupby(["algoritmo", "etapa"])["r2"].idxmax()]
    print(melhores[["algoritmo", "etapa", "variavel_x", "modelo", "r2", "rmse"]].sort_values(["algoritmo", "etapa"]).to_string(index=False))
    return df_reg


def gerar_graficos(caminho_csv: Path, df_reg: pd.DataFrame) -> None:
    """Gera gráficos das etapas principais (Prim vs n², Blossom vs k³, totais)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib não instalado; gráficos ignorados.")
        return

    pasta_graficos = PASTA_SAIDA / "graficos_e1"
    pasta_graficos.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(caminho_csv)

    alvos = [("twice_around_the_tree", "total", "n"),("twice_around_the_tree", "total/agm", "n"),("christofides", "total", "n"),("christofides", "total/emparelhamento", "k_impares"),]

    for alg, etapa, var_x in alvos:
        sub = df[(df["algoritmo"] == alg) & (df["etapa"] == etapa)].sort_values(var_x)
        x = sub[var_x].to_numpy(dtype=float)
        y = sub["tempo_mediana_s"].to_numpy(dtype=float)

        reg_sub = df_reg[(df_reg["algoritmo"] == alg)& (df_reg["etapa"] == etapa)& (df_reg["metrica"] == "tempo_mediana_s")]
        melhor = reg_sub.loc[reg_sub["r2"].idxmax()]
        x_suave = np.linspace(x.min(), x.max(), 200)
        y_suave = MODELOS[melhor["modelo"]](x_suave, melhor["a"], melhor["b"])

        plt.figure(figsize=(7, 4.5))
        plt.scatter(x, y, label="Medido (mediana)", color="#1f77b4", zorder=3)
        plt.plot(x_suave,y_suave,label=f"Ajuste {melhor['modelo']} ($R^2={melhor['r2']:.4f}$)",color="#d62728",linewidth=2,)
        plt.title(f"{alg} — {etapa}")
        plt.xlabel(var_x)
        plt.ylabel("Tempo (s)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        nome_arq = f"{alg}_{etapa.replace('/', '_')}.png"
        plt.savefig(pasta_graficos / nome_arq, dpi=150)
        plt.close()

    print(f"\nGráficos salvos em: {pasta_graficos.resolve()}")


if __name__ == "__main__":
    csv_dados = coletar_dados_aleatorios()
    df_regressao = ajustar_e_salvar_regressoes(csv_dados)
    gerar_graficos(csv_dados, df_regressao)