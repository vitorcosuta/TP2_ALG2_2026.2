
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analises.comum import FIGURAS, TABELAS, ler_resultado, salvar_figura, salvar_tabela

ENTRADA_COM_GC = "subrotinas.csv"
ENTRADA_SEM_GC = "subrotinas_sem_gc.csv"
ENTRADA_ALEATORIAS = "subrotinas_aleatorias_e1.csv"
ENTRADAS_BNB = ["bnb_pequenas.csv", "bnb_grandes.csv"]
PASTA_FIGURAS = "e1"

ALGORITMOS = [
    "vizinho_mais_proximo",
    "insercao_fecho_convexo",
    "twice_around_the_tree",
    "christofides",
]


#  etapas
def fracoes_por_instancia(sub: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (algoritmo, instância, etapa) com a fração do total."""
    total = sub[sub["etapa"] == "total"].set_index(["algoritmo", "instancia"])[
        "tempo_mediana_s"
    ]
    filhas = sub[sub["etapa"].str.count("/") == 1].copy()
    filhas["fracao"] = (
        filhas["tempo_mediana_s"].to_numpy()
        / total.reindex(
            pd.MultiIndex.from_frame(filhas[["algoritmo", "instancia"]])
        ).to_numpy()
    )
    return filhas[["algoritmo", "instancia", "n", "etapa", "fracao"]]


def perfil_etapas(sub: pd.DataFrame, origem: str) -> pd.DataFrame:
    """Resumo das frações por (algoritmo, etapa), da etapa mais cara à mais barata."""
    fr = fracoes_por_instancia(sub[sub["algoritmo"].isin(ALGORITMOS)])
    # Etapa ausente numa instância (ex.: mds só existe sem coordenadas) conta
    # como 0, para que as medianas das etapas de um algoritmo sejam comparáveis
    # e possam ser empilhadas.
    largo = fr.pivot_table(
        index=["algoritmo", "instancia"], columns="etapa", values="fracao"
    )
    linhas = []
    for alg, g in largo.groupby(level="algoritmo"):
        for etapa in g.columns[g.notna().any()]:
            col = g[etapa]
            linhas.append(
                {
                    "algoritmo": alg,
                    "etapa": etapa,
                    "fracao_mediana": float(col.fillna(0).median()),
                    "fracao_min": float(col.min()),
                    "fracao_max": float(col.max()),
                    "instancias_com_etapa": int(col.notna().sum()),
                    "fracao_mediana_quando_existe": float(col.median()),
                }
            )
    resumo = pd.DataFrame(linhas)
    resumo.insert(0, "instancias_origem", origem)
    ordem = {a: i for i, a in enumerate(ALGORITMOS)}
    resumo["_o"] = resumo["algoritmo"].map(ordem)
    resumo = resumo.sort_values(["_o", "fracao_mediana"], ascending=[True, False])
    return resumo.drop(columns="_o").reset_index(drop=True)


def perfil_bnb(bnb: pd.DataFrame) -> pd.DataFrame:
    """Frações do tempo do B&B por instância (estratégia Christofides)."""
    b = bnb[bnb["estrategia"] == "christofides"].copy()
    b["fracao_penalidades"] = b["tempo_penalidades_s"] / b["tempo_bnb_s"]
    b["fracao_busca"] = b["tempo_busca_s"] / b["tempo_bnb_s"]
    b["fracao_limite_na_busca"] = b["tempo_limite_inferior_s"] / b["tempo_busca_s"]
    b["us_por_no"] = 1e6 * b["tempo_busca_s"] / b["nos_expandidos"]
    colunas = [
        "instancia", "n", "status", "tempo_bnb_s", "fracao_penalidades",
        "fracao_busca", "fracao_limite_na_busca", "us_por_no",
    ]  # fmt: skip
    return b.sort_values("n")[colunas].reset_index(drop=True)


#  GC
def comparar_gc(com: pd.DataFrame, sem: pd.DataFrame) -> pd.DataFrame:
    """Razão sem/com GC do tempo total, por algoritmo, e coletas observadas."""
    chave = ["algoritmo", "instancia"]
    a = com[com["etapa"] == "total"]
    b = sem[sem["etapa"] == "total"]
    m = a.merge(b, on=chave, suffixes=("_com", "_sem"))
    m["razao"] = m["tempo_mediana_s_sem"] / m["tempo_mediana_s_com"]
    resumo = (
        m.groupby("algoritmo")
        .agg(
            instancias=("razao", "count"),
            razao_mediana=("razao", "median"),
            razao_min=("razao", "min"),
            razao_max=("razao", "max"),
            coletas_gc_com=("coletas_gc_mediana_com", "sum"),
            coletas_gc_sem=("coletas_gc_mediana_sem", "sum"),
        )
        .reset_index()
    )
    return resumo


#  figuras
def figura_etapas(perfil: pd.DataFrame, pasta: Path) -> Path:
    """Barras empilhadas com a fração mediana de cada etapa."""
    import matplotlib.pyplot as plt

    figura, eixo = plt.subplots(figsize=(8, 4.2), layout="constrained")
    cores = plt.colormaps["tab20"]
    etapas = sorted(perfil["etapa"].unique())
    algoritmos = [a for a in ALGORITMOS if a in set(perfil["algoritmo"])]
    base = np.zeros(len(algoritmos))
    for i, etapa in enumerate(etapas):
        valores = np.array(
            [
                perfil.loc[
                    (perfil["algoritmo"] == a) & (perfil["etapa"] == etapa),
                    "fracao_mediana",
                ].sum()
                for a in algoritmos
            ]
        )
        if not valores.any():
            continue
        eixo.barh(
            algoritmos, valores, left=base, color=cores(i), label=etapa.split("/")[1]
        )
        for j, v in enumerate(valores):
            if v > 0.08:
                eixo.text(base[j] + v / 2, j, f"{100 * v:.0f}%", ha="center", va="center",
                          fontsize=8)  # fmt: skip
        base += valores
    eixo.set(xlim=(0, 1), xlabel="fração mediana do tempo total")
    eixo.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    eixo.set_title("Etapa dominante (instâncias da TSPLIB)")
    return salvar_figura(figura, "perfil_etapas.png", pasta)


def figura_emparelhamento(aleatorias: pd.DataFrame, pasta: Path) -> Path:
    """Fração do Christofides gasta no emparelhamento × N (aleatórias)."""
    import matplotlib.pyplot as plt

    fr = fracoes_por_instancia(aleatorias[aleatorias["algoritmo"] == "christofides"])
    fr = fr[fr["etapa"] == "total/emparelhamento"]
    figura, eixo = plt.subplots(figsize=(7, 4), layout="constrained")
    eixo.scatter(fr["n"], 100 * fr["fracao"], alpha=0.5, s=22)
    mediana = fr.groupby("n")["fracao"].median()
    eixo.plot(mediana.index, 100 * mediana, "o-", color="#d62728", label="mediana")
    eixo.set(xlabel="N", ylabel="% do tempo do Christofides",
             title="Peso do emparelhamento (Blossom) no Christofides", ylim=(0, 100))  # fmt: skip
    eixo.set_xscale("log")
    eixo.grid(True, alpha=0.3)
    eixo.legend()
    return salvar_figura(figura, "perfil_emparelhamento.png", pasta)


def figura_gc(com: pd.DataFrame, sem: pd.DataFrame, pasta: Path) -> Path:
    import matplotlib.pyplot as plt

    a = com[com["etapa"] == "total"]
    b = sem[sem["etapa"] == "total"]
    m = a.merge(b, on=["algoritmo", "instancia", "n"], suffixes=("_com", "_sem"))
    figura, eixo = plt.subplots(figsize=(7, 4), layout="constrained")
    for alg in ALGORITMOS:
        g = m[m["algoritmo"] == alg]
        eixo.scatter(g["n"], g["tempo_mediana_s_sem"] / g["tempo_mediana_s_com"],
                     label=alg, s=25, alpha=0.8)  # fmt: skip
    eixo.axhline(1, color="gray", linestyle=":")
    eixo.set_yscale("log")
    eixo.set(xlabel="N", ylabel="tempo sem GC / tempo com GC",
             title="Efeito do gc.disable() (nenhuma coleta registrada)")  # fmt: skip
    eixo.grid(True, alpha=0.3, which="both")
    eixo.legend(fontsize=8)
    return salvar_figura(figura, "gc.png", pasta)


#  main
def main(
    pasta_resultados: Path | None = None,
    pasta_tabelas: Path = TABELAS,
    pasta_figuras: Path = FIGURAS,
) -> dict[str, pd.DataFrame]:
    com = ler_resultado(ENTRADA_COM_GC, pasta_resultados)
    sem = ler_resultado(ENTRADA_SEM_GC, pasta_resultados)
    aleatorias = ler_resultado(ENTRADA_ALEATORIAS, pasta_resultados)
    bnb = pd.concat(
        [ler_resultado(n, pasta_resultados) for n in ENTRADAS_BNB], ignore_index=True
    )
    etapas = pd.concat(
        [perfil_etapas(com, "tsplib"), perfil_etapas(aleatorias, "aleatorias")],
        ignore_index=True,
    )
    saidas = {
        "perfil_etapas.csv": etapas,
        "perfil_bnb.csv": perfil_bnb(bnb),
        "gc.csv": comparar_gc(com, sem),
    }
    for nome, t in saidas.items():
        salvar_tabela(t, nome, pasta_tabelas)
    destino = pasta_figuras / PASTA_FIGURAS
    figura_etapas(etapas[etapas["instancias_origem"] == "tsplib"], destino)
    figura_emparelhamento(aleatorias, destino)
    figura_gc(com, sem, destino)

    for nome, t in saidas.items():
        print(f"\n=== {nome} ===")
        print(t.round(3).to_string(index=False))
    return saidas


if __name__ == "__main__":
    main()
