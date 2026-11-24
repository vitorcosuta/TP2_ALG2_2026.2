

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analises.comum import FIGURAS, TABELAS, ler_resultado, salvar_figura, salvar_tabela

ENTRADAS_BNB = ["bnb_pequenas.csv", "bnb_grandes.csv"]
ENTRADA_OTIMOS = "otimos_bnb.csv"
PASTA_FIGURAS = "e2"

# Do limite mais fraco ao mais forte (ordem das colunas e das legendas).
ESTRATEGIAS = [
    "trivial",
    "vizinho_mais_proximo",
    "twice_around_the_tree",
    "christofides",
    "insercao_fecho_convexo",
]
REFERENCIA = "trivial"  # "B&B a frio" no enunciado


# ---------------------------------------------------------------- tabela
def montar_tabela(bnb: pd.DataFrame, otimos: pd.DataFrame) -> pd.DataFrame:
    """Tabela central: uma linha por (instância, estratégia)."""
    otimo = otimos.set_index("instancia")["melhor_custo"]
    t = bnb[bnb["estrategia"].isin(ESTRATEGIAS)].copy()
    t["otimo"] = t["instancia"].map(otimo)
    t["truncado"] = t["status"] != "OK"
    t["gap_limite_inicial"] = t["custo_inicial"] / t["otimo"] - 1.0
    t["gap_limite_raiz"] = 1.0 - t["limite_raiz"] / t["otimo"]
    t["tempo_total_s"] = t["tempo_heuristica_s"] + t["tempo_bnb_s"]

    ref = t[t["estrategia"] == REFERENCIA].set_index("instancia")
    t["nos_rel_trivial"] = t["nos_expandidos"] / t["instancia"].map(
        ref["nos_expandidos"]
    )
    # Aceleração > 1 = o warm-start compensou (heurística incluída).
    t["aceleracao_vs_trivial"] = (
        t["instancia"].map(ref["tempo_total_s"]) / t["tempo_total_s"]
    )
    t["ref_truncado"] = t["instancia"].map(ref["truncado"])

    colunas = [
        "instancia", "n", "estrategia", "otimo", "custo_inicial",
        "gap_limite_inicial", "limite_raiz", "gap_limite_raiz",
        "tempo_heuristica_s", "tempo_bnb_s", "tempo_total_s",
        "nos_expandidos", "nos_podados", "profundidade_maxima", "melhorias",
        "status", "truncado", "ref_truncado", "nos_rel_trivial",
        "aceleracao_vs_trivial",
    ]  # fmt: skip
    ordem = {e: i for i, e in enumerate(ESTRATEGIAS)}
    t["_o"] = t["estrategia"].map(ordem)
    return t.sort_values(["n", "instancia", "_o"])[colunas].reset_index(drop=True)


# ----------------------------------------------------------- pergunta 1
def _postos_esforco(grupo: pd.DataFrame) -> np.ndarray:
    """Postos dos nós expandidos, com as execuções NA no pior posto."""
    esforco = grupo["nos_expandidos"].to_numpy(dtype=float).copy()
    esforco[grupo["truncado"].to_numpy()] = np.inf
    return rankdata(esforco)


def correlacoes(tabela: pd.DataFrame) -> pd.DataFrame:
    """Spearman(gap do limite inicial, nós expandidos) por instância e geral.

    A linha ``(todas)`` usa os nós relativos ao trivial (``nos_rel_trivial``),
    que descontam o tamanho da instância.
    """
    linhas: list[dict[str, Any]] = []
    for inst, g in tabela.groupby("instancia", sort=False):
        postos = _postos_esforco(g)
        rho = spearmanr(g["gap_limite_inicial"], postos)[0]
        linhas.append(
            {
                "instancia": inst,
                "n": int(g["n"].iloc[0]),
                "execucoes": len(g),
                "spearman": float(rho) if np.isfinite(rho) else np.nan,
                "reducao_max_nos": float(1.0 - g["nos_rel_trivial"].min()),
                "melhor_estrategia": g.loc[g["nos_expandidos"].idxmin(), "estrategia"],
            }
        )
    validas = tabela[~tabela["truncado"] & ~tabela["ref_truncado"]]
    rho, p = spearmanr(validas["gap_limite_inicial"], validas["nos_rel_trivial"])
    linhas.append(
        {
            "instancia": "(todas)",
            "n": np.nan,
            "execucoes": len(validas),
            "spearman": float(rho),
            "p_valor": float(p),
            "reducao_max_nos": np.nan,
            "melhor_estrategia": "",
        }
    )
    return pd.DataFrame(linhas)


# ----------------------------------------------------------- pergunta 2
def tradeoff(tabela: pd.DataFrame) -> pd.DataFrame:
    """Aceleração do tempo total por estratégia (resumo sobre as instâncias)."""
    linhas = []
    for est in ESTRATEGIAS:
        if est == REFERENCIA:
            continue
        g = tabela[tabela["estrategia"] == est]
        acel = g["aceleracao_vs_trivial"]
        linhas.append(
            {
                "estrategia": est,
                "instancias": len(g),
                "aceleracao_mediana": float(acel.median()),
                "aceleracao_min": float(acel.min()),
                "aceleracao_max": float(acel.max()),
                # Abaixo de 1 = não compensou; tolera ±10% de ruído de medição.
                "nao_compensou": ", ".join(g.loc[acel < 0.9, "instancia"]),
                "fracao_tempo_heuristica_max": float(
                    (g["tempo_heuristica_s"] / g["tempo_total_s"]).max()
                ),
            }
        )
    return pd.DataFrame(linhas)


# ----------------------------------------------------------- pergunta 3
def colapso(tabela: pd.DataFrame) -> pd.DataFrame:
    """Por instância: quantas estratégias estouraram o tempo e o gap na raiz."""
    linhas = []
    for inst, g in tabela.groupby("instancia", sort=False):
        linhas.append(
            {
                "instancia": inst,
                "n": int(g["n"].iloc[0]),
                "gap_limite_raiz": float(g["gap_limite_raiz"].iloc[0]),
                "estrategias_na": int(g["truncado"].sum()),
                "quais_na": ", ".join(g.loc[g["truncado"], "estrategia"]),
                "tempo_bnb_max_s": float(g["tempo_bnb_s"].max()),
                "nos_max": int(g["nos_expandidos"].max()),
            }
        )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------- figuras
def figura_gap_vs_nos(tabela: pd.DataFrame, pasta: Path) -> Path:
    """Nós relativos ao trivial × gap do limite inicial, uma linha por instância."""
    import matplotlib.pyplot as plt

    figura, eixo = plt.subplots(figsize=(8, 5), layout="constrained")
    cores = plt.colormaps["viridis"]
    ns = tabela["n"]
    for inst, g in tabela.groupby("instancia", sort=False):
        g = g.sort_values("gap_limite_inicial")
        cor = cores((g["n"].iloc[0] - ns.min()) / max(1, ns.max() - ns.min()))
        eixo.plot(
            100 * g["gap_limite_inicial"],
            g["nos_rel_trivial"],
            "-",
            color=cor,
            alpha=0.6,
        )
        ok, na = g[~g["truncado"]], g[g["truncado"]]
        eixo.scatter(
            100 * ok["gap_limite_inicial"], ok["nos_rel_trivial"], color=cor, s=25
        )
        eixo.scatter(
            100 * na["gap_limite_inicial"], na["nos_rel_trivial"], color=cor,
            marker="^", s=60, edgecolor="black", label="_",
        )  # fmt: skip
        # Rótulo no ponto de menor árvore (à direita todos se acumulam em 1).
        melhor = g.loc[g["nos_rel_trivial"].idxmin()]
        eixo.annotate(
            inst, (100 * melhor["gap_limite_inicial"], melhor["nos_rel_trivial"]),
            fontsize=7, xytext=(3, 0), textcoords="offset points",
        )  # fmt: skip
    eixo.set_xscale("symlog", linthresh=1)
    eixo.set_yscale("log")
    eixo.axhline(1, color="gray", linestyle=":", linewidth=1)
    eixo.set(
        xlabel="gap do limite superior inicial (%)",
        ylabel="nós expandidos / nós do trivial",
        title="Warm-start × tamanho da árvore (▲ = tempo esgotado, valor truncado)",
    )
    eixo.grid(True, alpha=0.3, which="both")
    mapa = plt.cm.ScalarMappable(
        cmap=cores, norm=plt.Normalize(vmin=ns.min(), vmax=ns.max())
    )
    figura.colorbar(mapa, ax=eixo, label="N")
    return salvar_figura(figura, "gap_vs_nos.png", pasta)


def figura_mapa_nos(tabela: pd.DataFrame, pasta: Path) -> Path:
    """Mapa instância × estratégia com os nós relativos ao trivial."""
    import matplotlib.pyplot as plt

    m = tabela.pivot(index="instancia", columns="estrategia", values="nos_rel_trivial")
    trunc = tabela.pivot(index="instancia", columns="estrategia", values="truncado")
    linhas = tabela.drop_duplicates("instancia")["instancia"].tolist()
    m, trunc = (
        m.reindex(index=linhas, columns=ESTRATEGIAS),
        trunc.reindex(index=linhas, columns=ESTRATEGIAS),
    )
    valores = np.log10(m.to_numpy(dtype=float))
    figura, eixo = plt.subplots(figsize=(8, 6), layout="constrained")
    imagem = eixo.imshow(valores, cmap="RdYlGn_r", vmin=-2.5, vmax=0.5, aspect="auto")
    for (i, j), v in np.ndenumerate(m.to_numpy(dtype=float)):
        texto = f"{v:.2f}" if v >= 0.01 else f"{v:.1e}"
        if bool(trunc.iloc[i, j]):
            texto = "≥" + texto
        eixo.text(j, i, texto, ha="center", va="center", fontsize=8)
    eixo.set_xticks(range(len(ESTRATEGIAS)), ESTRATEGIAS, rotation=25, ha="right")
    eixo.set_yticks(range(len(linhas)), linhas)
    eixo.set_title("Nós expandidos relativos ao trivial (≥ = tempo esgotado)")
    figura.colorbar(imagem, ax=eixo, label="log10(nós / nós do trivial)")
    return salvar_figura(figura, "mapa_nos.png", pasta)


def figura_tempos(tabela: pd.DataFrame, pasta: Path) -> Path:
    """Tempo total por estratégia nas 3 maiores instâncias (escala log)."""
    import matplotlib.pyplot as plt

    maiores = tabela.drop_duplicates("instancia").nlargest(3, "n")["instancia"]
    figura, eixos = plt.subplots(
        1, len(maiores), figsize=(4.2 * len(maiores), 4), layout="constrained"
    )
    for eixo, inst in zip(np.atleast_1d(eixos), maiores, strict=True):
        g = tabela[tabela["instancia"] == inst].set_index("estrategia").loc[ESTRATEGIAS]
        cores = ["#d62728" if t else "#1f77b4" for t in g["truncado"]]
        eixo.bar(range(len(g)), g["tempo_total_s"], color=cores)
        eixo.set_yscale("log")
        eixo.set_ylim(1, 5000)  # mesma escala nos três painéis
        eixo.axhline(1800, color="gray", linestyle="--", linewidth=1)
        eixo.set_xticks(range(len(g)), ESTRATEGIAS, rotation=35, ha="right", fontsize=8)
        eixo.set(title=f"{inst} (N={int(g['n'].iloc[0])})", ylabel="tempo total (s)")
        eixo.grid(True, axis="y", alpha=0.3)
    figura.suptitle("Heurística + B&B (vermelho = estourou 30 min)")
    return salvar_figura(figura, "tempos_maiores.png", pasta)


# ------------------------------------------------------------------ main
def main(
    pasta_resultados: Path | None = None,
    pasta_tabelas: Path = TABELAS,
    pasta_figuras: Path = FIGURAS,
) -> dict[str, pd.DataFrame]:
    bnb = pd.concat(
        [ler_resultado(nome, pasta_resultados) for nome in ENTRADAS_BNB],
        ignore_index=True,
    )
    tabela = montar_tabela(bnb, ler_resultado(ENTRADA_OTIMOS, pasta_resultados))
    saidas = {
        "e2_tabela.csv": tabela,
        "e2_correlacoes.csv": correlacoes(tabela),
        "e2_tradeoff.csv": tradeoff(tabela),
        "e2_colapso.csv": colapso(tabela),
    }
    for nome, t in saidas.items():
        salvar_tabela(t, nome, pasta_tabelas)
    destino = pasta_figuras / PASTA_FIGURAS
    figura_gap_vs_nos(tabela, destino)
    figura_mapa_nos(tabela, destino)
    figura_tempos(tabela, destino)
    _imprimir_resumo(saidas)
    return saidas


def _imprimir_resumo(saidas: dict[str, pd.DataFrame]) -> None:
    tabela = saidas["e2_tabela.csv"]
    print("=== Nós expandidos (instância × estratégia; * = tempo esgotado) ===")
    marcado = tabela.assign(
        v=tabela["nos_expandidos"].astype(str) + np.where(tabela["truncado"], "*", "")
    )
    print(marcado.pivot(index="instancia", columns="estrategia", values="v")
          .reindex(columns=ESTRATEGIAS).to_string())  # fmt: skip
    print("\n=== Pergunta 1: Spearman(gap do limite, nós) ===")
    print(saidas["e2_correlacoes.csv"].round(3).to_string(index=False))
    print("\n=== Pergunta 2: aceleração do tempo total vs trivial ===")
    print(saidas["e2_tradeoff.csv"].round(3).to_string(index=False))
    print("\n=== Pergunta 3: colapso ===")
    print(saidas["e2_colapso.csv"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
