"""E3: análise de sensibilidade — como as features da instância afetam o gap.

Lê ``features.csv``, ``execucoes.csv`` e ``otimos_bnb.csv`` e grava em
``relatorio/``:

- ``tabelas/e3_base.csv``: gap de cada algoritmo em cada instância, ao lado
  das features (os dados que entram nos modelos);
- ``tabelas/e3_correlacoes.csv``: Spearman (ρ e p-valor) de cada feature com o
  gap e a correlação **parcial controlando N**;
- ``tabelas/e3_modelos.csv``: modelos avaliados por leave-one-out (Q² e RMSE);
- ``tabelas/e3_pesos.csv``: coeficientes padronizados da regressão Ridge e
  importâncias da árvore de decisão;
- ``figuras/e3/*.png``: mapa de correlações, gap × features e a árvore.

Uso (a partir da raiz do repositório)::

    python -m analises.e3

Decisões (ver EXPERIMENTOS.md, E3):

- **Só instâncias com ótimo provado** (``otimo_provado`` em
  ``otimos_bnb.csv``). Nas outras o gap é só um intervalo.
- **O trivial fica de fora**: o gap dele depende da ordem dos vértices no
  arquivo, não da geometria da instância.
- **Correlação de Spearman**, não de Pearson: com ~13 pontos, uma instância
  extrema dominaria a correlação linear; postos são robustos a isso.
- **Correlação parcial controlando N.** Várias features variam com N (a fração
  de pontos no fecho convexo cai com N, por exemplo). Sem descontar N, uma
  feature pode parecer importante só porque acompanha o tamanho. A parcial
  correlaciona os resíduos dos postos da feature e do gap depois de regredir
  cada um contra o posto de N.
- **Validação leave-one-out (LOO) e Q².** Com ~13 instâncias não há dados para
  separar treino e teste. No LOO, cada instância é prevista por um modelo
  treinado com as outras; ``Q² = 1 − PRESS/SS``. Q² ≤ 0 significa que o
  modelo prevê pior que a média: as features não generalizam.
- **Poucos parâmetros por modelo.** A base de comparação é "só N"; cada modelo
  "N + feature" responde se a feature acrescenta informação além do tamanho.
  Os modelos com todas as features (Ridge, que encolhe os coeficientes, e uma
  árvore rasa) servem para ranquear pesos, não como modelo preditivo final.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.base import RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from analises.comum import FIGURAS, TABELAS, ler_resultado, salvar_figura, salvar_tabela

ENTRADA_FEATURES = "features.csv"
ENTRADA_EXECUCOES = "execucoes.csv"
ENTRADA_OTIMOS = "otimos_bnb.csv"
PASTA_FIGURAS = "e3"

# Aproximados primeiro (foco do E3); heurísticas são opcionais no plano.
ALGORITMOS = [
    "christofides",
    "twice_around_the_tree",
    "insercao_fecho_convexo",
    "vizinho_mais_proximo",
]

FEATURES = [
    "n",
    # da matriz de custos (todas as instâncias)
    "grau_max_agm",
    "fracao_folhas_agm",
    "fracao_impares_agm",
    "cv_arestas_agm",
    "cv_vizinho_mais_proximo",
    "cv_distancias",
    # das coordenadas (MDS nas instâncias sem pontos)
    "razao_aspecto",
    "fracao_fecho_convexo",
    "entropia_espacial",
    "clark_evans",
]
CONTROLE = "n"

ARVORE = DecisionTreeRegressor(max_depth=2, min_samples_leaf=3, random_state=0)
RIDGE = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))


# ---------------------------------------------------------------- dados
def montar_base(
    execucoes: pd.DataFrame, otimos: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    """Uma linha por (algoritmo, instância com ótimo provado), com gap e features."""
    provados = otimos[otimos["otimo_provado"].astype(str) == "True"]
    exec_ = execucoes[execucoes["algoritmo"].isin(ALGORITMOS)]
    # execucoes.csv só tem GC ligado; o filtro protege contra misturar com _sem_gc.
    if "gc_desativado" in exec_:
        exec_ = exec_[exec_["gc_desativado"].astype(str) != "True"]
    base = exec_[["algoritmo", "instancia", "custo"]].merge(
        provados[["instancia", "melhor_custo"]], on="instancia"
    )
    colunas_features = ["instancia", "origem_coordenadas", *FEATURES]
    base = base.merge(features[colunas_features], on="instancia")
    base["gap"] = base["custo"] / base["melhor_custo"] - 1.0
    return base.sort_values(["algoritmo", "n", "instancia"]).reset_index(drop=True)


def features_disponiveis(base: pd.DataFrame) -> list[str]:
    """Features sem NaN (as geométricas somem se features.csv foi gerado sem --mds)."""
    return [f for f in FEATURES if base[f].notna().all() and base[f].nunique() > 1]


# ----------------------------------------------------------- correlações
def correlacao_parcial(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Correlação de Spearman entre x e y descontado z (parcial por postos)."""
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    desenho = np.column_stack([np.ones_like(rz), rz])

    def residuo(v: np.ndarray) -> np.ndarray:
        coef, *_ = np.linalg.lstsq(desenho, v, rcond=None)
        return v - desenho @ coef

    ex, ey = residuo(rx), residuo(ry)
    if np.allclose(ex, 0) or np.allclose(ey, 0):
        return float("nan")
    return float(pearsonr(ex, ey)[0])


def correlacoes(base: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (algoritmo, feature)."""
    linhas: list[dict[str, Any]] = []
    for alg, grupo in base.groupby("algoritmo", sort=False):
        gap = grupo["gap"].to_numpy(dtype=float)
        controle = grupo[CONTROLE].to_numpy(dtype=float)
        for feature in features_disponiveis(base):
            x = grupo[feature].to_numpy(dtype=float)
            rho, p = spearmanr(x, gap)
            parcial = (
                float("nan")
                if feature == CONTROLE
                else correlacao_parcial(x, gap, controle)
            )
            linhas.append(
                {
                    "algoritmo": alg,
                    "feature": feature,
                    "instancias": len(grupo),
                    "spearman": float(rho),
                    "p_valor": float(p),
                    "parcial_dado_n": parcial,
                }
            )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------- modelos
def avaliar_loo(
    modelo: RegressorMixin, x: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Q² e RMSE em leave-one-out, mais o R² no treino (para ver sobreajuste)."""
    previsto = cross_val_predict(clone(modelo), x, y, cv=LeaveOneOut())
    press = float(np.sum((y - previsto) ** 2))
    ss = float(np.sum((y - y.mean()) ** 2))
    ajustado = clone(modelo).fit(x, y)
    residuo_treino = float(np.sum((y - ajustado.predict(x)) ** 2))
    return {
        "q2_loo": 1.0 - press / ss if ss > 0 else float("nan"),
        "rmse_loo": float(np.sqrt(press / len(y))),
        "r2_treino": 1.0 - residuo_treino / ss if ss > 0 else float("nan"),
    }


def _candidatos(disponiveis: list[str]) -> list[tuple[str, list[str], RegressorMixin]]:
    """(nome, features, modelo) avaliados para cada algoritmo."""
    linear = make_pipeline(StandardScaler(), LinearRegression())
    outras = [f for f in disponiveis if f != CONTROLE]
    return [
        ("media", [CONTROLE], DummyRegressor()),
        ("linear", [CONTROLE], linear),
        *[("linear", [CONTROLE, f], linear) for f in outras],
        ("ridge", disponiveis, RIDGE),
        ("arvore", disponiveis, ARVORE),
    ]


def modelos(base: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (algoritmo, modelo), ordenada pelo Q² dentro do algoritmo."""
    disponiveis = features_disponiveis(base)
    linhas: list[dict[str, Any]] = []
    for alg, grupo in base.groupby("algoritmo", sort=False):
        y = grupo["gap"].to_numpy(dtype=float)
        for nome, cols, modelo in _candidatos(disponiveis):
            x = grupo[cols].to_numpy(dtype=float)
            descricao = "todas" if len(cols) > 2 else " + ".join(cols)
            if nome == "media":
                descricao = "(nenhuma)"
            linhas.append(
                {
                    "algoritmo": alg,
                    "modelo": nome,
                    "features": descricao,
                    "instancias": len(y),
                    **avaliar_loo(modelo, x, y),
                }
            )
    tabela = pd.DataFrame(linhas)
    ordem = {a: i for i, a in enumerate(ALGORITMOS)}
    tabela["_ordem"] = tabela["algoritmo"].map(ordem)
    tabela = tabela.sort_values(["_ordem", "q2_loo"], ascending=[True, False])
    return tabela.drop(columns="_ordem").reset_index(drop=True)


def pesos(base: pd.DataFrame) -> pd.DataFrame:
    """Coeficientes padronizados da Ridge e importâncias da árvore (dados completos)."""
    disponiveis = features_disponiveis(base)
    linhas: list[dict[str, Any]] = []
    for alg, grupo in base.groupby("algoritmo", sort=False):
        x = grupo[disponiveis].to_numpy(dtype=float)
        y = grupo["gap"].to_numpy(dtype=float)
        ridge = clone(RIDGE).fit(x, y)
        arvore = clone(ARVORE).fit(x, y)
        coeficientes = ridge[-1].coef_
        for feature, coef, imp in zip(
            disponiveis, coeficientes, arvore.feature_importances_, strict=True
        ):
            linhas.append(
                {
                    "algoritmo": alg,
                    "feature": feature,
                    "coef_ridge_padronizado": float(coef),
                    "alpha_ridge": float(ridge[-1].alpha_),
                    "importancia_arvore": float(imp),
                }
            )
    return pd.DataFrame(linhas)


# --------------------------------------------------------------- figuras
def figura_correlacoes(corr: pd.DataFrame, pasta: Path) -> Path:
    """Dois mapas lado a lado: Spearman bruto e parcial controlando N."""
    import matplotlib.pyplot as plt

    algoritmos = [a for a in ALGORITMOS if a in set(corr["algoritmo"])]
    figura, eixos = plt.subplots(
        1, 2, figsize=(11, 5.5), sharey=True, layout="constrained"
    )
    for eixo, coluna, titulo in zip(
        eixos,
        ("spearman", "parcial_dado_n"),
        ("Spearman ρ (gap × feature)", "Parcial controlando N"),
        strict=True,
    ):
        matriz = corr.pivot(index="feature", columns="algoritmo", values=coluna)
        matriz = matriz.reindex(
            index=[f for f in FEATURES if f in matriz.index], columns=algoritmos
        )
        imagem = eixo.imshow(matriz.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        for (i, j), valor in np.ndenumerate(matriz.to_numpy()):
            if np.isfinite(valor):
                cor = "white" if abs(valor) > 0.6 else "black"
                eixo.text(j, i, f"{valor:.2f}", ha="center", va="center", color=cor)
        eixo.set_xticks(range(len(algoritmos)), algoritmos, rotation=30, ha="right")
        eixo.set_yticks(range(len(matriz.index)), matriz.index)
        eixo.set_title(titulo)
    figura.colorbar(imagem, ax=eixos, shrink=0.8)
    return salvar_figura(figura, "correlacoes.png", pasta)


def figura_dispersao(base: pd.DataFrame, corr: pd.DataFrame, pasta: Path) -> list[Path]:
    """Gap × as 3 features de maior |parcial| para cada aproximado."""
    import matplotlib.pyplot as plt

    caminhos = []
    for alg in ("christofides", "twice_around_the_tree"):
        sub_corr = corr[(corr["algoritmo"] == alg) & (corr["feature"] != CONTROLE)]
        if sub_corr.empty:
            continue
        topo = sub_corr.reindex(
            sub_corr["parcial_dado_n"].abs().sort_values(ascending=False).index
        ).head(3)
        grupo = base[base["algoritmo"] == alg]
        figura, eixos = plt.subplots(
            1, len(topo), figsize=(5 * len(topo), 4), layout="constrained"
        )
        for eixo, (_, linha) in zip(np.atleast_1d(eixos), topo.iterrows(), strict=True):
            pontos = eixo.scatter(
                grupo[linha["feature"]],
                100 * grupo["gap"],
                c=grupo["n"],
                cmap="viridis",
            )
            for _, inst in grupo.iterrows():
                eixo.annotate(
                    inst["instancia"],
                    (inst[linha["feature"]], 100 * inst["gap"]),
                    fontsize=7,
                    xytext=(3, 3),
                    textcoords="offset points",
                )
            eixo.set(
                xlabel=linha["feature"],
                ylabel="gap (%)",
                title=f"ρ={linha['spearman']:.2f}  parcial={linha['parcial_dado_n']:.2f}",
            )
            eixo.grid(True, alpha=0.3)
        figura.colorbar(pontos, ax=eixos, label="N")
        figura.suptitle(alg)
        caminhos.append(salvar_figura(figura, f"gap_features_{alg}.png", pasta))
    return caminhos


def figura_arvore(
    base: pd.DataFrame, pasta: Path, alg: str = "christofides"
) -> Path | None:
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    grupo = base[base["algoritmo"] == alg]
    if grupo.empty:
        return None
    disponiveis = features_disponiveis(base)
    arvore = clone(ARVORE).fit(grupo[disponiveis], grupo["gap"])
    figura, eixo = plt.subplots(figsize=(10, 5))
    plot_tree(arvore, feature_names=disponiveis, filled=True, precision=3, ax=eixo)
    eixo.set_title(f"Árvore de decisão (profundidade 2) — gap do {alg}")
    return salvar_figura(figura, f"arvore_{alg}.png", pasta)


# ------------------------------------------------------------------ main
def main(
    pasta_resultados: Path | None = None,
    pasta_tabelas: Path = TABELAS,
    pasta_figuras: Path = FIGURAS,
) -> dict[str, pd.DataFrame]:
    base = montar_base(
        ler_resultado(ENTRADA_EXECUCOES, pasta_resultados),
        ler_resultado(ENTRADA_OTIMOS, pasta_resultados),
        ler_resultado(ENTRADA_FEATURES, pasta_resultados),
    )
    if base.empty:
        raise ValueError(
            "nenhuma instância com ótimo provado: rode o B&B e --consolidar"
        )
    tabelas = {
        "e3_base.csv": base,
        "e3_correlacoes.csv": correlacoes(base),
        "e3_modelos.csv": modelos(base),
        "e3_pesos.csv": pesos(base),
    }
    for nome, tabela in tabelas.items():
        salvar_tabela(tabela, nome, pasta_tabelas)

    destino_figuras = pasta_figuras / PASTA_FIGURAS
    figura_correlacoes(tabelas["e3_correlacoes.csv"], destino_figuras)
    figura_dispersao(base, tabelas["e3_correlacoes.csv"], destino_figuras)
    figura_arvore(base, destino_figuras)

    _imprimir_resumo(base, tabelas["e3_correlacoes.csv"], tabelas["e3_modelos.csv"])
    return tabelas


def _imprimir_resumo(
    base: pd.DataFrame, corr: pd.DataFrame, mods: pd.DataFrame
) -> None:
    instancias = sorted(base["instancia"].unique())
    print(f"Instâncias com ótimo provado ({len(instancias)}): {', '.join(instancias)}")
    gaps = base.pivot(index="instancia", columns="algoritmo", values="gap")
    print("\n=== Gap (%) ===")
    print((100 * gaps[[a for a in ALGORITMOS if a in gaps]]).round(1).to_string())
    print("\n=== Maior |parcial dado N| por algoritmo ===")
    sem_n = corr[corr["feature"] != CONTROLE].dropna(subset=["parcial_dado_n"])
    for alg, grupo in sem_n.groupby("algoritmo", sort=False):
        topo = grupo.loc[grupo["parcial_dado_n"].abs().idxmax()]
        print(
            f"{alg:<24} {topo['feature']:<24} parcial={topo['parcial_dado_n']:+.2f} "
            f"(ρ bruto={topo['spearman']:+.2f}, p={topo['p_valor']:.3f})"
        )
    print("\n=== Melhor modelo por algoritmo (Q² em leave-one-out) ===")
    melhores = mods.groupby("algoritmo", sort=False).head(1)
    colunas = ["algoritmo", "modelo", "features", "q2_loo", "r2_treino"]
    print(melhores[colunas].round(3).to_string(index=False))


if __name__ == "__main__":
    main()