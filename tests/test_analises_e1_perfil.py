import pandas as pd
import pytest

from analises import e1_perfil


def _subrotinas(fator_sem_gc: float = 1.0) -> pd.DataFrame:
    """Christofides: 80% no emparelhamento; TATT: 70% na AGM."""
    linhas = []
    for i, n in enumerate((20, 50, 100)):
        inst = f"inst{i}"
        partes = {
            "christofides": {"total/emparelhamento": 0.8, "total/agm": 0.15},
            "twice_around_the_tree": {"total/agm": 0.7, "total/pre_ordem": 0.2},
        }
        for alg, etapas in partes.items():
            total = 1e-3 * n * fator_sem_gc
            linhas.append(
                {"algoritmo": alg, "instancia": inst, "n": n, "etapa": "total",
                 "tempo_mediana_s": total, "coletas_gc_mediana": 0}
            )  # fmt: skip
            for etapa, fracao in etapas.items():
                linhas.append(
                    {"algoritmo": alg, "instancia": inst, "n": n, "etapa": etapa,
                     "tempo_mediana_s": fracao * total, "coletas_gc_mediana": 0}
                )  # fmt: skip
    # Etapa opcional (como o mds) presente só numa instância.
    linhas.append(
        {"algoritmo": "twice_around_the_tree", "instancia": "inst0", "n": 20,
         "etapa": "total/mds", "tempo_mediana_s": 0.002, "coletas_gc_mediana": 0}
    )  # fmt: skip
    return pd.DataFrame(linhas)


def _bnb() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "instancia": ["a", "b"],
            "n": [14, 52],
            "estrategia": "christofides",
            "status": "OK",
            "tempo_bnb_s": [1.0, 100.0],
            "tempo_penalidades_s": [0.5, 0.1],
            "tempo_busca_s": [0.5, 99.9],
            "tempo_limite_inferior_s": [0.45, 99.0],
            "nos_expandidos": [100, 10000],
        }
    )


def test_perfil_acha_a_etapa_dominante():
    perfil = e1_perfil.perfil_etapas(_subrotinas(), "teste")
    primeira = perfil.groupby("algoritmo").head(1).set_index("algoritmo")
    assert primeira.loc["christofides", "etapa"] == "total/emparelhamento"
    assert primeira.loc["christofides", "fracao_mediana"] == pytest.approx(0.8)
    assert primeira.loc["twice_around_the_tree", "etapa"] == "total/agm"


def test_etapa_opcional_conta_zero_na_mediana():
    perfil = e1_perfil.perfil_etapas(_subrotinas(), "teste").set_index(
        ["algoritmo", "etapa"]
    )
    mds = perfil.loc[("twice_around_the_tree", "total/mds")]
    assert mds["fracao_mediana"] == 0.0  # ausente em 2 de 3 instâncias
    assert mds["instancias_com_etapa"] == 1
    assert mds["fracao_mediana_quando_existe"] == pytest.approx(0.1)


def test_perfil_bnb():
    p = e1_perfil.perfil_bnb(_bnb()).set_index("instancia")
    assert p.loc["a", "fracao_penalidades"] == pytest.approx(0.5)
    assert p.loc["b", "fracao_limite_na_busca"] == pytest.approx(99.0 / 99.9)
    assert p.loc["b", "us_por_no"] == pytest.approx(1e6 * 99.9 / 10000)


def test_comparar_gc():
    gc = e1_perfil.comparar_gc(_subrotinas(), _subrotinas(fator_sem_gc=0.5))
    assert gc["razao_mediana"].tolist() == pytest.approx([0.5, 0.5])
    assert (gc["coletas_gc_com"] == 0).all()


def test_main(tmp_path):
    resultados = tmp_path / "resultados"
    resultados.mkdir()
    _subrotinas().to_csv(resultados / e1_perfil.ENTRADA_COM_GC, index=False)
    _subrotinas(0.9).to_csv(resultados / e1_perfil.ENTRADA_SEM_GC, index=False)
    _subrotinas().to_csv(resultados / e1_perfil.ENTRADA_ALEATORIAS, index=False)
    for nome in e1_perfil.ENTRADAS_BNB:
        _bnb().to_csv(resultados / nome, index=False)
    tabelas, figuras = tmp_path / "tabelas", tmp_path / "figuras"

    saidas = e1_perfil.main(resultados, tabelas, figuras)

    for nome in saidas:
        assert (tabelas / nome).exists()
    pngs = {p.name for p in (figuras / e1_perfil.PASTA_FIGURAS).glob("*.png")}
    assert pngs == {"perfil_etapas.png", "perfil_emparelhamento.png", "gc.png"}
