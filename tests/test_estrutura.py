"""Garante a hierarquia das camadas descrita em ESTRUTURA.md.

tsp (algoritmos) ← experimentos (coleta) → resultados/ → analises (análise).
Cada pacote só pode importar os pacotes permitidos abaixo.
"""

import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
PACOTES = ["tsp", "experimentos", "analises"]
PROIBIDOS = {
    "tsp": {"experimentos", "analises"},  # biblioteca não conhece quem a usa
    "experimentos": {"analises"},  # coleta não depende de análise
    "analises": {"tsp", "experimentos"},  # análise só lê CSV
}


def _importados(arquivo: Path) -> set[str]:
    arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
    nomes: set[str] = set()
    for no in ast.walk(arvore):
        if isinstance(no, ast.Import):
            nomes |= {alias.name.split(".")[0] for alias in no.names}
        elif isinstance(no, ast.ImportFrom) and no.module and no.level == 0:
            nomes.add(no.module.split(".")[0])
    return nomes


@pytest.mark.parametrize("pacote", PACOTES)
def test_camadas_respeitam_a_hierarquia(pacote):
    violacoes = {
        str(arquivo.relative_to(RAIZ)): sorted(_importados(arquivo) & PROIBIDOS[pacote])
        for arquivo in (RAIZ / pacote).rglob("*.py")
    }
    assert {k: v for k, v in violacoes.items() if v} == {}


def test_analises_nao_escrevem_em_resultados():
    # A análise lê resultados/ e escreve só em relatorio/.
    for arquivo in (RAIZ / "analises").rglob("*.py"):
        texto = arquivo.read_text(encoding="utf-8")
        assert "to_csv(RESULTADOS" not in texto, arquivo
