"""Informações do ambiente de execução, gravadas em todos os CSVs de coleta.

O relatório precisa dizer em que hardware cada tempo foi medido, e as
regressões só são válidas com tempos de uma mesma máquina. Por isso cada linha
de ``execucoes``, ``subrotinas``, ``subrotinas_aleatorias_e1`` e ``bnb`` leva o
modelo do processador, o nº de núcleos e a versão do Python. O nome do
computador não é gravado.
"""

from __future__ import annotations

import os
import platform
import subprocess
from functools import cache
from pathlib import Path
from typing import Any

COLUNAS_AMBIENTE = ["processador", "nucleos_fisicos", "nucleos_logicos", "python"]


def _comando(*args: str) -> str:
    try:
        saida = subprocess.run(
            args, capture_output=True, text=True, timeout=5, check=False
        )
        return saida.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _processador_linux() -> tuple[str, int | None]:
    """Modelo e nº de núcleos físicos lidos de ``/proc/cpuinfo``."""
    try:
        texto = Path("/proc/cpuinfo").read_text()
    except OSError:
        return "", None
    modelo, nucleos = "", set()
    fisico = ""
    for linha in texto.splitlines():
        chave, _, valor = (x.strip() for x in linha.partition(":"))
        if chave == "model name" and not modelo:
            modelo = valor
        elif chave == "physical id":
            fisico = valor
        elif chave == "core id":
            nucleos.add((fisico, valor))
    return modelo, len(nucleos) or None


def _processador_macos() -> tuple[str, int | None]:
    modelo = _comando("sysctl", "-n", "machdep.cpu.brand_string").strip()
    fisicos = _comando("sysctl", "-n", "hw.physicalcpu").strip()
    return modelo, int(fisicos) if fisicos.isdigit() else None


@cache
def info_ambiente() -> dict[str, Any]:
    """Processador, núcleos e versão do Python (colunas ``COLUNAS_AMBIENTE``).

    Núcleos físicos ficam vazios quando o sistema não informa (ex.: Windows);
    os lógicos incluem hyper-threading.
    """
    sistema = platform.system()
    modelo, fisicos = "", None
    if sistema == "Linux":
        modelo, fisicos = _processador_linux()
    elif sistema == "Darwin":
        modelo, fisicos = _processador_macos()
    return {
        "processador": modelo or platform.processor() or platform.machine(),
        "nucleos_fisicos": "" if fisicos is None else fisicos,
        "nucleos_logicos": os.cpu_count() or "",
        "python": platform.python_version(),
    }
