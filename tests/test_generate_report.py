from pathlib import Path

from pluma import generate_report

dir = Path(__file__).parent


def test_generate_report():
    generate_report(dir.joinpath("notebooks/gene.ipynb"), dir.joinpath("notebooks/report.tex"))
