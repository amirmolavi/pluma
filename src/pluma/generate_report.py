__all__ = ["generate_report"]

import os
from pathlib import Path
from jinja2 import Template

import nbformat
from openai import OpenAI
from tqdm import tqdm
from IPython.display import Markdown, display, update_display

system_prompt = """
You are an expert at writing a report based on some code and the results provided from running it. 
You will be given the code, some results, and maybe comments and descriptions. 
use the formal format of a report. put sections like introduction, Materials and Methods which includes Study Design
Summary of the clinical trial design or experiment.
Inclusion/exclusion criteria for subjects.
Data Collection
Description of PK/PD data collection (e.g., sampling times, bioanalytical methods).
PK/PD Analysis Methods
Software and statistical methods used.
Model descriptions (e.g., compartmental or non-compartmental).
Covariate analysis methods.
Bioanalytical Methods
Analytical techniques for measuring drug concentrations.
Validation of bioanalytical assays. Results, conclusion. 
You provide detailed science of the project.
Also provide table of content.
You will generate a report explaining the science, the graphs, the tables, and put them in the report. Respond in markdown.
"""


def generate_report(
    notebook_path: Path | str,
    report_path: Path | str,
    openai_api_key: str | None = None,
    model_name: str = "gpt-3.5-turbo",
    blue_print: Path | None = None,
) -> None:
    notebook_path = Path(notebook_path)
    report_path = Path(report_path)

    if openai_api_key is None:
        openai_api_key = os.environ.get("OPENAI_API_KEY")
    with open(notebook_path, "r", encoding="utf-8") as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

        cells = []

        for i, cell in enumerate(tqdm(notebook.cells)):
            if len(cell.source.strip()) > 0:
                cells.append(cell)

    user_content = provide_user_content(cells=cells)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    client = OpenAI(api_key=openai_api_key)

    stream = client.chat.completions.create(model=model_name, messages=messages, stream=True)

    response = ""
    display_handle = display(Markdown(""), display_id=True)
    for chunk in stream:
        response += chunk.choices[0].delta.content or ""
        response = response.replace("```", "").replace("markdown", "")
        update_display(Markdown(response), display_id=display_handle.display_id)

    # report_path.write_text(response.choices[0].message.content, encoding="utf-8")


def provide_user_content(cells: list):
    template_str = """Here is the code context and all the outputs for you to work with:

    [[CELLS START]]
    {% for cell in cells %}
    {{ cell.source }}

    {% endfor %}
    [[CELLS END]]

    -----

    Please provide a report of the full work as instructed.
    Take into account the comments, data, and figures. put the plots and tables in the report with
    the right caption and explain them.
    You provide detailed science of the project.
    Give the final report in markdown format. Add useful tables and figures.
    """

    template = Template(template_str)

    # Render the template with the provided cells
    rendered_content = template.render(cells=cells)

    return rendered_content
