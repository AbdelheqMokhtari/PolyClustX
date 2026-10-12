"""Render live catalogs without downloading datasets."""
from pathlib import Path

from docutils import nodes
from sphinx.ext.autodoc.mock import mock
from sphinx.util.docutils import SphinxDirective


def table(headers, rows):
    result = nodes.table()
    group = nodes.tgroup(cols=len(headers))
    result += group
    for _ in headers:
        group += nodes.colspec(colwidth=1)
    head, body = nodes.thead(), nodes.tbody()
    group += head
    group += body
    for parent, entries in [(head, headers), *[(body, row) for row in rows]]:
        row = nodes.row()
        for value in entries:
            entry = nodes.entry()
            if isinstance(value, nodes.Node):
                paragraph = value
            elif isinstance(value, tuple):
                label, url = value
                paragraph = nodes.paragraph()
                paragraph += nodes.reference(text=label, refuri=url)
            else:
                paragraph = nodes.paragraph(text=str(value))
            entry += paragraph
            row += entry
        parent += row
    return result


class Catalog(SphinxDirective):
    required_arguments = 1

    def run(self):
        with mock(["torch"]):
            from pymv.data import get_dataset_info, list_datasets, list_variants
            from pymv.models import get_model_info, list_models
        if self.arguments[0] == "models":
            rows = []
            for name in list_models():
                info = get_model_info(name)
                rows.append([name.upper(), info["description"], "Yes" if info["uses_graphs"] else "No",
                             ("Yes; ≥2 complete training samples" if getattr(info["class"], "min_complete_samples", 0)
                              else "Yes") if info["supports_missing_views"] else "No",
                             ("Paper", info["paper"]) if info["paper"] else "Not supplied",
                             ("Code", info["implementation"]) if info.get("implementation") else "Not supplied"])
            return [table(["Model", "Description", "Graphs", "Missing views", "Paper", "Original code"], rows)]
        rows = []
        for name in list_datasets():
            variants, descriptions, sources = nodes.bullet_list(), nodes.container(), nodes.container()
            for variant in list_variants(name):
                info = get_dataset_info(name, variant=variant)
                variants += nodes.list_item("", nodes.paragraph(
                    text=f"{variant}: {info['n_views']} views; {info['category']}"))
                descriptions += nodes.paragraph(text=f"{variant}: {info['description']}")
                if info["reference_details"]:
                    descriptions += nodes.paragraph(text="Reference collection: " + info["reference_details"])
                if info["availability_note"]:
                    descriptions += nodes.paragraph(text=info["availability_note"])
                paragraph = nodes.paragraph()
                paragraph += nodes.reference(text=f"{variant} artifact", refuri=info["url"])
                paragraph += nodes.Text(" · ")
                reference_label = "Reference" if info["reference_verified"] else "Provisional reference"
                paragraph += nodes.reference(text=reference_label, refuri=info["reference"])
                sources += paragraph
            rows.append([name, variants, descriptions, sources])
        return [table(["Dataset family", "Variants / setting", "Description", "Sources"], rows)]


class Screenshot(SphinxDirective):
    """Optional, git-ignored screenshots with useful placeholders in clean builds."""

    required_arguments = 1
    has_content = True

    def run(self):
        filename = Path(self.arguments[0]).name
        relative = f"_static/screenshots/{filename}"
        path = Path(self.env.srcdir) / relative
        self.env.note_dependency(str(path))
        caption = " ".join(self.content)
        if path.is_file():
            figure = nodes.figure()
            figure += nodes.image(uri="/" + relative, alt=caption)
            figure += nodes.caption(text=caption)
            return [figure]
        placeholder = nodes.admonition(classes=["note"])
        placeholder += nodes.title(text="Screenshot guide")
        placeholder += nodes.paragraph(text=caption)
        placeholder += nodes.paragraph(text=f"Image reference: {filename}")
        return [placeholder]


def setup(app):
    app.add_directive("pymv-catalog", Catalog)
    app.add_directive("pymv-screenshot", Screenshot)
    return {"version": "1", "parallel_read_safe": True}
