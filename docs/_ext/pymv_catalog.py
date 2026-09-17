"""Render live catalogs without downloading datasets."""
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
            if isinstance(value, tuple):
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
        with mock(["tensorflow"]):
            from pymv.data import get_dataset_info, list_datasets, list_variants
            from pymv.models import get_model_info, list_models
        if self.arguments[0] == "models":
            rows = []
            for name in list_models():
                info = get_model_info(name)
                rows.append([name.upper(), info["description"], "Yes" if info["uses_graphs"] else "No",
                             "Yes" if info["supports_missing_views"] else "No",
                             ("Original paper", info["paper"]) if info["paper"] else "Not supplied"])
            return [table(["Model", "Description", "Uses graphs", "Native missing views", "Reference"], rows)]
        rows = []
        for name in list_datasets():
            for variant in list_variants(name):
                info = get_dataset_info(name, variant=variant)
                rows.append([name, variant, info["n_views"], info["category"],
                             info["description"] + (". " + info["availability_note"]
                                                    if info["availability_note"] else ""),
                             ("Artifact", info["url"])])
        return [table(["Dataset", "Variant", "Views", "Setting", "Description", "Source"], rows)]


def setup(app):
    app.add_directive("pymv-catalog", Catalog)
    return {"version": "1", "parallel_read_safe": True}
