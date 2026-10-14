"""Render model and dataset catalogs from the same public metadata as the UI."""
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


def dataset_sections(get_info, list_datasets, list_variants):
    from polyclustx.data.registry import CATEGORY_LABELS
    sections = []
    for category, label in CATEGORY_LABELS.items():
        section = nodes.section(ids=[f"datasets-{category.replace('_', '-')}"])
        section += nodes.title(text=label)
        for name in list_datasets(category=category):
            info = get_info(name)
            family = nodes.section(ids=[f"dataset-{name.replace('_', '-')}"])
            family += nodes.title(text=name)
            family += nodes.paragraph(text=info["description"])
            samples = f"{info['n_samples']:,}" if info["n_samples"] is not None else "Unspecified"
            family += nodes.paragraph(
                text=f"{samples} samples/nodes; {info['n_clusters']} classes; "
                     f"{info['n_feature_views']} stored feature matrices; {info['n_graphs']} graph relations; "
                     f"{info['n_views']} default composed views.")
            dimensions = info["feature_dims"]
            if info["feature_descriptions"]:
                family += table(["Stored feature", "Meaning", "Columns"], [
                    (f"X{index + 1}*", description,
                     dimensions[index] if index < len(dimensions) else "Inspect the loaded matrix")
                    for index, description in enumerate(info["feature_descriptions"])
                ])
            if info["graph_descriptions"]:
                family += table(["Structural relation", "Adjacency shape"], [
                    (description, f"{samples} × {samples}")
                    for description in info["graph_descriptions"]
                ])
            stats = info["graph_statistics"]
            if stats:
                family += nodes.paragraph(
                    text=f"Reference graph: {stats['stored_arcs']:,} stored arcs; "
                         f"{stats['unordered_pairs']:,} unordered pairs; {stats['self_loops']} existing self-loops. "
                         f"Symmetric adjacency: {'yes' if stats['symmetric'] else 'no'}. "
                         "Graph direction and weights are preserved when loading.")
            if info["encoders"]:
                family += nodes.paragraph(text="Default encoders: " + info["encoders"])
            if info["reference_details"]:
                family += nodes.paragraph(text=info["reference_details"])
            source = nodes.paragraph()
            source += nodes.reference(text="Download default MAT", refuri=info["url"])
            source += nodes.Text(" · ")
            source += nodes.reference(text="Reference / benchmark background", refuri=info["reference"])
            family += source
            variants = list_variants(name)
            if len(variants) > 1:
                family += nodes.paragraph(text="Alternate feature representations:")
                family += table(["Variant", "Encoder pair", "Stored columns", "Download"], [
                    (variant, details["encoders"], " / ".join(map(str, details["feature_dims"])),
                     ("MAT file", details["url"]))
                    for variant in variants if variant != "default"
                    for details in [get_info(name, variant=variant)]
                ])
            section += family
        sections.append(section)
    return sections


class Catalog(SphinxDirective):
    required_arguments = 1

    def run(self):
        with mock(["torch"]):
            from polyclustx.data import get_dataset_info, list_datasets, list_variants
            from polyclustx.models import get_model_info, list_models
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
        return dataset_sections(get_dataset_info, list_datasets, list_variants)


def setup(app):
    app.add_directive("polyclustx-catalog", Catalog)
    return {"version": "2", "parallel_read_safe": True}
