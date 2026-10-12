"""On-demand dataset exports and disk-backed visualization snapshots."""

from __future__ import annotations

import io
import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from scipy.io import savemat

from polyclustx.experiment.exports import json_safe


def dataset_mat(data, *, include_labels=True):
    """Serialize prepared data to MATLAB v5, keeping sparse matrices sparse.

    Arrays are named view_1, graph_1_1, etc.; graph_mapping lists one-based
    view/relation pairs. Model-internal transformations are not performed.
    """
    arrays = {f"view_{i}": view for i, view in enumerate(data.Xs, 1)}
    mapping = []
    for i, relations in enumerate(data.As, 1):
        relations = relations if isinstance(relations, (list, tuple)) else [relations]
        for j, graph in enumerate(relations, 1):
            if graph is not None:
                arrays[f"graph_{i}_{j}"] = graph
                mapping.append((i, j))
    arrays.update(
        sample_ids=data.sample_ids, observed_mask=data.observed_mask.astype(np.uint8),
        view_names=np.asarray(data.view_names, dtype=object),
        modalities=np.asarray(data.modalities, dtype=object),
        graph_mapping=np.asarray(mapping, dtype=np.int64).reshape(-1, 2),
        metadata_json=json.dumps(json_safe(data.metadata), allow_nan=False),
        dataset_name=data.name or "Dataset",
    )
    if include_labels and data.labels is not None:
        arrays["labels"] = data.labels
    output = io.BytesIO()
    savemat(output, arrays, do_compression=True, oned_as="column", long_field_names=True)
    return output.getvalue()


class PlotSnapshot:
    """Disk-backed plot snapshot, independent of source datasets and run caches."""

    def __init__(self, name, figure, table):
        figure_json, csv = figure.to_json().encode(), table.to_csv(index=False).encode()
        if len(figure_json) + len(csv) > 32 * 1024**2:
            raise ValueError("This plot exceeds the 32 MB snapshot limit; reduce the plotted sample count")
        self.name = name
        self._directory = tempfile.TemporaryDirectory(prefix="polyclustx-plot-")
        self.path = Path(self._directory.name)
        (self.path / "figure.json").write_bytes(figure_json)
        (self.path / "data.csv").write_bytes(csv)

    def export(self, format):
        if format in {"json", "csv"}:
            return (self.path / ("figure.json" if format == "json" else "data.csv")).read_bytes()
        if format != "zip":
            raise ValueError("Plot format must be json, csv or zip")
        import plotly.io as pio

        figure = pio.from_json((self.path / "figure.json").read_text())
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(self.path / "figure.json", "figure.json")
            archive.write(self.path / "data.csv", "data.csv")
            archive.writestr("figure.html", figure.to_html(include_plotlyjs=True))
        return output.getvalue()

    def cleanup(self):
        self._directory.cleanup()
