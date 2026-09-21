"""Interactive pymv interface. Run with ``pymv-display``."""

from __future__ import annotations

import copy
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pymv.data import (
    dataset_from_arrays,
    get_dataset_info,
    list_datasets,
    list_variants,
    load_data,
    load_files,
    read_arrays,
)
from pymv.display.exporting import PlotSnapshot, dataset_mat
from pymv.display.model_controls import (
    ParameterControl,
    compatible_models,
    parameter_controls,
    parse_candidates,
    parse_int_tuple,
)
from pymv.display.preprocessing import PreprocessingSpec, ViewPreprocessing, prepare_dataset
from pymv.display.profiles import dataset_profile, format_bytes
from pymv.display.results import (
    aggregate_results,
    cluster_details,
    compact_confusion,
    detailed_rows,
    download_stem,
    record_download_stem,
    record_label,
    run_label,
)
from pymv.display.synthetic import SyntheticGraphSpec, SyntheticSpec, generate_synthetic
from pymv.display.theme import CSS
from pymv.display.workspace import (
    EXTERNAL_NAMES,
    INTERNAL_NAMES,
    ExperimentSpec,
    Workspace,
    export_record,
)
from pymv.experiment.exports import normalized_loss, report_csv, report_json, report_zip
from pymv.models.registry import get_model_info, list_models

ASSETS = Path(__file__).resolve().parent / "assets"
STEPS = ["Dataset", "Preprocessing", "Model", "Experiment", "Results", "Visualizations", "Export", "Search"]


def _workspace():
    if "workspace" not in st.session_state:
        st.session_state.workspace = Workspace()
    return st.session_state.workspace


def _select_dataset(workspace, *, required=True):
    if not workspace.datasets:
        if required:
            st.info("Add a dataset in step 1 to continue.")
        return None, None
    selected = st.session_state.get("selected_dataset")
    if selected not in workspace.datasets:
        selected = next(reversed(workspace.datasets))
        st.session_state.selected_dataset = selected
    if st.session_state.get("suspended_dataset") == selected:
        st.info("The active dataset is unloaded. Load it when you are ready to inspect or configure it.")
        if st.button("Load active dataset"):
            st.session_state.pop("suspended_dataset", None)
            st.rerun()
        return selected, None
    return selected, workspace.load_dataset(selected)


def _set_page(name):
    st.session_state._next_workflow_step = STEPS.index(name)


def _new_dataset_name(workspace, prefix, key):
    name = st.text_input("Dataset name", workspace.suggested_name(prefix),
                         key=f"{key}_{workspace.name_counters.get(prefix, 0)}")
    return name.strip() or workspace.suggested_name(prefix)


def _loaded_dataset(workspace, key, prefix=None):
    workspace.pending_datasets.add(key)
    if prefix:
        workspace.advance_name(prefix)
    st.session_state.selected_dataset = key
    st.session_state.dataset_browser = key
    st.session_state.pop("suspended_dataset", None)


def _dataset_picker(workspace, key, *, prepared_only=False):
    ids = [key for key in workspace.datasets if not prepared_only or key not in workspace.pending_datasets]
    if not ids:
        st.info("Load a dataset and apply its preparation on the Dataset page first.")
        return None
    preferred = st.session_state.get("selected_dataset")
    if st.session_state.get(key) not in ids:
        st.session_state[key] = preferred if preferred in ids else ids[-1]
    selected = st.selectbox("Prepared dataset" if prepared_only else "Browse datasets", ids,
                            format_func=lambda value: workspace.datasets[value].name, key=key)
    st.session_state.selected_dataset = selected
    return selected


def _stage_uploaded(workspace, data, files, mapping):
    """Keep uploaded source files on disk so inactive matrices can leave RAM."""
    directory = Path(tempfile.mkdtemp(prefix="pymv-display-"))
    paths = []
    try:
        seen = set()
        for file in files:
            name = Path(file.name).name
            if name in seen:
                raise ValueError("Uploaded filenames must be unique")
            seen.add(name)
            path = directory / name
            path.write_bytes(file.getvalue())
            paths.append(path)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    key = workspace.add_lazy_dataset(
        data.name or "Uploaded dataset",
        loader=lambda local=tuple(paths), options=copy.deepcopy(mapping): load_files(local, **options),
        source="upload",
        hint={
            "n_samples": data.n_samples,
            "n_views": data.n_views,
            "n_clusters": data.n_clusters,
            "complete": data.is_complete,
            "category": data.category,
        },
    )
    workspace.datasets[key].cleanup = lambda target=directory: shutil.rmtree(target, ignore_errors=True)
    workspace.datasets[key]._data = data
    _loaded_dataset(workspace, key, "uploaded-dataset")
    return key


def _read_uploads(files):
    arrays = {}
    for file in files:
        for key, value in read_arrays(file.getvalue(), filename=file.name).items():
            full = f"{file.name}:{key}"
            if full in arrays:
                raise ValueError(f"Duplicate file/array name: {full}")
            arrays[full] = value
    return arrays


def _profile(data, profile=None):
    profile = dataset_profile(data) if profile is None else profile
    columns = st.columns(6)
    values = [
        ("Samples", profile["n_samples"]),
        ("Views", profile["n_views"]),
        ("Classes", profile["labels"]["n_classes"] if profile["labels"] else "Unknown"),
        ("Complete", "Yes" if profile["complete"] else "No"),
        ("Graphs", len(profile["graphs"])),
        ("Estimated size", format_bytes(profile["memory_bytes"])),
    ]
    for column, (label, value) in zip(columns, values):
        column.metric(label, value)
    st.caption(
        f"{profile['category'].replace('_', ' ')} · "
        f"{profile['observed_pairs']:,} observed and {profile['missing_pairs']:,} missing view/sample pairs"
    )
    view_rows = []
    for view in profile["views"]:
        view_rows.append(
            {
                "view": view["name"],
                "modality": view["modality"],
                "shape": f"{view['shape'][0]} × {view['shape'][1]}",
                "observed": view["observed"],
                "missing": view["missing"],
                "mean": view["mean"],
                "variance": view["variance"],
                "zero fraction": view["zero_fraction"],
                "median feature variance": view["feature_variance_quantiles"][2],
                "memory": format_bytes(view["memory_bytes"]),
            }
        )
    st.subheader("Feature views")
    st.dataframe(pd.DataFrame(view_rows), hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader("Class distribution")
        if profile["labels"] is None:
            st.caption("No ground-truth labels were loaded.")
        else:
            counts = profile["labels"]["counts"]
            shown = sorted(counts, key=lambda item: item[1], reverse=True)[:30]
            st.bar_chart(pd.DataFrame(shown, columns=["class", "samples"]).set_index("class"), color="#8A1538")
            st.caption(
                f"Imbalance ratio {profile['labels']['imbalance_ratio']:.2f} · "
                f"normalized entropy {profile['labels']['normalized_entropy']:.3f}"
                + (" · showing 30 largest classes" if len(counts) > 30 else "")
            )
    with right:
        st.subheader("Graph relations")
        if profile["graphs"]:
            graph_rows = [
                {
                    "view": item["view_name"],
                    "relation": item["relation"],
                    "nonzeros": item["edges_nonzero"],
                    "density": item["density"],
                    "symmetric": item["symmetric"],
                    "weighted": item["weighted"],
                }
                for item in profile["graphs"]
            ]
            st.dataframe(pd.DataFrame(graph_rows), hide_index=True, width="stretch")
        else:
            st.caption("Feature-only dataset: no graph relation was loaded.")


def _catalog_loader(workspace):
    name = st.selectbox("Dataset", list_datasets())
    variant = st.selectbox("Variant", list_variants(name))
    info = get_dataset_info(name, variant=variant)
    st.write(info["description"])
    st.caption(f"{info['category']} · {info['n_views']} views · downloaded files are cached locally")
    if info["availability_note"]:
        st.warning(info["availability_note"])
    if st.button("Add catalog dataset", type="primary", disabled=bool(info["availability_note"])):
        try:
            with st.spinner("Loading and validating the dataset"):
                data = load_data(name, variant=variant)
                key = workspace.add_lazy_dataset(
                    data.name or name,
                    lambda dataset=name, choice=variant: load_data(dataset, variant=choice),
                    source="catalog",
                )
                workspace.datasets[key]._data = data
                workspace.datasets[key].hint = {
                    "n_samples": data.n_samples,
                    "n_views": data.n_views,
                    "n_clusters": data.n_clusters,
                    "complete": data.is_complete,
                    "category": data.category,
                }
                _loaded_dataset(workspace, key)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _simple_upload(workspace):
    name = _new_dataset_name(workspace, "uploaded-dataset", "simple_dataset_name")
    st.write("Upload one CSV, TSV, or NPY feature matrix for each view. Rows are samples; columns are features.")
    st.caption("CSV/TSV headers are detected automatically. Include Y in the same upload batch if you have it.")
    views = st.file_uploader(
        "View files and optional labels file", type=["csv", "tsv", "npy"],
        accept_multiple_files=True, key="simple_views"
    )
    location = st.radio(
        "Ground-truth labels",
        ["No labels", "A file in this upload", "A column in a view", "Upload labels separately"],
        key="simple_label_location",
    )
    labels_file = None
    if location == "Upload labels separately":
        labels_file = st.file_uploader(
            "Labels file", type=["csv", "tsv", "npy"], key="simple_labels"
        )
    files = [*views, *([labels_file] if labels_file else [])]
    if not views:
        return
    try:
        arrays = _read_uploads(files)
    except Exception as exc:
        st.error(str(exc))
        return
    keys = list(arrays)
    view_keys = list(keys)
    label_key, label_column = None, None
    ready = True
    if location == "A file in this upload":
        label_key = st.selectbox("Which file contains Y?", keys, key="simple_label_key")
        view_keys.remove(label_key)
    elif location == "Upload labels separately":
        ready = labels_file is not None
        if labels_file:
            label_key = next(key for key in keys if key.startswith(f"{labels_file.name}:"))
            view_keys.remove(label_key)
    elif location == "A column in a view":
        key = st.selectbox("View containing the labels", keys, key="simple_label_view")
        matrix = arrays[key]
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            st.error("Choose a matrix containing at least one feature column and one label column.")
            ready = False
        else:
            column = st.number_input(
                "Label column (1 = first column)", 1, matrix.shape[1], matrix.shape[1], key="simple_label_column"
            )
            label_column = (key, column - 1)
            st.caption("This column will be removed from that view's features before training.")
    st.caption(f"{len(view_keys)} feature view(s) selected. All files must use the same sample row order.")
    if st.button("Add uploaded dataset", type="primary", disabled=not view_keys or not ready):
        try:
            mapping = {"view_keys": view_keys, "label_key": label_key, "label_column": label_column, "name": name}
            data = dataset_from_arrays(arrays, **mapping)
            _stage_uploaded(workspace, data, files, mapping)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _container_upload(workspace):
    name = _new_dataset_name(workspace, "uploaded-dataset", "container_name")
    st.write("Use this mode when several arrays are stored together in NPZ or MATLAB v7 files.")
    file = st.file_uploader("Container file", type=["npz", "mat"], key="container_file")
    if not file:
        return
    try:
        arrays = _read_uploads([file])
    except Exception as exc:
        st.error(str(exc))
        return
    table = [{"array": key, "shape": str(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    keys = list(arrays)
    views = st.multiselect("Feature arrays", keys, key="container_views")
    label_key = st.selectbox("Labels (optional)", [None, *keys], key="container_labels")
    transpose = st.multiselect("Arrays stored as features × samples", views, key="container_transpose")
    align = st.checkbox("Views have different rows and include sample-ID arrays", key="container_align")
    id_keys = []
    label_id = None
    graphs, modalities = {}, []
    with st.expander("Advanced mapping: sample IDs, modalities, and graph relations"):
        for index, view in enumerate(views):
            st.markdown(f"**View {index + 1}: `{view}`**")
            modalities.append(st.text_input("Modality", "features", key=f"modality_{index}_{view}"))
            graphs[view] = st.multiselect(
                "Graph relations for this view", [key for key in keys if key != view], key=f"graphs_{index}_{view}"
            )
            if align:
                id_keys.append(st.selectbox("Sample IDs for this view", keys, key=f"ids_{index}_{view}"))
        if align and label_key:
            label_id = st.selectbox("Sample IDs for labels", keys, key="container_label_ids")
    if st.button("Add container dataset", type="primary", disabled=not views):
        try:
            mapping = {
                "view_keys": views,
                "graph_keys": graphs,
                "label_key": label_key,
                "id_keys": id_keys or None,
                "label_id_key": label_id,
                "transpose_views": transpose,
                "modalities": modalities,
                "name": name,
            }
            data = dataset_from_arrays(arrays, **mapping)
            _stage_uploaded(workspace, data, [file], mapping)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def _synthetic_graph_controls(samples, views):
    with st.expander("Synthetic graphs", expanded=True):
        mode = st.selectbox("Graph generation", ["No graphs", "Per-view graphs", "Shared graphs"])
        if mode == "No graphs":
            return [], None, None
        targets = st.multiselect("Views receiving graphs", list(range(views)), default=list(range(views)),
                                 format_func=lambda i: f"View {i + 1}", key=f"synthetic_graph_views_{views}")
        source = None
        if mode == "Shared graphs":
            source = st.selectbox("Build shared graphs from", list(range(views)),
                                  format_func=lambda i: f"View {i + 1}")
            st.caption("The same relations are attached to each selected view. "
                       "Edges use only the source view's features.")
        else:
            st.caption("Each selected view gets its own graphs, built from that view's features.")
        count = st.number_input("Relations per selected view", 1, 5, 1)
        kinds = {"Binary kNN": "knn", "Mutual kNN": "mutual_knn", "Weighted kNN (RBF)": "rbf_knn"}
        graphs = []
        for relation in range(count):
            st.markdown(f"Relation {relation + 1}")
            left, right = st.columns(2)
            kind = left.selectbox("Graph type", list(kinds), key=f"synthetic_graph_kind_{relation}")
            neighbors = right.number_input("Neighbors", 1, samples - 1, min(10 + 2 * relation, samples - 1),
                                            key=f"synthetic_graph_neighbors_{relation}")
            bandwidth = 1.0
            if kinds[kind] == "rbf_knn":
                bandwidth = st.number_input("RBF bandwidth", min_value=0.0001, value=1.0, step=0.1,
                                             key=f"synthetic_graph_bandwidth_{relation}",
                                             help="Feature-space length scale. Larger values give distant neighbors "
                                                  "more weight.")
            graphs.append(SyntheticGraphSpec(kinds[kind], neighbors, bandwidth))
        st.caption("Binary/weighted kNN connects either neighbor direction; mutual kNN requires both. "
                   "All graphs are sparse, symmetric and have no self-loops. Mutual graphs may have isolated nodes. "
                   "Generation is limited to 2 million directed neighbors across all attached relations.")
        st.caption(f"{len(targets) * count} graph relation(s) attached to {len(targets)} view(s). "
                   "Choose one relation on one view for a single graph, or shared graphs to reuse it across views.")
        return graphs, targets, source


def _synthetic_loader(workspace):
    name = _new_dataset_name(workspace, "synthetic-dataset", "synthetic_name")
    first, second, third = st.columns(3)
    samples = first.number_input("Samples", 20, 100000, 300, step=10)
    classes = second.number_input("Classes", 2, min(500, samples), min(3, samples), step=1)
    views = third.number_input("Views", 1, 12, 2, step=1)
    complexity = st.select_slider("Class overlap", ["Low", "Moderate", "High"], value="Moderate")
    defaults = {"Low": (5.0, 0.4), "Moderate": (3.0, 0.7), "High": (1.6, 1.1)}[complexity]
    separation_column, noise_column = st.columns(2)
    separation = separation_column.number_input("Class separation", 0.1, 20.0, defaults[0], step=0.1)
    noise = noise_column.number_input("Within-class noise", 0.01, 10.0, defaults[1], step=0.05)
    dimensions, distributions = [], []
    with st.expander("View dimensions and distributions", expanded=True):
        for view in range(views):
            left, right = st.columns(2)
            dimensions.append(left.number_input("Features", 2, 10000, 10, key=f"synthetic_dim_{view}"))
            distributions.append(
                right.selectbox(
                    "Distribution", ["Gaussian", "Anisotropic", "Heavy-tailed", "Nonlinear"],
                    key=f"synthetic_distribution_{view}",
                )
            )
    balance = st.selectbox("Class balance", ["Balanced", "Moderately imbalanced", "Strongly imbalanced"])
    same_modality = st.radio("View semantics", ["Same modality", "Different modalities"], horizontal=True)
    labeled = st.checkbox("Include ground-truth labels", value=True)
    graphs, graph_views, shared_graph_view = _synthetic_graph_controls(samples, views)
    seed = st.number_input("Random seed", 0, 2**31 - 1, 7)
    if st.button("Create synthetic dataset", type="primary"):
        try:
            spec = SyntheticSpec(
                n_samples=samples,
                n_classes=classes,
                dimensions=dimensions,
                distributions=distributions,
                class_balance=balance,
                separation=separation,
                noise=noise,
                same_modality=same_modality == "Same modality",
                labeled=labeled,
                graph_specs=graphs,
                graph_views=graph_views,
                shared_graph_view=shared_graph_view,
                random_state=seed,
            )
            with st.spinner("Generating features and sparse graphs"):
                data = generate_synthetic(spec)
            key = workspace.add_lazy_dataset(
                name, lambda recipe=copy.deepcopy(spec): generate_synthetic(recipe), source="synthetic"
            )
            workspace.datasets[key]._data = data
            workspace.datasets[key].hint = {
                "n_samples": data.n_samples,
                "n_views": data.n_views,
                "n_clusters": data.n_clusters,
                "complete": data.is_complete,
                "category": data.category,
            }
            _loaded_dataset(workspace, key, "synthetic-dataset")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_dataset(workspace):
    st.header("Dataset")
    st.write("Load a catalog dataset, upload your own aligned views, or generate a controlled example.")
    source = st.radio("Data source", ["Catalog", "Files", "Synthetic"], horizontal=True)
    if source == "Catalog":
        _catalog_loader(workspace)
    elif source == "Files":
        mode = st.radio("File organization", ["One file per view", "NPZ / MATLAB container"], horizontal=True)
        _simple_upload(workspace) if mode == "One file per view" else _container_upload(workspace)
    else:
        _synthetic_loader(workspace)
    if not workspace.datasets:
        return
    st.divider()
    st.subheader("Dataset collection")
    dataset_id = _dataset_picker(workspace, "dataset_browser")
    resource = workspace.datasets[dataset_id]
    left, right = st.columns(2)
    if left.button("Unload", disabled=not resource.loaded, width="stretch"):
        workspace.unload_dataset(dataset_id)
        st.session_state.suspended_dataset = dataset_id
        st.rerun()
    if right.button("Remove dataset", width="stretch"):
        workspace.remove_dataset(dataset_id)
        st.session_state.selected_dataset = None
        st.rerun()
    dataset_id, data = _select_dataset(workspace, required=False)
    if data is not None:
        st.subheader(data.name or "Selected dataset")
        with st.expander("Rename dataset"):
            new_name = st.text_input("New dataset name", resource.name, key=f"rename_{dataset_id}")
            if st.button("Save dataset name"):
                try:
                    workspace.rename_dataset(dataset_id, new_name)
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        workspace.benchmark[dataset_id] = st.checkbox(
            "Benchmark mode: evaluate against ground-truth labels",
            value=workspace.benchmark_mode(dataset_id), disabled=data.labels is None,
            key=f"benchmark_{dataset_id}",
        )
        st.caption("Labels are used only for evaluation, never for fitting. Uncheck for a real-world experiment.")
        profile = resource.hint.get("prepared_profile")
        if profile:
            inspect_mode = st.radio("Inspect", ["Prepared data", "Original data"], horizontal=True,
                                    key=f"inspect_{dataset_id}")
            if inspect_mode == "Original data":
                profile = None
        else:
            st.caption("Loaded source details · apply changes to prepare it")
        _profile(data, profile)
        with st.expander("Add graphs to a view"):
            st.write("Upload adjacency matrices (NPY, NPZ, CSV, TSV) or graph files (GraphML, GEXF). "
                     "Each file becomes one additional relation for the selected view.")
            st.caption(f"Matrices must follow sample order. Graph node IDs must match sample IDs, "
                       f"including isolated samples. First IDs: {data.sample_ids[:5].tolist()}")
            view = st.selectbox("Graph belongs to view", list(range(data.n_views)),
                                 format_func=lambda index: data.view_names[index], key=f"graph_view_{dataset_id}")
            files = st.file_uploader("Graph files", type=["npy", "npz", "csv", "tsv", "graphml", "gexf"],
                                      accept_multiple_files=True, key=f"graph_files_{dataset_id}")
            if st.button("Attach graphs", disabled=not files):
                from pymv.display.graph_uploads import attach_graphs, read_graph_file

                try:
                    graphs = [read_graph_file(file.getvalue(), file.name, data.sample_ids) for file in files]
                    attach_graphs(workspace, dataset_id, view, graphs)
                    workspace.pending_datasets.add(dataset_id)
                    resource.hint.pop("prepared_profile", None)
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.divider()
        render_preprocessing(workspace)


def _view_controls(prefix, *, has_graph, value=None):
    value = value or ViewPreprocessing()
    normalize = st.checkbox("L2-normalize rows", value=value.normalize, key=f"{prefix}_normalize")
    tf_idf = st.checkbox("TF-IDF + L2", value=value.tf_idf, key=f"{prefix}_tfidf")
    center = st.checkbox("Center columns", value=value.center, key=f"{prefix}_center")
    scale = st.checkbox("Scale to unit variance", value=value.scale, key=f"{prefix}_scale")
    propagate = st.checkbox(
        "Propagate through graph", value=value.propagate, disabled=not has_graph, key=f"{prefix}_propagate"
    )
    beta = st.number_input("Graph self-loop weight", 0.0, 100.0, float(value.beta), key=f"{prefix}_beta")
    return ViewPreprocessing(normalize, tf_idf, center, scale, propagate, beta)


def render_preprocessing(workspace):
    st.subheader("Preparation")
    dataset_id, data = _select_dataset(workspace)
    if data is None:
        return
    current = workspace.preprocessing.get(dataset_id, PreprocessingSpec())
    st.caption("Apply changes to make this preparation available in experiments and searches. "
               "Each preparation starts from the original data, so transformations are not applied twice.")
    st.subheader("Missing views")
    missing_rate = st.slider(
        "Simulate an additional missing-view fraction",
        0.0,
        0.95,
        float(current.missing_rate if data.n_views > 1 else 0.0),
        0.05,
        disabled=data.n_views == 1,
        help="Missing-view simulation needs at least two views so every sample retains one observation.",
        key=f"missing_rate_{dataset_id}",
    )
    imputation_label = st.selectbox(
        "Explicit imputation",
        ["Keep missing views", "Mean imputation", "Zero imputation"],
        index={None: 0, "mean": 1, "zero": 2}.get(current.imputation, 0),
        key=f"imputation_{dataset_id}",
    )
    imputation = {"Keep missing views": None, "Mean imputation": "mean", "Zero imputation": "zero"}[
        imputation_label
    ]
    if (not data.is_complete or missing_rate) and imputation is None:
        st.info("This dataset can only be paired with missing-view models unless you select imputation.")

    st.subheader("Feature transformations")
    mode_label = st.radio(
        "Apply transformations to",
        ["No views", "All views", "Selected views", "Configure each view"],
        index={"none": 0, "all": 1, "selected": 2, "per_view": 3}.get(current.mode, 0),
        horizontal=True,
        key=f"preprocessing_mode_{dataset_id}",
    )
    mode = {"No views": "none", "All views": "all", "Selected views": "selected", "Configure each view": "per_view"}[
        mode_label
    ]
    selected_views, shared, per_view = [], current.shared, {}
    if mode == "selected":
        selected_names = st.multiselect(
            "Views", data.view_names,
            default=[data.view_names[index] for index in current.selected_views if index < data.n_views],
            key=f"preprocessing_views_{dataset_id}",
        )
        selected_views = [data.view_names.index(name) for name in selected_names]
    if mode in {"all", "selected"}:
        with st.container(border=True):
            shared = _view_controls(
                f"shared_{dataset_id}", has_graph=any(graph is not None for graph in data.As), value=current.shared
            )
    elif mode == "per_view":
        for index, name in enumerate(data.view_names):
            with st.expander(name, expanded=index == 0):
                per_view[index] = _view_controls(
                    f"view_{dataset_id}_{index}", has_graph=data.As[index] is not None,
                    value=current.per_view.get(index),
                )

    fusion = current.fusion
    weights = {}
    multi_relations = [(index, graph) for index, graph in enumerate(data.As) if isinstance(graph, (list, tuple))]
    if multi_relations:
        st.subheader("Multiple graph relations")
        fusion = st.radio(
            "Early fusion", ["sum", "hadamard"], index=0 if current.fusion == "sum" else 1, horizontal=True,
            key=f"fusion_{dataset_id}",
        )
        if fusion == "sum":
            for view, graphs in multi_relations:
                st.caption(f"Weights for {data.view_names[view]}")
                columns = st.columns(min(len(graphs), 4))
                values = []
                old = current.relation_weights.get(view, [1.0] * len(graphs))
                for relation in range(len(graphs)):
                    values.append(
                        columns[relation % len(columns)].number_input(
                            f"Relation {relation + 1}", 0.0, 1000.0,
                            float(old[relation] if relation < len(old) else 1.0),
                            key=f"weight_{dataset_id}_{view}_{relation}",
                        )
                    )
                weights[view] = values
    recipe = PreprocessingSpec(
        mode=mode, selected_views=selected_views, shared=shared, per_view=per_view,
        missing_rate=missing_rate, missing_seed=0, imputation=imputation,
        fusion=fusion, relation_weights=weights,
    )
    if st.button("Apply changes", type="primary"):
        try:
            workspace.apply_dataset(dataset_id, recipe)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    if dataset_id not in workspace.pending_datasets:
        st.success("Saved preparation is ready for experiments and searches.")


def _draft(workspace, dataset_id, data):
    if "drafts" not in st.session_state:
        st.session_state.drafts = {}
    if dataset_id not in st.session_state.drafts:
        default_k = (data.n_clusters or min(3, data.n_samples)) if data is not None else 3
        st.session_state.drafts[dataset_id] = {
            "model": None,
            "params": {},
            "search_space": {},
            "trials": 1,
            "clusters": [default_k],
            "repeats": 1,
            "seed": 0,
            "benchmark": False,
            "metrics": list(INTERNAL_NAMES),
        }
    return st.session_state.drafts[dataset_id]


def _parameter_input(model, name, control, default, existing, *, disabled=False):
    key = f"parameter_{model}_{name}"
    value = existing.get(name, default)
    if control.kind == "bool":
        return st.checkbox(control.label, value=bool(value), key=key, help=control.help, disabled=disabled)
    if control.kind == "int":
        return st.number_input(
            control.label, min_value=int(control.minimum) if control.minimum is not None else None,
            max_value=int(control.maximum) if control.maximum is not None else None,
            value=int(value), step=int(control.step or 1), key=key, help=control.help, disabled=disabled,
        )
    if control.kind == "float":
        return st.number_input(
            control.label, min_value=float(control.minimum) if control.minimum is not None else None,
            max_value=float(control.maximum) if control.maximum is not None else None,
            value=float(value), step=float(control.step or 0.1), key=key, help=control.help, disabled=disabled,
        )
    if control.kind == "int_tuple":
        text = st.text_input(control.label, ", ".join(map(str, value)), key=key, help=control.help, disabled=disabled)
        return parse_int_tuple(text)
    return st.text_input(control.label, str(value), key=key, help=control.help, disabled=disabled)


def render_model(workspace, *, search=False):
    st.header("Hyperparameter search" if search else "Model")
    dataset_id, data = "model_library", None
    if search:
        if _dataset_picker(workspace, "search_dataset", prepared_only=True) is None:
            return
        dataset_id, data = _select_dataset(workspace)
        if data is None:
            return
    else:
        st.caption("Prepare named model configurations independently of datasets. "
                   "Compatibility is checked when you select experiment combinations.")
    recipe = workspace.preprocessing.get(dataset_id, PreprocessingSpec()) if search else PreprocessingSpec()
    draft = _draft(workspace, dataset_id, data)
    available, unavailable = compatible_models(data, recipe) if search else (list_models(), [])
    if unavailable:
        st.warning("Unavailable until missing views are imputed: " + ", ".join(name.upper() for name in unavailable))
    if not available:
        st.error("No registered model can use this incomplete dataset. Select imputation on the Dataset page.")
        return
    index = available.index(draft["model"]) if draft["model"] in available else 0
    model = st.selectbox("Model", available, index=index, format_func=str.upper)
    if not search:
        config_name = st.text_input("Configuration name", workspace.suggested_name(model.upper()),
                                    key=f"model_name_{model}_{workspace.name_counters.get(model.upper(), 0)}")
    info = get_model_info(model)
    st.write(info.get("description", ""))
    if info.get("paper"):
        st.caption(f"Original paper: {info['paper']}")
    defaults, controls = parameter_controls(model)
    params = {}
    validation_errors = []
    external_preprocessing = recipe.changes_features
    if external_preprocessing:
        st.info("Model-internal preprocessing is disabled to avoid applying transformations twice.")
    with st.container(border=True):
        columns = st.columns(2)
        for position, (name, control) in enumerate(controls.items()):
            with columns[position % 2]:
                if name == "preprocess" and external_preprocessing:
                    params[name] = False
                    st.checkbox(control.label, value=False, disabled=True, key=f"forced_{model}_{name}")
                else:
                    try:
                        params[name] = _parameter_input(model, name, control, defaults[name], draft["params"])
                        if control.kind in {"int", "float"}:
                            params[name] = parse_candidates([params[name]], control)[0]
                    except ValueError as exc:
                        validation_errors.append(str(exc))

    st.subheader("Number of clusters")
    known = data.n_clusters if data is not None else None
    limit = data.n_samples if data is not None else None
    if known is not None:
        st.caption(f"The dataset has {known} ground-truth classes. K is editable and is never inferred during fitting.")
    primary = st.number_input("K", 1, limit, int(draft["clusters"][0]), key=f"clusters_primary_{dataset_id}")
    compare = st.checkbox("Compare several K values", value=len(draft["clusters"]) > 1)
    clusters = [primary]
    if compare:
        saved = [str(value) for value in draft["clusters"][1:]]
        suggestions = list(dict.fromkeys([*map(str, range(2, min(limit or 20, 20) + 1)), *saved]))
        selected = st.multiselect(
            "Additional K values", suggestions,
            default=saved,
            accept_new_options=True,
        )
        if selected:
            try:
                clusters.extend(parse_candidates(
                    selected, ParameterControl("K", "int", minimum=1, maximum=limit)
                ))
            except ValueError as exc:
                validation_errors.append(str(exc))
    clusters = list(dict.fromkeys(clusters))

    strategy = st.selectbox("Search strategy", ["Random search", "Grid search"]) if search else "Fixed parameters"
    search_space = {}
    trials = 1
    if search:
        st.caption("Select suggestions or type any valid number and press Enter to add it. Each value is checked.")
        for name, control in controls.items():
            if not control.search_values or (name == "preprocess" and external_preprocessing):
                continue
            if st.checkbox(
                f"Search {control.label}", value=name in draft["search_space"], key=f"search_{model}_{name}"
            ):
                saved = [str(value) for value in draft["search_space"].get(name, control.search_values[:2])]
                options = list(dict.fromkeys([*map(str, control.search_values), *saved]))
                candidates = st.multiselect(
                    f"Candidates for {control.label}", options,
                    default=saved,
                    accept_new_options=True, key=f"candidates_{model}_{name}",
                )
                try:
                    search_space[name] = parse_candidates(candidates, control)
                except ValueError as exc:
                    validation_errors.append(str(exc))
        if not search_space and not validation_errors:
            validation_errors.append("Select at least one parameter to search.")
        if strategy == "Random search":
            trials = st.number_input("Random configurations", 1, 1000, int(draft.get("trials", 1)))
    for message in validation_errors:
        st.error(message)
    if search:
        draft.update(search_space=search_space, trials=trials)
        _search_runner(workspace, dataset_id, data, model, params, clusters, search_space,
                       trials, strategy, validation_errors)
        return
    if not config_name.strip():
        validation_errors.append("Choose a nonempty model name.")
        st.error(validation_errors[-1])

    def save_model():
        draft.update(model=model, params=params, search_space={}, trials=1, clusters=clusters)
        existing = next((key for key, value in workspace.models.items()
                         if value["name"] == config_name.strip() and value["model"] == model
                         and value["params"] == params
                         and value["clusters"] == clusters), None)
        key = existing or workspace.add_model(config_name, model, params, clusters)
        if not existing:
            workspace.advance_name(model.upper())
        st.session_state.preferred_model = key

    if st.button("Save model settings", type="primary", disabled=bool(validation_errors)):
        save_model()
        st.rerun()
    if st.button("Continue to experiment", disabled=bool(validation_errors)):
        save_model()
        _set_page("Experiment")
        st.rerun()
    if workspace.models:
        st.subheader("Saved model configurations")
        st.dataframe(pd.DataFrame([{"name": item["name"], "model": item["model"].upper(),
                                   "K": str(item["clusters"]), **item["params"]}
                                  for item in workspace.models.values()]), hide_index=True)
        selected = st.selectbox("Configuration to remove", list(workspace.models),
                                format_func=lambda key: workspace.models[key]["name"])
        if st.button("Remove configuration"):
            workspace.models.pop(selected)
            st.rerun()


def render_experiment(workspace):
    st.header("Experiment")
    ready = [key for key in workspace.datasets if key not in workspace.pending_datasets]
    if not ready or not workspace.models:
        st.info("Apply dataset preparation in Dataset and save configurations in Model. Either can be prepared first.")
        return
    active = st.session_state.get("selected_dataset")
    datasets = st.multiselect("Datasets", ready,
                              default=[active] if active in ready else [],
                              format_func=lambda key: workspace.datasets[key].name)
    preferred = st.session_state.get("preferred_model", next(reversed(workspace.models)))
    models = st.multiselect("Model configurations", list(workspace.models),
                            default=[preferred] if preferred in workspace.models else [],
                            format_func=lambda key: workspace.models[key]['name'])
    clusters_by_dataset = {}
    dataset_defaults = st.checkbox("Use dataset class counts for benchmark datasets", value=True,
                                   key="dataset_class_counts",
                                   help="Uncheck to use the K values saved with each model, including K comparisons.")
    if dataset_defaults:
        for key in datasets:
            if workspace.benchmark_mode(key):
                resource = workspace.datasets[key]
                known = resource.hint.get("n_clusters") or workspace.load_dataset(key).n_clusters
                clusters_by_dataset[key] = [st.number_input(
                    f"K for {resource.name}", 1, resource.hint.get("n_samples"), int(known),
                    key=f"experiment_dataset_k_{key}",
                )]
    compatible = set(workspace.compatible_pairs(datasets, models, clusters_by_dataset=clusters_by_dataset))
    mode = st.radio("Combination selection", ["All compatible pairs", "Choose in table"], horizontal=True)
    selected = [(dataset, model) for dataset in datasets for model in models if (dataset, model) in compatible]
    if datasets and models and mode == "Choose in table":
        # Human names are unique within each collection. Positional column keys
        # avoid collisions with arbitrary user names such as 'Dataset'.
        table = pd.DataFrame([
            {"dataset": workspace.datasets[d].name,
             **{f"model_{index}": (True if (d, m) in compatible else None) for index, m in enumerate(models)}}
            for d in datasets
        ])
        edited = st.data_editor(
            table, hide_index=True, width="stretch", disabled=["dataset"],
            column_config={"dataset": st.column_config.TextColumn("Dataset"),
                           **{f"model_{i}": st.column_config.CheckboxColumn(workspace.models[m]["name"])
                              for i, m in enumerate(models)}},
            key="pair_matrix_" + "_".join(datasets + models),
        )
        selected = [(d, m) for row, d in enumerate(datasets) for i, m in enumerate(models)
                    if (d, m) in compatible and bool(edited.iloc[row][f"model_{i}"])]
    if len(compatible) < len(datasets) * len(models):
        st.info("Incompatible pairs are excluded (missing-view support or K exceeds the number of samples). "
                "Blank cells cannot enable an incompatible pair.")
    modes = {workspace.benchmark_mode(key) for key, _ in selected}
    mixed = len(modes) > 1
    if mixed:
        st.error("Benchmark and real-world datasets cannot be combined. "
                 "Change their mode in Dataset or select a subset.")
    benchmark = modes == {True}
    st.caption("Benchmark mode" if benchmark else "Real-world mode · internal metrics only")
    name = st.text_input("Experiment name", f"Experiment {workspace.experiment_counter + 1}",
                         key=f"experiment_name_{workspace.experiment_counter}")
    repeats = st.slider("Independent runs per configuration", 1, 20, 1)
    with st.expander("Reproducibility and saving"):
        seed = st.number_input("Base random seed", 0, 2**31 - 21, 0,
                               help="Run r uses seed + r − 1. Keep this value to reproduce the runs.")
        output_dir = st.text_input("Output directory", "experiments", key="experiment_output_dir")
    available_metrics = INTERNAL_NAMES + (EXTERNAL_NAMES if benchmark else [])
    metrics = st.multiselect("Metrics", available_metrics, default=available_metrics,
                             key=f"experiment_metrics_{benchmark}")
    total = sum(len(clusters_by_dataset.get(d, workspace.models[m]["clusters"])) * repeats for d, m in selected)
    st.caption(f"{len(selected)} dataset/model combinations · {total} fits, executed one at a time.")
    if total > 2000:
        st.warning("Reduce the selected combinations, K values or repeats to at most 2,000 fits.")
    if st.button("Run experiment", type="primary", disabled=mixed or not metrics or not selected or total > 2000):
        specs = []
        for dataset_id, model_id in selected:
            settings = workspace.models[model_id]
            specs.append(ExperimentSpec(
                dataset_id=dataset_id, model=settings["model"], model_name=settings["name"],
                name=name, params=settings["params"],
                clusters=clusters_by_dataset.get(dataset_id, settings["clusters"]),
                repeats=repeats, seed=seed, benchmark=benchmark, metrics=metrics,
                preprocessing=workspace.preprocessing.get(dataset_id, PreprocessingSpec()),
            ))
        progress = st.progress(0.0, text="Running fits")
        try:
            with st.spinner("Fitting models"):
                workspace.run_batch(specs, name=name, output_dir=output_dir, progress=progress.progress)
            _set_page("Results")
            st.rerun()
        except Exception as exc:
            progress.empty()
            st.error(str(exc))


def _downloads(record, *, prefix):
    filename = record_download_stem(record)
    columns = st.columns(3)
    columns[0].download_button("Results JSON", report_json(record.report), f"{filename}.json",
                               "application/json", key=f"{prefix}_json")
    columns[1].download_button("Metrics CSV", report_csv(record.report), f"{filename}.csv",
                               "text/csv", key=f"{prefix}_csv")
    cached = st.session_state.get("_prepared_record_download")
    if cached is None or cached["id"] != record.id:
        if not columns[2].button("Prepare ZIP", key=f"{prefix}_prepare"):
            return
        with st.spinner("Compressing run arrays for download"):
            cached = {"id": record.id, "zip": export_record(record)}
            st.session_state._prepared_record_download = cached
    columns[2].download_button("Complete ZIP", cached["zip"], f"{filename}.zip",
                               "application/zip", key=f"{prefix}_zip")


def _adjustment_notice(record):
    changes = sorted({f"{key}: {values['requested']} → {values['effective']}"
                      for row in record.rows for key, values in row.get("parameter_adjustments", {}).items()})
    if changes:
        st.info("Adjusted for this dataset: " + "; ".join(changes)
                + ". Tables and exports show the values actually used. Your saved model is unchanged.")


def _search_runner(workspace, dataset_id, data, model, params, clusters, search_space,
                   trials, strategy, errors):
    import math

    benchmark = workspace.benchmark_mode(dataset_id)
    available_metrics = INTERNAL_NAMES + (EXTERNAL_NAMES if benchmark else [])
    metrics = st.multiselect("Search metrics", available_metrics, default=available_metrics,
                             key=f"search_metrics_{benchmark}")
    if not metrics:
        st.info("Select at least one metric.")
        return
    objective = st.selectbox("Optimize metric", metrics)
    st.caption("The best configuration optimizes the mean score across repeated runs. Davies–Bouldin is minimized.")
    repeats = st.slider("Independent runs per configuration", 1, 20, 1, key="search_repeats")
    with st.expander("Reproducibility and saving"):
        seed = st.number_input("Base random seed", 0, 2**31 - 21, 0, key="search_seed")
        output_dir = st.text_input("Output directory", "experiments", key="search_output_dir")
    count = trials if strategy == "Random search" else math.prod(len(values) for values in search_space.values())
    total = count * len(clusters) * repeats
    st.caption(f"{count * len(clusters)} configurations × {repeats} runs = {total} fits.")
    if total > 2000:
        st.warning("Reduce the search space to at most 2,000 fits per session job.")
    if st.button("Run search", type="primary", disabled=bool(errors) or total > 2000):
        draft = _draft(workspace, dataset_id, data)
        draft.update(model=model, params=params, clusters=clusters, search_space=search_space, trials=trials)
        spec = ExperimentSpec(
            dataset_id, model=model, name=f"Search {sum(r.spec.kind == 'search' for r in workspace.records) + 1}",
            params=params, clusters=clusters, repeats=repeats, seed=seed, benchmark=benchmark, metrics=metrics,
            search_space=search_space, trials=trials, kind="search", objective=objective,
            search_strategy="random" if strategy == "Random search" else "grid",
            preprocessing=workspace.preprocessing.get(dataset_id, PreprocessingSpec()),
        )
        progress = st.progress(0.0, text="Searching")
        try:
            workspace.run(spec, progress=progress.progress, output_dir=output_dir)
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_search(workspace):
    render_model(workspace, search=True)
    st.divider()
    st.subheader("Search results")
    searches = [record for record in workspace.records if record.spec.kind == "search"]
    if not searches:
        st.caption("Completed searches will appear here, independently of your experiments.")
    for position, record in enumerate(searches):
        with st.expander(f"{record.spec.name} · {record.spec.model.upper()} · {record.dataset_summary['name']}",
                         expanded=position == 0):
            search = record.report["search"]
            st.caption(f"{search['strategy'].title()} search · {search['direction']} {search['objective']}")
            st.dataframe(pd.DataFrame(aggregate_results(record)), hide_index=True, width="stretch")
            _adjustment_notice(record)
            with st.expander("Tuning steps and detailed runs"):
                st.dataframe(pd.DataFrame([{k: v for k, v in step.items() if k != "parameters"}
                                           for step in search["steps"]]), hide_index=True)
                st.dataframe(pd.DataFrame(detailed_rows(record)), hide_index=True)
            if search["best_configuration"] is None:
                st.warning("No fully successful configuration has a finite objective score.")
            else:
                st.success(f"Best {search['objective']}: {search['best_score']:.4f}")
            st.divider()
            prefix = f"{record.spec.model.upper()}_{search['strategy']}_best_"
            best_name = st.text_input("Best model name", workspace.suggested_name(prefix),
                                      key=f"best_name_{record.id}_{workspace.name_counters.get(prefix, 0)}")
            actions = st.columns(2)
            if actions[0].button("Copy best to experiment", key=f"copy_best_{record.id}",
                         disabled=search["best_configuration"] is None
                         or record.spec.dataset_id not in workspace.datasets or not best_name.strip(),
                          width="stretch"):
                params = copy.deepcopy(search["best_parameters"])
                clusters = [params.pop("n_clusters")]
                st.session_state.dataset_class_counts = False
                params.pop("random_state", None)
                key = workspace.add_model(best_name, record.spec.model, params, clusters)
                workspace.advance_name(prefix)
                workspace.preprocessing[record.spec.dataset_id] = copy.deepcopy(record.spec.preprocessing)
                workspace.pending_datasets.discard(record.spec.dataset_id)
                workspace.benchmark[record.spec.dataset_id] = record.spec.benchmark
                st.session_state[f"benchmark_{record.spec.dataset_id}"] = record.spec.benchmark
                st.session_state.selected_dataset = record.spec.dataset_id
                st.session_state.preferred_model = key
                _set_page("Experiment")
                st.rerun()
            if actions[1].button("Delete search result", key=f"delete_search_{record.id}", width="stretch"):
                workspace.delete_record(record.id)
                if st.session_state.get("_prepared_record_download", {}).get("id") == record.id:
                    st.session_state.pop("_prepared_record_download")
                for key in list(st.session_state):
                    if key.startswith(f"projection_{record.id}_"):
                        del st.session_state[key]
                st.rerun()
            if st.button("Open Export", key=f"export_search_{record.id}"):
                _set_page("Export")
                st.rerun()


def _record_selector(workspace, key):
    if not workspace.records:
        st.info("No experiments have been run in this session.")
        return None
    labels = {record.id: f"{record_label(record)} · {record.created_at[:19]} · Result {len(workspace.records) - i}"
              for i, record in enumerate(workspace.records)}
    record_id = st.selectbox(
        "Experiment",
        [record.id for record in workspace.records],
        format_func=lambda value: labels[value],
        key=key,
    )
    return next(record for record in workspace.records if record.id == record_id)


def render_results(workspace):
    st.header("Results")
    if not workspace.records:
        st.info("No experiments have been run in this session.")
        return
    st.caption("Newest first. Completed runs remain available until you remove their result.")
    for position, record in enumerate([r for r in workspace.records if r.spec.kind == "experiment"]):
        title = f"{record_label(record)} · {record.created_at[:19]}"
        with st.expander(title, expanded=position == 0):
            st.caption(
                f"{record.dataset_summary['name']} · {len(record.rows)} fits"
            )
            st.subheader("Repeated-run summary")
            _adjustment_notice(record)
            st.dataframe(pd.DataFrame(aggregate_results(record)), hide_index=True, width="stretch")
            failed = [row for row in record.rows if row["status"] == "failed"]
            if failed:
                st.warning(f"{len(failed)} fit(s) failed. Expand detailed runs to inspect their errors.")
            with st.expander("Detailed runs"):
                st.dataframe(
                    pd.DataFrame(detailed_rows(record)),
                    hide_index=True,
                    width="stretch",
                )
            st.divider()
            st.caption("Result actions")
            left, right = st.columns(2)
            if left.button("Use these settings for a new experiment", key=f"duplicate_{record.id}",
                           width="stretch"):
                dataset_id = record.spec.dataset_id
                if dataset_id in workspace.datasets:
                    draft = _draft(workspace, "model_library", None)
                    draft.update(
                        model=record.spec.model,
                        params=copy.deepcopy(record.spec.params),
                        clusters=copy.deepcopy(record.spec.clusters),
                        repeats=record.spec.repeats,
                        seed=record.spec.seed,
                        benchmark=record.spec.benchmark,
                        metrics=copy.deepcopy(record.spec.metrics),
                        search_space=copy.deepcopy(record.spec.search_space),
                        trials=record.spec.trials,
                    )
                    workspace.preprocessing[dataset_id] = copy.deepcopy(
                        record.spec.preprocessing or PreprocessingSpec()
                    )
                    st.session_state.selected_dataset = dataset_id
                    st.session_state._restore_model = copy.deepcopy(draft)
                    st.session_state.dataset_class_counts = False
                    _set_page("Model")
                    st.rerun()
                else:
                    st.error(
                        "The source dataset was removed. Results remain available, "
                        "but the experiment cannot be rerun."
                    )
            if right.button("Delete this result", key=f"delete_{record.id}", width="stretch"):
                workspace.delete_record(record.id)
                if st.session_state.get("_prepared_record_download", {}).get("id") == record.id:
                    st.session_state.pop("_prepared_record_download")
                for key in list(st.session_state):
                    if key.startswith(f"projection_{record.id}_"):
                        del st.session_state[key]
                st.rerun()
            if st.button("Open Export", key=f"export_result_{record.id}"):
                _set_page("Export")
                st.rerun()


def _scatter_figure(projected, labels, title, method):
    import plotly.express as px

    labels = np.asarray(labels)
    unique = np.unique(labels)
    frame = pd.DataFrame({"x": projected[:, 0], "y": projected[:, 1], "label": labels.astype(str)})
    if len(unique) <= 20:
        figure = px.scatter(
            frame, x="x", y="y", color="label", title=title,
            color_discrete_sequence=["#8A1538", "#376D72", "#64748B", "#947D64", "#1E3A5F"],
        )
    else:
        codes, _ = pd.factorize(frame["label"], sort=True)
        frame["class index"] = codes
        figure = px.scatter(
            frame, x="x", y="y", color="class index", title=title, color_continuous_scale="Viridis"
        )
        figure.update_layout(coloraxis_colorbar_title="Class index")
    figure.update_traces(marker={"size": 6, "opacity": 0.75})
    figure.update_layout(
        legend_title_text="Class / cluster", xaxis_title=f"{method} 1", yaxis_title=f"{method} 2"
    )
    return figure


def _figure_downloads(figure, table, key, *, filename=None):
    """Save only on request; downloads are centralized on Export."""
    if st.button("Save plot to Export", key=f"save_plot_{key}"):
        workspace = _workspace()
        if key not in workspace.plots and len(workspace.plots) >= 100:
            st.warning("Remove unused saved plots in Export before adding more (limit: 100).")
            return
        try:
            snapshot = PlotSnapshot(filename or key, figure, table)
            if key in workspace.plots:
                workspace.plots[key].cleanup()
            workspace.plots[key] = snapshot
            st.success("Plot saved. Open Export → Visualizations to name and download it.")
        except ValueError as exc:
            st.error(str(exc))


def _loss_comparison(workspace):
    import plotly.graph_objects as go

    available = [(record.id, index) for record in workspace.records for index, artifact in enumerate(record.artifacts)
                 if "loss" in artifact and artifact["loss"].ndim == 1]
    if not available:
        return
    records = {record.id: record for record in workspace.records}

    def label(value):
        record = records[value[0]]
        title = f"{record_label(record)} · {run_label(record, value[1])}"
        # Distinguish repeated jobs with the same user names without leaking IDs.
        peers = [r for r in workspace.records if record_label(r) == record_label(record)]
        if len(peers) > 1:
            title += f" · Result {next(i for i, r in enumerate(reversed(peers), 1) if r.id == record.id)}"
        return title

    st.subheader("Compare normalized loss")
    selected = st.multiselect("Models and runs from any experiment", available,
                              default=[v for v in available if records[v[0]].rows[v[1]]["repeat"] == 1],
                              format_func=label)
    st.caption("Run 1 represents each configuration by default; select other runs to compare variability. "
               "Each trajectory uses (loss − minimum) / (maximum − minimum). "
               "Curves compare optimization shape, not objective quality across different methods.")
    if not selected:
        return
    figure = go.Figure()
    rows = []
    for value in selected:
        losses = normalized_loss(records[value[0]].artifacts[value[1]]["loss"])
        iterations = list(range(1, len(losses) + 1))
        figure.add_trace(go.Scatter(x=iterations, y=losses, mode="lines", name=label(value)))
        rows.extend({"run": label(value), "iteration": i, "normalized_loss": loss}
                    for i, loss in zip(iterations, losses))
    figure.update_layout(xaxis_title="Iteration", yaxis_title="Normalized loss", template="plotly_white",
                         height=500, showlegend=len(selected) <= 12)
    st.plotly_chart(figure, width="stretch")
    if len(selected) > 12:
        st.caption("Legend hidden for more than 12 curves; hover over a curve to identify its run.")
    _figure_downloads(figure, pd.DataFrame(rows), "loss-comparison")


def render_visualizations(workspace):
    import plotly.graph_objects as go

    st.header("Visualizations")
    _loss_comparison(workspace)
    record = _record_selector(workspace, "visual_record")
    if record is None:
        return
    successful = [index for index, artifact in enumerate(record.artifacts) if "labels" in artifact]
    if not successful:
        st.warning("This experiment has no successful fit to visualize.")
        return
    index = st.selectbox(
        "Run", successful,
        format_func=lambda value: run_label(record, value),
    )
    plot_stem = f"{record_download_stem(record)}_{download_stem(run_label(record, index))}"
    artifact = record.artifacts[index]
    _, labels = np.unique(artifact["labels"], return_inverse=True)
    st.caption("Predicted clusters use display IDs 0 to K−1. Raw model labels are preserved in result exports.")
    left, right = st.columns(2)
    with left:
        st.subheader("Predicted cluster sizes")
        counts = pd.Series(labels).value_counts().sort_index()
        figure = go.Figure(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color="#8A1538"))
        figure.update_layout(xaxis_type="category", xaxis_title="Predicted cluster", yaxis_title="Samples")
        st.plotly_chart(figure, width="stretch")
        _figure_downloads(figure, counts.rename_axis("cluster").reset_index(name="samples"),
                          f"sizes-{record.id}-{index}", filename=f"{plot_stem}_cluster-sizes")
    with right:
        st.subheader("Normalized optimization loss")
        if "loss" in artifact and artifact["loss"].ndim == 1:
            losses = normalized_loss(artifact["loss"])
            iterations = list(range(1, len(losses) + 1))
            figure = go.Figure(go.Scatter(x=iterations, y=losses, mode="lines", line_color="#8A1538"))
            figure.update_layout(xaxis_title="Iteration", yaxis_title="Normalized loss")
            st.plotly_chart(figure, width="stretch")
            _figure_downloads(figure, pd.DataFrame({"iteration": iterations, "normalized_loss": losses}),
                              f"loss-{record.id}-{index}", filename=f"{plot_stem}_loss")
        else:
            st.caption("This model did not expose a one-dimensional loss history.")
    st.subheader("Projection")
    if "embedding" not in artifact:
        st.caption("This model did not expose an embedding.")
    else:
        method = st.selectbox("Projection method", ["PCA", "t-SNE", "UMAP"])
        cache_key = f"projection_{record.id}_{index}_{method}"
        if st.button("Compute projection"):
            from pymv.display.projection import project_embedding

            try:
                st.session_state[cache_key] = project_embedding(
                    artifact["embedding"], method, random_state=record.spec.seed
                )
            except Exception as exc:
                st.error(str(exc))
        if cache_key in st.session_state:
            projected, rows = st.session_state[cache_key]
            columns = st.columns(2 if "true_labels" in artifact else 1)
            predicted_figure = _scatter_figure(projected, labels[rows], "Predicted clusters", method)
            columns[0].plotly_chart(predicted_figure, width="stretch")
            projection_table = pd.DataFrame({"sample_index": rows, "x": projected[:, 0], "y": projected[:, 1],
                                             "predicted_cluster": labels[rows]})
            _figure_downloads(predicted_figure, projection_table, f"projection-{record.id}-{index}-{method}",
                              filename=f"{plot_stem}_projection-{method}")
            if "true_labels" in artifact:
                true_figure = _scatter_figure(projected, artifact["true_labels"][rows], "Ground-truth classes", method)
                columns[1].plotly_chart(true_figure, width="stretch")
                projection_table["true_class"] = artifact["true_labels"][rows]
                _figure_downloads(true_figure, projection_table, f"true-projection-{record.id}-{index}-{method}",
                                  filename=f"{plot_stem}_true-projection-{method}")
            st.caption(
                f"Showing {len(rows):,} uniformly selected samples at most. Legends switch to a compact color scale "
                "when there are more than 20 classes."
            )
    if "true_labels" in artifact:
        st.subheader("Confusion matrix")
        try:
            matrix, row_names, column_names = compact_confusion(artifact["true_labels"], labels)
        except ValueError as exc:
            st.warning(str(exc))
            return
        normalize = st.checkbox("Normalize confusion matrix by true class", value=True)
        values = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) if normalize else matrix
        figure = go.Figure(go.Heatmap(z=values, x=column_names, y=row_names, colorscale="Blues",
                                     customdata=matrix,
                                     hovertemplate=("True class: %{y}<br>Cluster: %{x}<br>"
                                                    "Count: %{customdata}<extra></extra>")))
        figure.update_layout(xaxis_title="Predicted cluster", yaxis_title="True class", height=650)
        figure.update_xaxes(type="category", categoryorder="array", categoryarray=column_names)
        figure.update_yaxes(type="category", categoryorder="array", categoryarray=row_names, autorange="reversed")
        st.plotly_chart(figure, width="stretch")
        st.caption("Hungarian matching maximizes the diagonal using every class before display compaction. "
                   "Rows are true classes; columns are matched predicted cluster IDs. "
                   "Unmatched clusters remain separate.")
        confusion_table = pd.DataFrame(values, index=row_names, columns=column_names)
        _figure_downloads(figure, confusion_table.rename_axis("true_class").reset_index(),
                          f"confusion-{record.id}-{index}", filename=f"{plot_stem}_confusion")
        if "Other classes" in row_names or "Other clusters" in column_names:
            st.caption("Up to 30 high-frequency matched pairs are shown; remaining counts are combined into Other.")
    st.subheader("Cluster details")
    details = cluster_details(labels, artifact.get("true_labels"))
    st.dataframe(pd.DataFrame(details), hide_index=True, width="stretch")


def render_export(workspace):
    st.header("Export")
    kind = st.radio("Export type", ["Experiment batches", "Individual results", "Visualizations", "Prepared datasets"],
                    horizontal=True)
    st.caption("Download reports, saved figures, or prepared data. "
               "Session files are temporary; download before closing.")
    if kind == "Visualizations":
        if not workspace.plots:
            st.info("In Visualizations, click Save plot to Export below any figure first.")
            return
        key = st.selectbox("Saved plot", list(workspace.plots), format_func=lambda value: workspace.plots[value].name)
        plot = workspace.plots[key]
        name = st.text_input("Download name", plot.name, key=f"plot_name_{key}")
        format = st.selectbox("Plot format", ["zip", "json", "csv"],
                              format_func=lambda f: {"zip": "ZIP · interactive HTML, JSON and data",
                                                     "json": "JSON · Plotly figure", "csv": "CSV · plotted values"}[f])
        st.download_button("Download visualization", plot.export(format),
                            f"{download_stem(name)}.{format}", key=f"plot_download_{key}")
        if st.button("Remove saved plot"):
            workspace.plots.pop(key).cleanup()
            st.rerun()
        return
    if kind == "Prepared datasets":
        ready = [key for key in workspace.datasets if key not in workspace.pending_datasets]
        if not ready:
            st.info("Load a dataset and click Apply changes on the Dataset page first.")
            return
        key = st.selectbox("Prepared dataset", ready, format_func=lambda k: workspace.datasets[k].name)
        resource = workspace.datasets[key]
        include_labels = st.checkbox("Include ground-truth labels", value=workspace.benchmark_mode(key))
        recipe = workspace.preprocessing.get(key, PreprocessingSpec())
        token = (key, resource.name, repr(recipe), resource.hint.get("prepared_revision", 0), include_labels)
        st.caption("MATLAB file: prepared views, graphs, sample IDs, availability mask and metadata. "
                   "This includes applied Dataset preprocessing, not model-internal transformations.")
        if st.button("Prepare MATLAB file"):
            try:
                with st.spinner("Preparing dataset download"):
                    source = workspace.load_dataset(key)
                    if dataset_profile(source)["memory_bytes"] > 256 * 1024**2:
                        raise ValueError("This dataset exceeds the 256 MB interface export limit; "
                                         "use dataset_mat in Python")
                    data = prepare_dataset(source, recipe)
                    st.session_state._prepared_mat = {
                        "token": token, "data": dataset_mat(data, include_labels=include_labels),
                    }
            except Exception as exc:
                st.error(str(exc))
        cached = st.session_state.get("_prepared_mat", {})
        if cached.get("token") == token:
            st.download_button("Download prepared data (.mat)", cached["data"],
                               f"{download_stem(resource.name)}_prepared.mat", "application/octet-stream")
        return
    if kind == "Experiment batches":
        if not workspace.batches:
            st.info("Run an experiment batch first, or choose Individual results for searches and standalone runs.")
            return
        st.subheader("Complete experiment batches")
        batch_index = st.selectbox("Batch", list(range(len(workspace.batches))),
                                   format_func=lambda i: f"{workspace.batches[i]['name']} · "
                                   f"{workspace.batches[i]['report']['started_at'][:19]}")
        batch = workspace.batches[batch_index]
        report = batch["report"]
        st.caption(f"Results saved in: {Path(batch['json_path']).parent}")
        filename = download_stem(batch["name"], report["started_at"][:19])
        columns = st.columns(3)
        columns[0].download_button("Batch JSON", report_json(report), f"{filename}.json", "application/json")
        columns[1].download_button("Batch CSV", report_csv(report), f"{filename}.csv", "text/csv")
        columns[2].download_button("Batch ZIP", report_zip(report), f"{filename}.zip", "application/zip")
        return
    record = _record_selector(workspace, "export_record")
    if record is None:
        return
    st.write("JSON contains model/dataset details, runs, summaries and diagnostics. "
             "The complete ZIP also includes predictions and embeddings as compressed arrays.")
    _downloads(record, prefix=f"export_{record.id}")


def _sidebar(workspace):
    with st.sidebar:
        from pymv import __version__

        _, center, _ = st.columns([1, 3, 1])
        center.image(str(ASSETS / "logo.png"), width="stretch")
        st.markdown(f'<div class="pymv-brand"><h1>pymv</h1><p>Version {__version__}</p></div>',
                    unsafe_allow_html=True)
        st.caption("Multi-view & multimodal clustering and representation learning")
        step = st.session_state.get("workflow_step", 0)
        pages = ["Dataset", "Model", "Search", "Experiment", "Results", "Visualizations", "Export"]
        with st.container(key="sidebar_navigation"):
            for label in pages:
                if st.button("Hyperparameter search" if label == "Search" else label,
                             key=f"nav_{label}", width="stretch",
                             type="primary" if STEPS[step] == label else "secondary"):
                    _set_page(label)
                    st.rerun()
        st.divider()
        st.caption(f"{len(workspace.datasets)} datasets · {len(workspace.models)} model configurations")
        st.caption("Manage your data on the Dataset page. Runs are executed sequentially.")
        st.markdown(
            "[Documentation](https://pymv.readthedocs.io/) · "
            "[Source](https://github.com/AbdelheqMokhtari/pymv)"
        )
        return STEPS[step]


def main():
    st.set_page_config(page_title="pymv", page_icon=str(ASSETS / "logo.png"), layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    
    # --- CI Test & Cross-Page Persistence Fixes (Minimum Version Support) ---
    st.session_state.setdefault("dataset_class_counts", True)
    
    if "workspace" in st.session_state:
        for dataset_id in st.session_state.workspace.datasets:
            st.session_state.setdefault(f"experiment_dataset_k_{dataset_id}", 3)
            
    if "drafts" in st.session_state:
        for draft in st.session_state.drafts.values():
            model = draft.get("model")
            if model:
                try:
                    _, controls = parameter_controls(model)
                    for param in controls:
                        st.session_state.setdefault(f"search_{model}_{param}", param in draft.get("search_space", {}))
                except Exception:
                    pass
                    
                for param, candidates in draft.get("search_space", {}).items():
                    st.session_state.setdefault(f"candidates_{model}_{param}", [str(c) for c in candidates])
    # -------------------------------------------------------------------------

    if "_next_workflow_step" in st.session_state:
        st.session_state.workflow_step = st.session_state.pop("_next_workflow_step")
    workspace = _workspace()
    
    if "_restore_model" in st.session_state:
        settings = st.session_state.pop("_restore_model")
        _, controls = parameter_controls(settings["model"])
        for name, value in settings["params"].items():
            if name in controls:
                if controls[name].kind == "int_tuple":
                    value = ", ".join(map(str, value))
                st.session_state[f"parameter_{settings['model']}_{name}"] = value
        st.session_state.clusters_primary_model_library = settings["clusters"][0]
        
    page = _sidebar(workspace)
    
    renderers = {
        "Dataset": render_dataset,
        "Preprocessing": render_dataset,
        "Model": render_model,
        "Experiment": render_experiment,
        "Results": render_results,
        "Visualizations": render_visualizations,
        "Export": render_export,
        "Search": render_search,
    }
    renderers[page](workspace)


if __name__ == "__main__":
    main()