"""Scientific multi-experiment workspace. Run via pymv-display."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pymv.data import (
    MultiViewDataset,
    dataset_from_arrays,
    get_dataset_info,
    list_datasets,
    list_variants,
    load_data,
    read_arrays,
)
from pymv.display.workspace import (
    EXTERNAL_NAMES,
    INTERNAL_NAMES,
    ExperimentSpec,
    Workspace,
    export_record,
)
from pymv.models import list_models
from pymv.models.registry import get_model_info

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def _add_data(workspace, data):
    key = workspace.add_dataset(data)
    st.session_state["selected_dataset"] = key
    st.success(f"Added {data.name or 'dataset'}: {data.n_samples} samples, {data.n_views} views.")


def render_import(workspace):
    st.header("Datasets")
    st.caption("Load once, reuse across experiments. Each saved dataset is an independent snapshot.")
    source = st.radio("Source", ["Catalog", "Upload files", "Synthetic example"], horizontal=True)
    if source == "Catalog":
        name = st.selectbox("Dataset", list_datasets())
        variant = st.selectbox("Variant", list_variants(name))
        info = get_dataset_info(name, variant=variant)
        st.write(info["description"])
        st.caption(f"{info['category']} · {info['n_views']} views · Downloads are cached locally")
        if info["availability_note"]:
            st.warning(info["availability_note"])
        if st.button("Load dataset", disabled=bool(info["availability_note"])):
            try:
                with st.spinner("Loading dataset"):
                    _add_data(workspace, load_data(name, variant=variant))
            except Exception as exc:
                st.error(str(exc))
    elif source == "Synthetic example":
        views = st.number_input("Number of views", 1, 12, 2)
        labeled = st.checkbox("Include evaluation labels", value=False)
        if st.button("Create example"):
            rng = np.random.default_rng(7)
            y = np.repeat(np.arange(3), 20)
            xs = [rng.normal(size=(60, 6)) + y[:, None] * 3 for _ in range(views)]
            _add_data(workspace, MultiViewDataset(xs, labels=y if labeled else None, name="Synthetic example"))
    else:
        files = st.file_uploader(
            "NPY, NPZ, numeric CSV/TSV, MATLAB v7 or earlier",
            type=["npy", "npz", "csv", "tsv", "mat"],
            accept_multiple_files=True,
        )
        skip_header = st.number_input("CSV/TSV header rows to skip", 0, 100, 0)
        st.caption("No variable is assumed to be a feature, label, ID or graph. NumPy pickle files are disabled.")
        arrays = {}
        try:
            for file in files:
                for key, value in read_arrays(file.getvalue(), filename=file.name, skip_header=skip_header).items():
                    full = f"{file.name}:{key}"
                    if full in arrays:
                        raise ValueError(f"Duplicate filename/key: {full}")
                    arrays[full] = value
        except Exception as exc:
            st.error(str(exc))
            return
        if arrays:
            st.dataframe(
                pd.DataFrame([{"key": k, "shape": str(v.shape), "dtype": str(v.dtype)} for k, v in arrays.items()]),
                hide_index=True,
                use_container_width=True,
            )
            keys = list(arrays)
            views = st.multiselect("Feature arrays (one per view)", keys)
            label_key = st.selectbox("Ground-truth labels (optional)", [None, *keys])
            id_alignment = st.checkbox("Align views using sample IDs")
            transpose = st.multiselect("Transpose these feature arrays (samples are columns)", views)
            id_keys, graphs, modalities = [], {}, []
            for view in views:
                with st.expander(f"Mapping: {view}", expanded=True):
                    modalities.append(st.text_input("Modality", "features", key=f"modality_{view}"))
                    graphs[view] = st.multiselect("Graph relations belonging to this view", keys, key=f"graphs_{view}")
                    if id_alignment:
                        id_keys.append(st.selectbox("Sample IDs for this view", keys, key=f"ids_{view}"))
            label_id = None
            if id_alignment and label_key:
                label_id = st.selectbox("IDs for the label vector", keys)
            mask_key, detect_nan = None, False
            if not id_alignment:
                mask_key = st.selectbox("Observation mask (optional; rows × views, 1 = observed)", [None, *keys])
                detect_nan = st.checkbox("Treat fully NaN feature rows as missing views")
                aligned = st.checkbox("I confirm the arrays use the same sample row order")
            else:
                aligned = True
                st.info("IDs are aligned by union. Each graph must follow its associated view's local ID order.")
            name = st.text_input("Dataset name", "Uploaded data")
            if st.button("Save uploaded dataset", disabled=not views or not aligned):
                try:
                    data = dataset_from_arrays(
                        arrays,
                        view_keys=views,
                        graph_keys=graphs,
                        label_key=label_key,
                        id_keys=id_keys or None,
                        label_id_key=label_id,
                        mask_key=mask_key,
                        transpose_views=transpose,
                        detect_nan_rows=detect_nan,
                        modalities=modalities,
                        name=name,
                    )
                    _add_data(workspace, data)
                except Exception as exc:
                    st.error(str(exc))
    if workspace.datasets:
        st.subheader("Session datasets")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "name": d.name,
                        "samples": d.n_samples,
                        "views": d.n_views,
                        "category": d.category,
                        "complete": d.is_complete,
                        "observed pairs": int(d.observed_mask.sum()),
                    }
                    for d in workspace.datasets.values()
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


def _apply_pending():
    spec = st.session_state.pop("pending_recipe", None)
    if spec is None:
        return
    mapping = {
        "selected_dataset": spec.dataset_id,
        "model": spec.model,
        "experiment_name": spec.name + " (copy)",
        "params": json.dumps(spec.params, indent=2),
        "clusters": ", ".join(map(str, spec.clusters)),
        "repeats": spec.repeats,
        "seed": spec.seed,
        "benchmark": spec.benchmark,
        "fusion": spec.fusion,
        "weights": json.dumps(spec.relation_weights),
        "missing_rate": spec.missing_rate,
        "imputation": spec.imputation or "None",
        "center": spec.center,
        "scale": spec.scale,
        "search_space": json.dumps(spec.search_space),
        "trials": spec.trials,
        "metrics": spec.metrics,
    }
    st.session_state.update(mapping)
    st.session_state["params_model"] = spec.model


def render_editor(workspace):
    st.header("Experiment workspace")
    if not workspace.datasets:
        st.info("Add a dataset in the Datasets tab to start. You can return to any tab at any time.")
        return
    data_id = st.selectbox(
        "Dataset snapshot",
        list(workspace.datasets),
        key="selected_dataset",
        format_func=lambda k: f"{workspace.datasets[k].name} · {k[:6]}",
    )
    data = workspace.datasets[data_id]
    st.caption(
        f"{data.n_samples} samples · {data.n_views} views · {data.category} · "
        f"{'complete' if data.is_complete else 'incomplete'}"
    )
    with st.expander("View availability and graph mapping"):
        st.dataframe(
            pd.DataFrame(data.observed_mask, columns=data.view_names, index=data.sample_ids), use_container_width=True
        )
        st.write(
            {
                name: len(a) if isinstance(a, (list, tuple)) else int(a is not None)
                for name, a in zip(data.view_names, data.As)
            }
        )
    left, right = st.columns(2)
    with left:
        st.text_input("Experiment name", value="Experiment", key="experiment_name")
        model_name = st.selectbox("Model", list_models(), key="model")
        info = get_model_info(model_name)
        st.caption(info.get("description", ""))
        if st.session_state.get("params_model") != model_name:
            st.session_state["params"] = json.dumps(info["defaults"], indent=2)
            st.session_state["params_model"] = model_name
        st.text_input("Cluster counts K (comma-separated)", value="2, 3, 4", key="clusters")
        st.number_input("Repeats per configuration", 1, 100, 1, key="repeats")
        st.number_input("Base random seed", 0, 2**31 - 101, 0, key="seed")
    with right:
        if data.labels is None:
            st.session_state["benchmark"] = False
        benchmark = st.checkbox(
            "Benchmark evaluation (use ground-truth labels)", key="benchmark", disabled=data.labels is None
        )
        st.caption("Training is always unsupervised. Turning this off excludes labels from scoring and K selection.")
        available = INTERNAL_NAMES + (EXTERNAL_NAMES if benchmark else [])
        if "metrics" in st.session_state:
            st.session_state["metrics"] = [m for m in st.session_state["metrics"] if m in available]
        st.multiselect("Metrics", available, default=INTERNAL_NAMES, key="metrics")
        st.caption(
            "Internal scores use the learned embedding: Silhouette and Calinski–Harabasz ↑; "
            "Davies–Bouldin ↓. Different embedding spaces are not directly comparable."
        )
        st.number_input("Simulated missing-view fraction", 0.0, 0.99, 0.0, step=0.05, key="missing_rate")
        st.selectbox("Explicit missing-view imputation", ["None", "mean", "zero"], key="imputation")
        st.caption(
            "LMGEC and SCM currently require complete or explicitly imputed views. "
            "At least one view per sample is retained; infeasible removal rates are rejected."
        )
    with st.expander("Model parameters", expanded=False):
        st.text_area("Constructor parameters (JSON; K and seeds use the fields above)", key="params", height=280)
        st.caption(
            "LMGEC beta/tf_idf/center/scale accept per-view values where supported. "
            "SCM's miss_rate is training augmentation, not dataset missingness."
        )
    with st.expander("Relations and preprocessing"):
        st.selectbox("Early graph fusion", ["sum", "hadamard"], key="fusion")
        st.text_input(
            'Relation weights (JSON by zero-based view index, e.g. {"0": [0.3, 0.7]})', value="{}", key="weights"
        )
        st.caption(
            "Sum defaults to weights of one. Hadamard is an elementwise product and rejects weights. "
            "Fusion precedes graph propagation; SCM does not consume graphs."
        )
        st.checkbox("Center features before model preprocessing", key="center")
        st.checkbox("Scale features before model preprocessing", key="scale")
    with st.expander("Optional random parameter search"):
        st.text_area('Search space (JSON lists, e.g. {"embedding_dim": [4, 8]})', value="{}", key="search_space")
        st.number_input("Sampled configurations (each evaluated for every K)", 1, 1000, 1, key="trials")
    if st.button("Run experiment", type="primary"):
        try:
            state = st.session_state
            spec = ExperimentSpec(
                dataset_id=data_id,
                model=model_name,
                name=state["experiment_name"],
                params=json.loads(state["params"]),
                clusters=list(dict.fromkeys(int(k.strip()) for k in state["clusters"].split(","))),
                repeats=state["repeats"],
                seed=state["seed"],
                benchmark=benchmark,
                metrics=state["metrics"],
                fusion=state["fusion"],
                relation_weights=json.loads(state["weights"]),
                missing_rate=state["missing_rate"],
                imputation=None if state["imputation"] == "None" else state["imputation"],
                center=state["center"],
                scale=state["scale"],
                search_space=json.loads(state["search_space"]),
                trials=state["trials"],
            )
            bar = st.progress(0.0, text="Running independent trials")
            with st.spinner("Fitting models"):
                record = workspace.run(spec, progress=bar.progress)
            failed = sum(row["status"] == "failed" for row in record.rows)
            if failed:
                st.warning(f"Saved experiment with {failed} failed trial(s). See the recorded errors in History.")
            else:
                st.success("Experiment saved to History. Change any settings to run another experiment.")
        except Exception as exc:
            st.error(str(exc))


def render_history(workspace):
    st.header("Experiment history")
    st.caption(
        "Newest first. Recipes and results remain unchanged when you edit or run another experiment. "
        "Session memory is temporary: download records before closing or restarting the server."
    )
    if not workspace.records:
        st.info("No experiments yet.")
    for record in list(workspace.records):
        with st.expander(f"{record.spec.name} · {record.spec.model.upper()} · {record.created_at[:19]}", expanded=True):
            a, b, c = st.columns(3)
            if a.button("Edit / duplicate settings", key=f"copy_{record.id}"):
                st.session_state["pending_recipe"] = copy.deepcopy(record.spec)
                st.rerun()
            if b.button("Delete result", key=f"delete_{record.id}"):
                workspace.delete_record(record.id)
                st.rerun()
            c.download_button(
                "Download experiment ZIP",
                export_record(record),
                file_name=f"pymv-{record.id[:8]}.zip",
                mime="application/zip",
                key=f"export_{record.id}",
            )
            st.caption(
                f"{record.dataset_summary['name']} · "
                f"{'benchmark evaluation' if record.spec.benchmark else 'unlabeled evaluation'} · "
                f"fingerprint {record.dataset_summary['fingerprint'][:12]}"
            )
            st.dataframe(
                pd.DataFrame(record.rows).drop(columns=["params"], errors="ignore"),
                hide_index=True,
                use_container_width=True,
            )
            with st.expander("Recipe and provenance"):
                st.json({"recipe": asdict(record.spec), "dataset": record.dataset_summary})
            successful = [i for i, a in enumerate(record.artifacts) if "embedding" in a]
            if successful:
                with st.expander("Explore a trial"):
                    idx = st.selectbox(
                        "Trial",
                        successful,
                        format_func=lambda i: f"{i + 1}: K={record.rows[i]['K']}",
                        key=f"trial_{record.id}",
                    )
                    artifact = record.artifacts[idx]
                    counts = pd.Series(artifact["labels"]).value_counts().sort_index()
                    st.bar_chart(counts, color="#8A1538")
                    if "loss" in artifact and artifact["loss"].ndim == 1:
                        st.line_chart(artifact["loss"], color="#8A1538")
                    if st.button("Show PCA projection", key=f"plot_{record.id}"):
                        import plotly.express as px
                        from sklearn.decomposition import PCA

                        z = artifact["embedding"]
                        if z.shape[1] >= 2 and len(z) >= 2:
                            projected = PCA(n_components=2).fit_transform(z)
                            fig = px.scatter(
                                x=projected[:, 0],
                                y=projected[:, 1],
                                color=artifact["labels"].astype(str),
                                labels={"x": "PC 1", "y": "PC 2", "color": "Cluster"},
                                color_discrete_sequence=["#8A1538", "#475569", "#64748B", "#376D72", "#947D64"],
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.download_button(
                                "Download interactive plot",
                                fig.to_html(),
                                "projection.html",
                                key=f"plot_download_{record.id}",
                            )
                        else:
                            st.info("A 2D projection requires at least two embedding dimensions.")


def main():
    st.set_page_config(page_title="pymv · Scientific workbench", page_icon=str(ASSETS / "logo.png"), layout="wide")
    css = (ASSETS / "workbench.css").read_text()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = Workspace()
    _apply_pending()
    workspace = st.session_state["workspace"]
    with st.sidebar:
        st.image(str(ASSETS / "logo.png"), width=112)
        st.title("pymv")
        st.caption("A workspace for multi-view clustering research")
        st.write(f"{len(workspace.datasets)} datasets · {len(workspace.records)} experiments")
        st.markdown(
            "[Documentation](https://pymv.readthedocs.io/) · "
            "[Source & contributions](https://github.com/AbdelheqMokhtari/pymv)"
        )
        st.caption(
            "Local session. Data stays with the server running this application; "
            "do not expose an unauthenticated server to the public internet."
        )
    st.title("Scientific workbench")
    st.write("Explore views, compare cluster counts, and keep each experiment.")
    data_tab, editor_tab, history_tab = st.tabs(["Datasets", "Experiment workspace", "History"])
    with data_tab:
        render_import(workspace)
    with editor_tab:
        render_editor(workspace)
    with history_tab:
        render_history(workspace)


if __name__ == "__main__":
    main()
