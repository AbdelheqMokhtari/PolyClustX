"""Dataset descriptions and named feature/graph composition controls."""

import pandas as pd
import streamlit as st


def view_controls(features, graphs, selection, *, prefix, descriptions=None, modalities=None):
    """Return an explicit selection; users can reuse features and graph relations."""
    descriptions, modalities = descriptions or {}, modalities or {}
    entries = list(selection.items())
    count = st.number_input("Number of views", min_value=1, max_value=32,
                            value=min(len(entries), 32), key=f"{prefix}_count")
    result, kinds = {}, []
    valid = True
    for index in range(count):
        initial_name, initial = entries[index] if index < len(entries) else (f"view_{index + 1}", (features[0], []))
        feature, relations = initial
        left, right = st.columns(2)
        name = left.text_input("View name", initial_name, key=f"{prefix}_name_{index}").strip()
        chosen = right.selectbox("Feature array", features, index=features.index(feature),
                                 key=f"{prefix}_feature_{index}")
        if descriptions.get(chosen):
            st.caption(descriptions[chosen])
        attached = st.multiselect("Graph relations (none = feature-only)", graphs,
                                  default=relations, key=f"{prefix}_graphs_{index}")
        kind = st.text_input("Modality", modalities.get(chosen, "features"),
                             key=f"{prefix}_modality_{index}_{chosen}").strip() or "features"
        if not name or name in result:
            st.warning("Give every view a nonempty, unique name.")
            valid = False
        result[name] = (chosen, attached)
        kinds.append(kind)
    return (result if valid else None), kinds


def catalog_details(info):
    """Describe stored arrays separately from the default algorithm views."""
    st.write(info["description"])
    st.caption(info["category_label"])
    columns = st.columns(4)
    for column, (label, value) in zip(columns, [
        ("Samples", info["n_samples"]), ("Classes", info["n_clusters"]),
        ("Stored features", info["n_feature_views"]), ("Stored graphs", info["n_graphs"]),
    ]):
        column.metric(label, f"{value:,}" if value is not None else "Not specified")
    st.caption(f"Default composition: {info['n_views']} view(s). "
               "You can change feature/graph combinations after loading.")
    if info["encoders"]:
        st.write(f"Feature encoders: {info['encoders']}")
    if info["reference_details"]:
        st.caption(info["reference_details"])
    if info["feature_descriptions"]:
        dimensions = info["feature_dims"]
        st.dataframe(pd.DataFrame([
            {"feature": index + 1, "meaning": description,
             "expected dimensions": dimensions[index] if index < len(dimensions) else "Inspect after loading"}
            for index, description in enumerate(info["feature_descriptions"])
        ]), hide_index=True, width="stretch")
    if info["graph_descriptions"]:
        st.write("Structural relations: " + "; ".join(info["graph_descriptions"]))
    stats = info["graph_statistics"]
    if stats:
        st.caption(f"Stored arcs: {stats['stored_arcs']:,} · unordered pairs: {stats['unordered_pairs']:,} · "
                   f"self-loops: {stats['self_loops']:,} · symmetric: {'yes' if stats['symmetric'] else 'no'}. "
                   "The loader preserves graph direction and weights.")
    st.markdown(f"[Download MAT file]({info['url']}) · [Dataset reference]({info['reference']})")

