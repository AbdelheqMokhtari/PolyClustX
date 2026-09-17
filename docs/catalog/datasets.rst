Available datasets and variants
=================================

This catalog is generated from the registry used by Python, the CLI, and the
workbench. Each row is an artifact; related artifacts share a family and variant.

.. code-block:: python

   from pymv.data import list_variants, get_dataset_info, load_data
   print(list_variants("pascal"))
   print(get_dataset_info("pascal", variant="clip"))  # no download
   data = load_data("pascal", variant="clip")

.. pymv-catalog:: datasets

Interpretation and provenance
-----------------------------

The four settings describe graph availability and whether views have the same
modality. Completeness is separate; missing-view simulation can be applied to
complete multi-view datasets. Different image descriptors remain one modality,
while image and text are distinct modalities. Custom/unknown modalities default
to a common unspecified modality until the user supplies explicit annotations.
Verify these annotations against your study design.

Linked Zenodo records identify the distributed artifacts, not a blanket license
grant. Consult each dataset's original publications and terms before
redistributing it. Files are cached on first use. The hosted BBC4view file is
known to be corrupt; the interface disables its download button.

Graph-bearing and feature-only variants can share a family:
``load_data("acm")`` loads two graph views, whereas
``load_data("acm", variant="5view")`` loads five feature views.

Old names such as ``pascal_clip`` and ``acm_5view`` remain deprecated aliases.
Use ``list_datasets(include_aliases=True)`` when migrating older configurations.
