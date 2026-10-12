Available datasets and variants
=================================

This catalog is generated from the registry used by Python, the CLI, and the
interface. Each family appears once, with its variants and download links together.
An artifact link identifies the actual hosted file. A reference link provides
dataset background; provisional references are suggested reading, not verified
provenance. Reference-collection counts describe that collection's version;
inspect loaded arrays for the exact dimensions of your chosen representation.

.. code-block:: python

   from polyclustx.data import list_variants, get_dataset_info, load_data
   print(list_variants("pascal"))
   print(get_dataset_info("pascal", variant="clip"))  # no download
   data = load_data("pascal", variant="clip")

.. polyclustx-catalog:: datasets

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
