Available datasets and representations
======================================

The catalog contains 37 families and 66 representation files in four settings.
Each family appears once below, with its stored features, structural relations,
sample/class counts, reference and direct download links. A representation is an
alternate feature encoding of the same samples, not another dataset family.

Files download directly from Hugging Face: no extra client, login or MATLAB
installation is needed. Files whose uploads are still pending become usable at
the same links when published. Standardized files contain X* features, optional
A* adjacency matrices and a labels vector. See :doc:`../user_guide/data` for
loading and composing views.

.. code-block:: python

   from polyclustx.data import load_data, list_variants, get_dataset_info

   print(list_variants("pascal"))
   print(get_dataset_info("grocery-S", variant="clip_minilm"))  # no download
   data = load_data("pascal", variant="clip")
   print(data.feature_dims)  # actual stored widths, not source-encoder widths

Graph datasets may have fewer feature matrices than composed views. ACM stores
one keyword matrix and two adjacencies; its default composition reuses the
keywords once per relation. You can select features without graphs, pair a
feature with any graph, or fuse several relations in one view using ``views=``.

Dataset families and variants
-----------------------------

.. polyclustx-catalog:: datasets

Notes on interpretation
-----------------------

**Same modality or different modalities.** Several visual descriptors count as
multi-view input. Images paired with text count as multimodal input, including
Wikipedia-test. Grocery-S and Reddit-S are separate smaller datasets; encoder
variants are listed within each of them.

**Source encoders versus stored columns.** ESP-Game, IAPR-TC12, MIRFlickr and
NUS-WIDE originate from 4096-dimensional VGG-16 image and 768-dimensional BERT
text representations. The uploaded files contain processed 100-dimensional
views. The loader uses their actual shapes and does not invent missing columns.
Grocery-S CLIP+MiniLM stores 512 image and 384 text columns.

**Graph counts.** Stored arcs count directed nonzero entries, not necessarily
unique undirected edges. Existing self-loops and asymmetry are retained. The
tables report the supplied reference statistics; inspect the Dataset page for
properties measured on the chosen file.

**References and reuse.** Background links identify benchmark collections or
related reference implementations, not an assertion that every processed
artifact is identical to an original benchmark. Photo and Arts provenance,
and Grocery-S subset construction, remain unspecified. Check original dataset
terms before redistributing files.

**Missing views.** Completeness is independent of the four categories. Use
:doc:`../user_guide/missing_views` to simulate missing observations or align
different sample IDs. Model requirements are in :doc:`models`.

Old representation names such as ``pascal_clip`` and older ``*_multimodal``
names remain migration aliases. Prefer ``load_data("pascal", variant="clip")``.
Legacy artifacts without a supplied Hugging Face replacement (for example ACM
5view, BBC4view, CiteSeer and Cora) are not advertised by this catalog; use
explicit local-file mapping for your own copies.
