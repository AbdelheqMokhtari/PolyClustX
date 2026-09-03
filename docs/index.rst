.. pymv documentation master file

.. raw:: html

   <!-- Hero banner -->
   <div class="hero-banner">
     <h2>pymv</h2>
     <p><strong>A comparative framework for multi-view &amp; multi-modal clustering in Python</strong></p>
     <p>Benchmark · Compare · Publish — in a few lines of code</p>
     <span class="version-badge">v 1.1.1</span>
   </div>

   <!-- Badges -->
   <div class="badge-row">
     <a href="https://github.com/AbdelheqMokhtari/pymv"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-repo-blue?logo=github"></a>
     <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white">
     <img alt="License MIT" src="https://img.shields.io/badge/license-MIT-green">
     <img alt="Version" src="https://img.shields.io/badge/version-1.1.1-orange">
   </div>

   <hr class="landing-divider">

   <!-- Feature cards -->
   <div class="feature-grid">
     <div class="feature-card">
       <div class="card-icon">🧠</div>
       <h3>Clustering Models</h3>
       <p>Drop-in model classes with a <code>fit</code> / <code>fit_predict</code> API.
          Ships <strong>LMGEC</strong>, and a registry so you can add your own
          in one decorator.</p>
     </div>
     <div class="feature-card">
       <div class="card-icon">📦</div>
       <h3>55 Built-in Datasets</h3>
       <p>ACM, DBLP, IMDB, Cora, Wiki, Amazon Photos, BDGP, COCO, Pascal, Prokaryotic
          and many more — auto-downloaded &amp; cached from Zenodo.</p>
     </div>
     <div class="feature-card">
       <div class="card-icon">🔬</div>
       <h3>Experiment Framework</h3>
       <p>Run <em>datasets × models × metrics × runs</em> in one call.
          Pretty tables on the console, full logs on disk.</p>
     </div>
     <div class="feature-card">
       <div class="card-icon">🎯</div>
       <h3>Hyper-parameter Search</h3>
       <p><code>RandomSearchExperiment</code> finds the best config and reports
          <code>best_params</code> &amp; <code>best_score</code>.</p>
     </div>
     <div class="feature-card">
       <div class="card-icon">📊</div>
       <h3>7 Clustering Metrics</h3>
       <p>ACC, NMI, ARI, F1 (external) + Silhouette, Davies-Bouldin,
          Calinski-Harabasz (internal) — all with docstrings &amp; math.</p>
     </div>
     <div class="feature-card">
       <div class="card-icon">🎨</div>
       <h3>Visualization</h3>
       <p>t-SNE &amp; PCA scatter plots of learned embeddings — one function call
          with automatic coloring by cluster or ground truth.</p>
     </div>
   </div>

   <hr class="landing-divider">

Installation
------------

.. raw:: html

   <div class="install-box">pip install pymv</div>

Or install from source for the latest development version:

.. code-block:: bash

   git clone https://github.com/AbdelheqMokhtari/pymv.git
   cd pymv
   pip install -e .

Quick Example
-------------

.. raw:: html

   <div class="quickstart-section">

.. code-block:: python

   from pymv.data import load_acm
   from pymv.models import LMGEC
   from pymv.metrics import clustering_accuracy, nmi_score

   # 1. Load a multi-view attributed graph
   Xs, As, y = load_acm()

   # 2. Fit & predict
   labels = LMGEC(n_clusters=3, embedding_dim=16).fit_predict(Xs, As)

   # 3. Evaluate
   print("ACC:", clustering_accuracy(y, labels))
   print("NMI:", nmi_score(y, labels))

.. raw:: html

   </div>

Run a full comparative experiment in 10 lines:

.. code-block:: python

   from pymv.experiment import Experiment
   from pymv.models import LMGEC
   from pymv.metrics import clustering_accuracy, nmi_score, ari_score

   result = Experiment(
       datasets=["acm", "dblp", "imdb"],
       models=[
           LMGEC(n_clusters=3, embedding_dim=16, temperature=1.0),
           LMGEC(n_clusters=3, embedding_dim=32, temperature=5.0),
       ],
       metrics=[clustering_accuracy, nmi_score, ari_score],
       runs=10,
   ).run()

➡ See :doc:`user_guide/examples` for 8 more in-depth examples.

.. raw:: html

   <hr class="landing-divider">

.. toctree::
   :maxdepth: 2
   :caption: 📖 User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/examples
   user_guide/changelog

.. toctree::
   :maxdepth: 2
   :caption: 📚 API Reference

   api/models
   api/data
   api/preprocessing
   api/metrics
   api/experiment
   api/visualization
   api/cli
   api/core

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
