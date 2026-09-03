# pymv notebooks

A hands-on walkthrough of the library, one stage at a time. Every notebook
runs **offline in a few seconds** on a small synthetic multi-view dataset
defined in its first cell — swap in a real benchmark by uncommenting the
`load_dataset(...)` line.

| # | Notebook | Stage |
|---|----------|-------|
| 00 | [`00_quickstart.ipynb`](00_quickstart.ipynb) | the whole pipeline at a glance |
| 01 | [`01_data_and_preprocessing.ipynb`](01_data_and_preprocessing.ipynb) | the dataset catalog + normalize → propagate |
| 02 | [`02_lmgec_baseline.ipynb`](02_lmgec_baseline.ipynb) | the LMGEC algorithm, step by step |
| 03 | [`03_experiments_and_tuning.ipynb`](03_experiments_and_tuning.ipynb) | repeated comparisons + random search |

## Running

```bash
pip install -e .
jupyter lab            # then open the notebooks in order
```

Or execute the whole set headless:

```bash
for nb in notebooks/0*.ipynb; do
  jupyter nbconvert --to notebook --execute --inplace "$nb"
done
```
