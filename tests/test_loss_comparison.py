import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from polyclustx.experiment import Experiment  # noqa: E402
from polyclustx.metrics import nmi_score  # noqa: E402
from polyclustx.visualization import plot_loss, plot_loss_comparison  # noqa: E402
from tests.test_experiment_reports import ReportModel, dataset  # noqa: E402


def test_compare_default_run_one_and_explicit_repeats(tmp_path):
    result = Experiment(dataset(), [ReportModel(2), ReportModel(3)], {"NMI": nmi_score},
                        runs=2, verbose=0, save_path=tmp_path).run()
    ax = plot_loss_comparison(result, show=False)
    assert len(ax.lines) == 2
    for line in ax.lines:
        np.testing.assert_allclose(line.get_ydata(), [1, .375, 0])
    plt.close(ax.figure)
    ax = plot_loss_comparison([result, result.to_dict()], run=None, show=False)
    assert len(ax.lines) == 8
    plt.close(ax.figure)


def test_single_loss_normalization_preserves_legacy_default():
    ax = plot_loss([10, 5, 2], show=False)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [10, 5, 2])
    plt.close(ax.figure)
    ax = plot_loss([10, 5, 2], normalize=True, show=False)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, .375, 0])
    plt.close(ax.figure)
