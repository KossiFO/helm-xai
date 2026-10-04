"""HTML dashboards do not import optional plotting libraries."""
from .dashboard import Dashboard

_PLOTS = {
    'plot_attributions', 'plot_method_comparison', 'plot_decision_matrix_heatmap',
    'plot_fidelity_comparison', 'plot_ucb1_evolution', 'plot_helm_architecture',
}
__all__ = ['Dashboard', *sorted(_PLOTS)]


def __getattr__(name):
    if name in _PLOTS:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(name)
