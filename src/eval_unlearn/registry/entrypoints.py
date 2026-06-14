from importlib.metadata import entry_points

from .local import register_technique, register_metric, register_dataset
from ..logging_utils import get_logger

logger = get_logger(__name__)


def load_entrypoints():
    """Discover and register plugins from package entry points."""
    groups = {
        "eval_unlearn.techniques": register_technique,
        "eval_unlearn.metrics": register_metric,
        "eval_unlearn.datasets": register_dataset,
    }

    logger.debug("Loading plugins from entry points...")

    for group_name, register in groups.items():
        try:
            eps = entry_points(group=group_name)
        except TypeError:
            # Python 3.9 fallback: entry_points() returns a dict
            all_eps = entry_points()
            eps = all_eps.get(group_name, [])

        for ep in eps:
            try:
                plugin_obj = ep.load()
                register(ep.name)(plugin_obj)
                logger.debug("Registered plugin '%s' (%s)", ep.name, group_name)
            except Exception as e:
                logger.error("Failed to load plugin %s: %s", ep.name, e)
