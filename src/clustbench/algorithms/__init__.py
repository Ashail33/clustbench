"""Clustering algorithm registry.

Importing this package triggers the ``@register`` decorators in each
algorithm module, populating :data:`clustbench.algorithms.base.ALGO_REGISTRY`.
"""

from . import base  # noqa: F401
from . import kmeans  # noqa: F401
from . import minibatch_kmeans  # noqa: F401
from . import dbscan  # noqa: F401
from . import birch  # noqa: F401
from . import clarans  # noqa: F401
from . import parallel_kmeans  # noqa: F401
from . import pwcc  # noqa: F401
from . import s5c  # noqa: F401
from . import sklearn_extras  # noqa: F401
