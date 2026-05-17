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
from . import chameleon  # noqa: F401
from . import mri  # noqa: F401
from . import fmm  # noqa: F401
from . import lmm  # noqa: F401
from . import amm  # noqa: F401
from . import sklearn_extras  # noqa: F401
from . import improvements  # noqa: F401
from . import aura  # noqa: F401
from . import meta_clusterer  # noqa: F401
from . import rapid  # noqa: F401
from . import aura_v2  # noqa: F401
from . import meta_clusterer_v2  # noqa: F401
from . import rapid_v2  # noqa: F401
from . import aura_v3  # noqa: F401
from . import meta_clusterer_v3  # noqa: F401
from . import rapid_v3  # noqa: F401
from . import learned_router  # noqa: F401
from . import learned_router_v2  # noqa: F401
