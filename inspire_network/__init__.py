"""Tools for fetching and analyzing data from INSPIRE-HEP."""

from inspire_network.client import InspireClient
from inspire_network.analysis import (
    get_author_papers,
    count_arxiv_categories,
    build_collaboration_network,
)

__all__ = [
    "InspireClient",
    "get_author_papers",
    "count_arxiv_categories",
    "build_collaboration_network",
]
