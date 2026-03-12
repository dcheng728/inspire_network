"""Tools for fetching and analyzing data from INSPIRE-HEP."""

from inspire_tools.client import InspireClient
from inspire_tools.analysis import (
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
