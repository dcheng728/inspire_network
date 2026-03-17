"""High-level analysis helpers built on top of :class:`InspireClient`."""

from __future__ import annotations

import datetime
import json
import math
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations

import networkx as nx

from inspire_network.client import InspireClient


# ======================================================================
# Data classes for clean output
# ======================================================================

@dataclass
class PaperInfo:
    """Condensed metadata for a single paper."""

    recid: str
    title: str
    authors: list[str]
    author_bais: list[str | None]
    author_count: int
    arxiv_id: str | None
    arxiv_categories: list[str]
    doi: str | None
    citation_count: int
    date: str | None
    journal: str | None

    @classmethod
    def from_hit(cls, hit: dict) -> PaperInfo:
        meta = hit.get("metadata", {})

        title = ""
        if titles := meta.get("titles"):
            title = re.sub(r"<[^>]+/?>", "", titles[0].get("title", ""))

        author_entries = meta.get("authors", [])
        authors: list[str] = []
        author_bais: list[str | None] = []
        for a in author_entries:
            authors.append(a.get("full_name", ""))
            bai = None
            for id_obj in a.get("ids", []):
                if id_obj.get("schema") == "INSPIRE BAI":
                    bai = id_obj.get("value")
                    break
            if bai is None:
                bai = a.get("bai")
            author_bais.append(bai)

        arxiv = meta.get("arxiv_eprints", [{}])
        arxiv_id = arxiv[0].get("value") if arxiv else None
        arxiv_cats = arxiv[0].get("categories", []) if arxiv else []

        dois = meta.get("dois", [])
        doi = dois[0].get("value") if dois else None

        pub = meta.get("publication_info", [{}])
        journal = pub[0].get("journal_title") if pub else None

        return cls(
            recid=str(hit.get("id", "")),
            title=title,
            authors=authors,
            author_bais=author_bais,
            author_count=meta.get("author_count", len(authors)),
            arxiv_id=arxiv_id,
            arxiv_categories=arxiv_cats,
            doi=doi,
            citation_count=meta.get("citation_count", 0),
            date=meta.get("earliest_date"),
            journal=journal,
        )


# ======================================================================
# 1. Fetch papers for an author (fewer than N authors)
# ======================================================================

def get_author_papers(
    author_id: str,
    *,
    max_authors: int = 10,
    client: InspireClient | None = None,
) -> list[PaperInfo]:
    """Return all papers by *author_id* that have at most *max_authors* authors.

    Parameters
    ----------
    author_id:
        An INSPIRE BAI such as ``"M.B.Green.1"`` or a name query.
    max_authors:
        Only include papers with at most this many authors.
    client:
        An existing :class:`InspireClient`; one is created if not provided.
    """
    client = client or InspireClient()
    query = f"a {author_id} and ac 1->{max_authors}"
    hits = client.search_literature_all(
        query,
        fields="titles,authors,arxiv_eprints,dois,publication_info,"
               "citation_count,earliest_date,author_count",
    )
    return [PaperInfo.from_hit(h) for h in hits]


# ======================================================================
# 2. Count papers by arXiv category
# ======================================================================

@dataclass
class CategoryCount:
    category: str
    count: int


def count_arxiv_categories(
    author_id: str,
    *,
    max_authors: int = 10,
    client: InspireClient | None = None,
) -> list[CategoryCount]:
    """Count how many of *author_id*'s papers fall under each arXiv category.

    Only considers papers with at most *max_authors* authors (to exclude
    large-collaboration papers).  Returns a list sorted by count descending.
    """
    papers = get_author_papers(author_id, max_authors=max_authors, client=client)
    counter: Counter[str] = Counter()
    for p in papers:
        for cat in p.arxiv_categories:
            counter[cat] += 1
    return [CategoryCount(cat, n) for cat, n in counter.most_common()]


# ======================================================================
# 3. Collaboration network graph
# ======================================================================

@dataclass
class CollabEdge:
    author_a: str
    author_b: str
    num_papers: int
    weighted_citations: float


@dataclass
class CollabNetwork:
    """A collaboration network among a set of authors."""

    graph: nx.Graph
    edges: list[CollabEdge] = field(default_factory=list)
    decay: float = 1.0
    max_authors: int = 10
    author_papers: dict[str, dict[str, PaperInfo]] = field(
        default_factory=dict, repr=False,
    )
    author_order: list[str] = field(default_factory=list)

    def formula_str(self) -> str:
        return f"W = Sum_i[ c_i * exp(-lambda * age_i) ],  lambda={self.decay}"

    def summary(self) -> str:
        lines = [
            f"Collaboration network: {self.graph.number_of_nodes()} authors, "
            f"{self.graph.number_of_edges()} edges",
            self.formula_str(),
        ]
        for e in sorted(self.edges, key=lambda x: x.weighted_citations, reverse=True):
            lines.append(
                f"  {e.author_a} <-> {e.author_b}: "
                f"{e.num_papers} papers, W={e.weighted_citations:.2f}"
            )
        return "\n".join(lines)

    def plot(
        self,
        save_path: str | None = None,
        show: bool = True,
    ) -> None:
        """Visualize the collaboration network as an interactive HTML page.

        Two-panel layout with controls bar for adding/removing authors and
        tuning the decay parameter live.
        """
        from pyvis.network import Network as PyvisNetwork
        import webbrowser
        import os

        G = self.graph
        if G.number_of_edges() == 0:
            print("No collaboration edges to plot.")
            return

        save_path = save_path or "collab_network.html"

        # ── Prepare data for embedding as JSON ──────────────────────────
        # Preserve insertion order; fall back to dict key order
        network_bais = (
            self.author_order
            if self.author_order
            else list(self.author_papers.keys())
        )

        # Extract full name for each BAI from their papers
        author_full_names: dict[str, str] = {}
        for bai, papers_dict in self.author_papers.items():
            for p in papers_dict.values():
                for i, ab in enumerate(p.author_bais):
                    if ab == bai and i < len(p.authors):
                        author_full_names[bai] = p.authors[i]
                        break
                if bai in author_full_names:
                    break

        def _last_name(full_name: str) -> str:
            """Extract last name from INSPIRE 'Last, First' format."""
            comma = full_name.find(", ")
            return full_name[:comma] if comma >= 0 else full_name

        paper_network_bais: dict[str, set[str]] = {}
        for bai, papers_dict in self.author_papers.items():
            for recid in papers_dict:
                paper_network_bais.setdefault(recid, set()).add(bai)

        def _paper_json(p: PaperInfo) -> dict:
            age = _paper_age_years(p.date)
            wc = p.citation_count * math.exp(-self.decay * age)
            author_list = []
            for i, name in enumerate(p.authors):
                bai = p.author_bais[i] if i < len(p.author_bais) else None
                author_list.append({"name": name, "bai": bai})
            net_bais = sorted(paper_network_bais.get(p.recid, set()))
            return {
                "recid": p.recid,
                "title": p.title,
                "authors": author_list,
                "author_count": p.author_count,
                "arxiv_id": p.arxiv_id,
                "primary_category": (
                    p.arxiv_categories[0] if p.arxiv_categories else None
                ),
                "arxiv_categories": p.arxiv_categories,
                "doi": p.doi,
                "citation_count": p.citation_count,
                "date": p.date,
                "journal": p.journal,
                "weighted_citation": round(wc, 4),
                "network_bais": net_bais,
            }

        authors_data: dict[str, dict] = {}
        for bai, papers_dict in self.author_papers.items():
            papers = list(papers_dict.values())
            cat_counter: Counter[str] = Counter()
            for p in papers:
                for cat in p.arxiv_categories:
                    cat_counter[cat] += 1
            categories = [
                {"category": c, "count": n} for c, n in cat_counter.most_common()
            ]
            total_wc = _weighted_citations_for_papers(papers, self.decay)
            full_name = author_full_names.get(bai, "")
            authors_data[bai] = {
                "num_papers": len(papers),
                "weighted_citations": round(total_wc, 4),
                "categories": categories,
                "papers": [_paper_json(p) for p in papers],
                "full_name": full_name,
                "last_name": _last_name(full_name) if full_name else "",
            }

        edges_data: dict[str, dict] = {}
        for e in self.edges:
            shared_recids = (
                set(self.author_papers.get(e.author_a, {}))
                & set(self.author_papers.get(e.author_b, {}))
            )
            shared = [self.author_papers[e.author_a][r] for r in shared_recids]
            key = f"{e.author_a}|{e.author_b}"
            edges_data[key] = {
                "author_a": e.author_a,
                "author_b": e.author_b,
                "num_papers": e.num_papers,
                "weighted_citations": round(e.weighted_citations, 4),
                "papers": [_paper_json(p) for p in shared],
            }

        graph_data_json = json.dumps({
            "decay": self.decay,
            "max_authors": self.max_authors,
            "current_year": datetime.datetime.now().year,
            "network_bais": network_bais,
            "authors": authors_data,
            "edges": edges_data,
        }).replace("</", "<\\/")

        # ── Build pyvis graph ───────────────────────────────────────────
        # Node sizes based on total weighted citations per author
        node_wc: dict[str, float] = {}
        for bai, adata in authors_data.items():
            node_wc[bai] = adata["weighted_citations"]
        max_wc = max(node_wc.values()) if node_wc else 1.0
        min_wc = min(node_wc.values()) if node_wc else 0.0
        max_log_wc = math.log1p(max_wc)
        min_log_wc = math.log1p(min_wc)
        log_wc_range = max_log_wc - min_log_wc if max_log_wc > min_log_wc else 1.0

        intensities = [G[u][v]["weighted_citations"] for u, v in G.edges()]
        max_log = math.log1p(max(intensities)) if intensities else 1.0

        def _hyphenate_label(name: str, max_len: int = 9) -> str:
            """Break long names with a hyphen for better node fit."""
            if len(name) <= max_len:
                return name
            # Try to break at a natural point (space, hyphen)
            mid = len(name) // 2
            for offset in range(mid):
                for pos in [mid + offset, mid - offset]:
                    if 0 < pos < len(name) and name[pos] in " -":
                        return name[:pos + 1] + "\n" + name[pos + 1:]
            # Force break with hyphen
            return name[:mid] + "-\n" + name[mid:]

        net = PyvisNetwork(
            height="100%", width="100%",
            bgcolor="#ffffff",
            heading="",
        )
        net.barnes_hut(
            gravity=-3000, central_gravity=0.5,
            spring_length=150, spring_strength=0.06, damping=0.5,
        )

        # Nodes and edges are added procedurally by JS from GRAPH_DATA
        # (empty pyvis network — JS init populates it)

        # ── Header HTML ────────────────────────────────────────────────
        header_html = (
            '<div id="header">'
            '<div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap">'
            '<h2 style="margin:0;font-size:16px;white-space:nowrap">'
            'Collaboration Network</h2>'
            '<div style="font-size:13px">'
            '<i>W</i> = &Sigma;<sub>i</sub> '
            'c<sub>i</sub> &middot; '
            'e<sup>&minus;&lambda;&middot;age<sub>i</sub></sup>'
            '<span id="formula-help" style="cursor:pointer;margin-left:6px;'
            'color:#888;font-size:12px;border:1px solid #ccc;border-radius:50%;'
            'display:inline-block;width:16px;height:16px;text-align:center;'
            'line-height:16px" title="Click for explanation">?</span>'
            '</div>'
            '</div>'
            '<div id="formula-explanation" style="display:none;font-size:12px;'
            'color:#555;line-height:1.6;margin-top:6px;padding:8px 12px;'
            'background:#f9f9f9;border-radius:4px;border:1px solid #eee">'
            'The <b>weighted citation score</b> (<i>W</i>) measures the strength '
            'of collaboration between two authors. For each co-authored paper '
            '<i>i</i>, the paper\'s citation count <i>c<sub>i</sub></i> is '
            'multiplied by an exponential decay factor '
            '<i>e<sup>&minus;&lambda;&middot;age<sub>i</sub></sup></i>, '
            'where <i>age<sub>i</sub></i> is the number of years since '
            'publication and <i>&lambda;</i> (lambda) controls how quickly '
            'older papers lose weight. '
            'A higher &lambda; emphasises recent work; &lambda;=0 treats all '
            'papers equally regardless of age. '
            'The edge score is the sum over all shared papers between the pair.'
            '</div>'
            '</div>'
        )

        # ── Controls HTML ──────────────────────────────────────────────
        controls_html = (
            '<div id="controls">'
            '<div id="author-section">'
            '<div id="author-tags"></div>'
            '<div id="author-search-wrap">'
            '<input id="add-author-input" autocomplete="off" '
            'placeholder="Search by name (e.g. Witten)" />'
            '<div id="author-dropdown"></div>'
            '</div>'
            '<button id="add-author-btn">Add</button>'
            '<button id="clear-all-btn">Clear All</button>'
            '</div>'
            '<div id="lambda-section">'
            '<span>&lambda; =</span>'
            f'<input type="number" id="lambda-value" value="{self.decay}" '
            'step="0.01" min="0" />'
            '</div>'
            '</div>'
        )

        # ── Info panel placeholder ─────────────────────────────────────
        info_panel_html = (
            '</div>'  # close #graph-panel
            '<div id="resize-handle"></div>'
            '<div id="info-panel">'
            '<div id="info-content">'
            '<div class="welcome-msg">'
            '<p style="font-size:16px;font-weight:600;color:#333;margin-bottom:12px">'
            'Collaboration Network</p>'
            '<p><b>Nodes</b> represent authors. '
            'Size reflects weighted citations.</p>'
            '<p><b>Edges</b> connect co-authors. '
            'Thickness and colour reflect the weighted citation score '
            'of their shared papers.</p>'
            '<p style="margin-top:16px"><b>How to explore:</b></p>'
            '<ul style="margin:4px 0 0 0;padding-left:20px">'
            '<li>Click a node to see an author\'s papers and categories</li>'
            '<li>Click an edge to see shared publications</li>'
            '<li>Hover over nodes to highlight connections</li>'
            '<li>Use Tab/Shift+Tab to cycle through authors</li>'
            '<li>Press Escape to deselect</li>'
            '</ul>'
            '</div>'
            '</div>'
            '</div>'
            '</div>'  # close #main-container
        )

        # ── CSS ────────────────────────────────────────────────────────
        custom_css = (
            '<style>'
            'html,body{margin:0;padding:0;height:100%;overflow:hidden;'
            'font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}'
            'body{display:flex;flex-direction:column}'
            '#header{padding:8px 20px;border-bottom:1px solid #e0e0e0;'
            'background:#fafafa;flex-shrink:0}'
            '#controls{display:flex;align-items:center;gap:12px;padding:6px 20px;'
            'border-bottom:1px solid #e0e0e0;background:#f5f5f5;flex-shrink:0;'
            'flex-wrap:wrap}'
            '#author-section{display:flex;align-items:center;gap:5px;'
            'flex-wrap:wrap;flex:1;min-width:0}'
            '#author-tags{display:flex;gap:4px;flex-wrap:wrap}'
            '.author-tag{display:inline-flex;align-items:center;gap:3px;'
            'background:#4C72B0;color:#fff;padding:3px 10px;'
            'border-radius:12px;font-size:14px;white-space:nowrap}'
            '.author-tag-name{cursor:pointer}'
            '.author-tag-name:hover{text-decoration:underline}'
            '.author-tag button{background:none;border:none;color:#fff;'
            'cursor:pointer;font-size:16px;padding:0 2px;opacity:.7;'
            'line-height:1}'
            '.author-tag button:hover{opacity:1}'
            '#author-search-wrap{position:relative;display:inline-block}'
            '#add-author-input{padding:5px 10px;border:1px solid #ccc;'
            'border-radius:4px;font-size:14px;width:300px;font-family:inherit}'
            '#author-dropdown{display:none;position:absolute;top:100%;left:0;'
            'width:100%;min-width:360px;max-width:calc(100vw - 40px);'
            'background:#fff;border:1px solid #ddd;'
            'border-radius:6px;box-shadow:0 4px 16px rgba(0,0,0,.15);'
            'z-index:10000;max-height:320px;overflow-y:auto;margin-top:2px}'
            '.author-option{padding:8px 12px;cursor:pointer;border-bottom:1px solid #f0f0f0}'
            '.author-option:last-child{border-bottom:none}'
            '.author-option:hover,.author-option.active{background:#f0f4fa}'
            '.author-option-name{font-size:14px;font-weight:600;color:#222}'
            '.author-option-meta{font-size:12px;color:#888;margin-top:2px}'
            '.author-option-bai{color:#4C72B0;font-family:monospace}'
            '.author-dropdown-empty{padding:10px 12px;font-size:13px;color:#999;'
            'font-style:italic}'
            '.author-dropdown-loading{padding:10px 12px;font-size:13px;color:#888}'
            '#add-author-btn{padding:5px 14px;background:#4C72B0;color:#fff;'
            'border:none;border-radius:4px;cursor:pointer;font-size:14px;'
            'font-family:inherit}'
            '#add-author-btn:hover{background:#3a5a8c}'
            '#add-author-btn:disabled{opacity:.5;cursor:not-allowed}'
            '#clear-all-btn{padding:5px 14px;background:#888;color:#fff;'
            'border:none;border-radius:4px;cursor:pointer;font-size:14px;'
            'font-family:inherit}'
            '#clear-all-btn:hover{background:#666}'
            '#lambda-section{display:flex;align-items:center;gap:8px;'
            'font-size:15px;color:#444;flex-shrink:0}'
            '#lambda-value{width:60px;padding:4px 6px;border:1px solid #ccc;'
            'border-radius:4px;font-size:14px;text-align:center;font-family:inherit}'
            '#main-container{flex:1;min-height:0;display:flex}'
            '#graph-panel{flex:1;min-width:0;position:relative;overflow:hidden}'
            '#zoom-controls{position:absolute;top:10px;left:10px;z-index:100;'
            'display:flex;flex-direction:column;gap:4px}'
            '.zoom-btn{width:32px;height:32px;border:1px solid #ccc;border-radius:4px;'
            'background:#fff;font-size:18px;cursor:pointer;display:flex;'
            'align-items:center;justify-content:center;color:#444;'
            'box-shadow:0 1px 4px rgba(0,0,0,.1);line-height:1}'
            '.zoom-btn:hover{background:#f0f4fa;border-color:#4C72B0;color:#4C72B0}'
            '#graph-panel .card{width:100%!important;height:100%!important;'
            'margin:0!important;border:none!important}'
            '#graph-panel .card-body,#graph-panel #mynetwork{width:100%!important;'
            'height:100%!important;border:none!important;padding:0!important}'
            '#resize-handle{width:5px;cursor:col-resize;background:#e0e0e0;'
            'flex-shrink:0;transition:background .15s}'
            '#resize-handle:hover,#resize-handle.active{background:#4C72B0}'
            '#info-panel{width:420px;min-width:200px;border-left:1px solid #ddd;'
            'overflow-y:auto;background:#fcfcfc;padding:0}'
            '#info-content{padding:16px}'
            '.sort-row{font-size:14px;color:#666;margin-bottom:10px}'
            '#sort-select{font-size:14px;padding:4px 8px;border:1px solid #ccc;'
            'border-radius:4px;margin-left:4px;font-family:inherit}'
            '.paper-entry{padding:12px 0;border-bottom:1px solid #f0f0f0}'
            '.paper-entry:last-child{border-bottom:none}'
            '.paper-meta{font-size:14px;margin-bottom:4px}'
            '.paper-cat{background:#e8eaf6;color:#283593;padding:2px 8px;'
            'border-radius:3px;font-size:13px;font-weight:500;margin-right:6px}'
            '.paper-arxiv{color:#1565c0;text-decoration:none;margin-right:8px;'
            'font-size:14px}'
            '.paper-arxiv:hover{text-decoration:underline}'
            '.paper-date{color:#999;font-size:13px}'
            '.paper-title{font-size:15px;font-weight:500;margin:4px 0;color:#222}'
            '.paper-authors{font-size:14px;color:#555;line-height:1.5}'
            '.net-author{font-weight:700;color:#1565c0;cursor:pointer}'
            '.net-author:hover{text-decoration:underline}'
            '.paper-stats{font-size:13px;color:#999;margin-top:4px}'
            '.cat-row{display:flex;align-items:center;margin:3px 0}'
            '.cat-name{width:120px;font-size:14px;color:#444}'
            '.cat-bar{height:16px;background:#4C72B0;border-radius:2px;'
            'min-width:2px;margin-right:8px}'
            '.cat-count{font-size:13px;color:#888}'
            '.info-section{margin-bottom:18px}'
            '.info-label{font-size:13px;color:#999;text-transform:uppercase;'
            'letter-spacing:.5px;margin-bottom:4px}'
            '.info-value{font-size:16px;color:#333}'
            '.info-title{font-size:18px;font-weight:600;color:#222;'
            'margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #4C72B0}'
            '.edge-title{font-size:16px;font-weight:600;color:#222;'
            'margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #4C72B0}'
            '.welcome-msg{padding:20px;color:#666;font-size:14px;line-height:1.7}'
            '.welcome-msg ul{line-height:2}'
            '#controls-toggle-bar{padding:4px 20px;background:#f5f5f5;'
            'border-bottom:1px solid #e0e0e0;flex-shrink:0}'
            '#controls-toggle{background:none;border:1px solid #ccc;border-radius:4px;'
            'cursor:pointer;font-size:12px;padding:2px 10px;color:#666;font-family:inherit}'
            '#controls-toggle:hover{background:#e8e8e8}'
            '#edge-tooltip{display:none;position:fixed;pointer-events:none;'
            'background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 12px;'
            'box-shadow:0 2px 8px rgba(0,0,0,.15);font-size:13px;z-index:9999;'
            'max-width:280px;line-height:1.5}'
            '.back-link{cursor:pointer;color:#1565c0;font-size:13px;'
            'text-decoration:none;display:inline-block;margin-bottom:8px}'
            '.back-link:hover{text-decoration:underline}'
            '.filter-bar{margin-bottom:10px}'
            '#paper-search{width:100%;padding:6px 10px;border:1px solid #ccc;'
            'border-radius:4px;font-size:13px;margin-bottom:6px;'
            'box-sizing:border-box;font-family:inherit}'
            '.filter-row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;'
            'font-size:13px;color:#666}'
            '#cat-filter{font-size:13px;padding:4px 8px;border:1px solid #ccc;'
            'border-radius:4px;font-family:inherit}'
            '@media(max-width:768px){'
            '#main-container{flex-direction:column}'
            '#graph-panel{min-height:50vh}'
            '#info-panel{width:100%!important;min-width:0!important;'
            'max-height:50vh;border-left:none;border-top:1px solid #ddd}'
            '#resize-handle{display:none}'
            '#add-author-input{width:180px}'
            '#author-dropdown{min-width:0}'
            '}'
            '</style>'
        )

        # ── JavaScript ─────────────────────────────────────────────────
        js_template = r'''
var GRAPH_DATA = __GRAPH_DATA__;
var globalSort = "recent";
var currentYear = GRAPH_DATA.current_year || 2026;
var currentInfoType = null;
var currentInfoId = null;
var navHistory = [];
var currentPapers = [];
var userHasZoomed = false;
var addingAuthor = false;

// ── Utilities ──────────────────────────────────────
function paperAge(d) {
    if (!d) return 30;
    var y = parseInt(d.substring(0,4));
    return isNaN(y) ? 30 : Math.max(currentYear - y, 0);
}
function computeWC(p, decay) {
    return p.citation_count * Math.exp(-decay * paperAge(p.date));
}
function flipName(s) {
    if (!s) return "";
    var i = s.indexOf(", ");
    return i < 0 ? s : s.substring(i+2) + " " + s.substring(0,i);
}
function esc(s) {
    if (!s) return "";
    return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function escA(s) {
    if (!s) return "";
    return String(s).replace(/\\/g,"\\\\").replace(/'/g,"\\'");
}
function hex2(n) { return ("0"+Math.round(n).toString(16)).slice(-2); }
function shortBai(b) {
    var p = b.split(".");
    return p.length >= 2 ? p[p.length-2]+"."+p[p.length-1] : b;
}
function getMaxLog() {
    var mx = 0;
    var keys = Object.keys(GRAPH_DATA.edges);
    for (var i=0;i<keys.length;i++) {
        var w = GRAPH_DATA.edges[keys[i]].weighted_citations;
        if (w > mx) mx = w;
    }
    return Math.log1p(mx) || 1;
}
function edgeColor(intensity, maxLog) {
    var t = maxLog > 0 ? Math.log1p(intensity)/maxLog : 0;
    t = Math.min(t,1);
    return "#"+hex2(204-t*178)+hex2(204-t*146)+hex2(204-t*97);
}

// ── Weight recomputation ───────────────────────────
function recomputeAllWeights() {
    var decay = GRAPH_DATA.decay;
    var akeys = Object.keys(GRAPH_DATA.authors);
    for (var i=0;i<akeys.length;i++) {
        var a = GRAPH_DATA.authors[akeys[i]]; var tot=0;
        for (var j=0;j<a.papers.length;j++) {
            var w = computeWC(a.papers[j], decay);
            a.papers[j].weighted_citation = w; tot += w;
        }
        a.weighted_citations = tot;
    }
    var ekeys = Object.keys(GRAPH_DATA.edges);
    for (var i=0;i<ekeys.length;i++) {
        var e = GRAPH_DATA.edges[ekeys[i]]; var tot=0;
        for (var j=0;j<e.papers.length;j++) {
            var w = computeWC(e.papers[j], decay);
            e.papers[j].weighted_citation = w; tot += w;
        }
        e.weighted_citations = tot;
    }
}

function hyphenateLabel(name, maxLen) {
    maxLen = maxLen || 9;
    if (name.length <= maxLen) return name;
    var mid = Math.floor(name.length / 2);
    for (var off=0; off<mid; off++) {
        var positions = [mid+off, mid-off];
        for (var pi=0; pi<positions.length; pi++) {
            var pos = positions[pi];
            if (pos > 0 && pos < name.length && (name[pos]===" "||name[pos]==="-")) {
                return name.substring(0, pos+1) + "\n" + name.substring(pos+1);
            }
        }
    }
    return name.substring(0, mid) + "-\n" + name.substring(mid);
}

function computeNodeSizeFont(wc, minLogWC, logWCRange) {
    var t = logWCRange > 0 ? (Math.log1p(wc) - minLogWC) / logWCRange : 0;
    return {margin: Math.floor(5 + 25 * t), fontSize: Math.max(11, Math.floor(13 + 5 * t))};
}

function getNodeWCRange() {
    var authors = GRAPH_DATA.authors;
    var bais = Object.keys(authors);
    var maxWC = 0, minWC = Infinity;
    for (var i=0;i<bais.length;i++) {
        var wc = authors[bais[i]].weighted_citations;
        if (wc > maxWC) maxWC = wc;
        if (wc < minWC) minWC = wc;
    }
    if (minWC === Infinity) minWC = 0;
    var maxLogWC = Math.log1p(maxWC);
    var minLogWC = Math.log1p(minWC);
    var logWCRange = maxLogWC > minLogWC ? maxLogWC - minLogWC : 1;
    return {minLogWC: minLogWC, logWCRange: logWCRange};
}

function updateGraphVisuals() {
    // Recompute node sizes based on weighted citations
    var nodeDS = network.body.data.nodes;
    var authors = GRAPH_DATA.authors;
    var bais = Object.keys(authors);
    var wcr = getNodeWCRange();
    var nodeUpdates = [];
    for (var i=0;i<bais.length;i++) {
        var b = bais[i];
        var sf = computeNodeSizeFont(authors[b].weighted_citations, wcr.minLogWC, wcr.logWCRange);
        nodeUpdates.push({id:b, margin:sf.margin,
            font:{size:sf.fontSize, color:"#ffffff", face:"arial", multi:true, strokeWidth:2, strokeColor:"#2a4a7a"}});
    }
    nodeDS.update(nodeUpdates);

    // Update edges
    var maxLog = getMaxLog();
    var edgeDS = network.body.data.edges;
    var all = edgeDS.get();
    for (var i=0;i<all.length;i++) {
        var ve = all[i];
        var nodes = network.getConnectedNodes(ve.id);
        var key = nodes[0]+"|"+nodes[1];
        var data = GRAPH_DATA.edges[key];
        if (!data) { key = nodes[1]+"|"+nodes[0]; data = GRAPH_DATA.edges[key]; }
        if (!data) continue;
        var intensity = data.weighted_citations;
        var logW = Math.log1p(intensity);
        edgeDS.update({
            id: ve.id, value: 1+8*logW/maxLog,
            color: edgeColor(intensity, maxLog)
        });
    }
}

// ── Author management ──────────────────────────────
function renderAuthorTags() {
    var c = document.getElementById("author-tags"); var h = "";
    var bais = GRAPH_DATA.network_bais;
    for (var i=0;i<bais.length;i++) {
        h += '<span class="author-tag"><span class="author-tag-name" onclick="selectNode(\''+escA(bais[i])+'\')">'+esc(bais[i])+'</span> <button onclick="removeAuthor(\''+escA(bais[i])+'\')">&times;</button></span>';
    }
    c.innerHTML = h;
}

function parseHit(hit) {
    var m = hit.metadata||{};
    var titles = m.titles||[];
    var title = titles.length ? (titles[0].title||"") : "";
    var ae = m.authors||[], authors=[];
    for (var i=0;i<ae.length;i++) {
        var a=ae[i], bai=null, ids=a.ids||[];
        for (var j=0;j<ids.length;j++) { if(ids[j].schema==="INSPIRE BAI"){bai=ids[j].value;break;} }
        if(!bai) bai=a.bai||null;
        authors.push({name:a.full_name||"", bai:bai});
    }
    var ax=m.arxiv_eprints||[], af=ax.length?ax[0]:{};
    var dois=m.dois||[], pub=m.publication_info||[];
    return {
        recid:String(hit.id||""), title:title, authors:authors,
        author_count:m.author_count||authors.length,
        arxiv_id:af.value||null,
        primary_category:(af.categories||[]).length?(af.categories[0]):null,
        arxiv_categories:af.categories||[],
        doi:dois.length?(dois[0].value||null):null,
        citation_count:m.citation_count||0,
        date:m.earliest_date||null,
        journal:pub.length?(pub[0].journal_title||null):null,
        weighted_citation:0, network_bais:[]
    };
}

async function fetchAuthorPapers(bai) {
    var q = "a "+bai+" and ac 1->"+GRAPH_DATA.max_authors;
    var all=[], page=1;
    while (true) {
        var params = new URLSearchParams({
            q:q, size:"250", page:String(page), sort:"mostrecent",
            fields:"titles,authors,arxiv_eprints,dois,publication_info,citation_count,earliest_date,author_count"
        });
        var resp = await fetch("https://inspirehep.net/api/literature?"+params);
        if (!resp.ok) throw new Error("API error: "+resp.status);
        var data = await resp.json();
        var batch = (data.hits||{}).hits||[];
        for (var i=0;i<batch.length;i++) all.push(parseHit(batch[i]));
        var total = (data.hits||{}).total||0;
        if (!batch.length || all.length >= total) break;
        page++;
        await new Promise(function(r){setTimeout(r,400);});
    }
    return all;
}

// ── Author search autocomplete ───────────────────
var searchTimer = null;
var searchAbort = null;
var dropdownIndex = -1;
var dropdownResults = [];

async function searchAuthors(query) {
    if (searchAbort) searchAbort.abort();
    var controller = new AbortController();
    searchAbort = controller;
    var dropdown = document.getElementById("author-dropdown");
    dropdown.innerHTML = '<div class="author-dropdown-loading">Searching...</div>';
    dropdown.style.display = "block";
    try {
        var params = new URLSearchParams({q:query, size:"8",
            fields:"ids,name,positions,advisors"});
        var resp = await fetch("https://inspirehep.net/api/authors?"+params,
            {signal:controller.signal});
        if (!resp.ok) throw new Error("API error");
        var data = await resp.json();
        var hits = (data.hits||{}).hits||[];
        dropdownResults = [];
        dropdownIndex = -1;
        if (!hits.length) {
            dropdown.innerHTML = '<div class="author-dropdown-empty">No authors found</div>';
            return;
        }
        var html = "";
        for (var i=0;i<hits.length;i++) {
            var m = hits[i].metadata||{};
            var name = (m.name||{}).preferred_name || (m.name||{}).value || "";
            var bai = "";
            var ids = m.ids||[];
            for (var j=0;j<ids.length;j++) {
                if (ids[j].schema === "INSPIRE BAI") { bai = ids[j].value; break; }
            }
            if (!bai) continue;
            var inst = "";
            var positions = m.positions||[];
            for (var j=0;j<positions.length;j++) {
                if (positions[j].current) { inst = positions[j].institution||""; break; }
            }
            if (!inst && positions.length) inst = positions[0].institution||"";
            var already = GRAPH_DATA.authors[bai] ? " (already added)" : "";
            dropdownResults.push({bai:bai, name:name, institution:inst});
            html += '<div class="author-option" data-index="'+dropdownResults.length+'" data-bai="'+esc(bai)+'">'
                + '<div class="author-option-name">'+esc(name)+already+'</div>'
                + '<div class="author-option-meta">'
                + '<span class="author-option-bai">'+esc(bai)+'</span>'
                + (inst ? ' &middot; '+esc(inst) : '')
                + '</div></div>';
        }
        dropdown.innerHTML = html || '<div class="author-dropdown-empty">No authors found</div>';
        var options = dropdown.querySelectorAll(".author-option");
        for (var i=0;i<options.length;i++) {
            (function(opt){
                opt.addEventListener("mousedown", function(e) {
                    e.preventDefault();
                    selectDropdownAuthor(opt.getAttribute("data-bai"));
                });
            })(options[i]);
        }
    } catch(e) {
        if (e.name !== "AbortError") {
            dropdown.innerHTML = '<div class="author-dropdown-empty">Search failed</div>';
        }
    }
}

function selectDropdownAuthor(bai) {
    var input = document.getElementById("add-author-input");
    var dropdown = document.getElementById("author-dropdown");
    input.value = bai;
    dropdown.style.display = "none";
    dropdownResults = [];
    dropdownIndex = -1;
    addAuthor(bai);
    input.value = "";
}

function navigateDropdown(dir) {
    var dropdown = document.getElementById("author-dropdown");
    var options = dropdown.querySelectorAll(".author-option");
    if (!options.length) return;
    if (dropdownIndex >= 0 && dropdownIndex < options.length)
        options[dropdownIndex].classList.remove("active");
    dropdownIndex += dir;
    if (dropdownIndex < 0) dropdownIndex = options.length - 1;
    if (dropdownIndex >= options.length) dropdownIndex = 0;
    options[dropdownIndex].classList.add("active");
    options[dropdownIndex].scrollIntoView({block:"nearest"});
}

function updateNetworkBais() {
    var baiToRecids = {};
    var akeys = Object.keys(GRAPH_DATA.authors);
    for (var i=0;i<akeys.length;i++) {
        var papers = GRAPH_DATA.authors[akeys[i]].papers;
        for (var j=0;j<papers.length;j++) {
            var r = papers[j].recid;
            if (!baiToRecids[r]) baiToRecids[r] = [];
            if (baiToRecids[r].indexOf(akeys[i]) < 0) baiToRecids[r].push(akeys[i]);
        }
    }
    for (var i=0;i<akeys.length;i++) {
        var papers = GRAPH_DATA.authors[akeys[i]].papers;
        for (var j=0;j<papers.length;j++) papers[j].network_bais = baiToRecids[papers[j].recid]||[];
    }
    var ekeys = Object.keys(GRAPH_DATA.edges);
    for (var i=0;i<ekeys.length;i++) {
        var papers = GRAPH_DATA.edges[ekeys[i]].papers;
        for (var j=0;j<papers.length;j++) papers[j].network_bais = baiToRecids[papers[j].recid]||[];
    }
}

async function addAuthor(bai) {
    bai = bai.trim();
    if (!bai) return;
    if (addingAuthor) return;
    if (GRAPH_DATA.authors[bai]) { alert(bai+" is already in the network."); return; }
    addingAuthor = true;
    var btn = document.getElementById("add-author-btn");
    var input = document.getElementById("add-author-input");
    btn.disabled = true; btn.textContent = "Fetching...";
    try {
        var papers = await fetchAuthorPapers(bai);
        if (!papers.length) { alert("No papers found for "+bai); return; }
        var decay = GRAPH_DATA.decay;
        var byId = {};
        for (var i=0;i<papers.length;i++) {
            papers[i].weighted_citation = computeWC(papers[i], decay);
            papers[i].network_bais = [bai];
            byId[papers[i].recid] = papers[i];
        }
        var catCount = {};
        for (var i=0;i<papers.length;i++) {
            var cats = papers[i].arxiv_categories||[];
            for (var j=0;j<cats.length;j++) catCount[cats[j]] = (catCount[cats[j]]||0)+1;
        }
        var categories = Object.keys(catCount).map(function(c){return{category:c,count:catCount[c]};}).sort(function(a,b){return b.count-a.count;});
        var totalWC = 0;
        for (var i=0;i<papers.length;i++) totalWC += papers[i].weighted_citation;
        var fullName="", lastName="";
        for (var i=0;i<papers.length&&!fullName;i++) {
            var auList = papers[i].authors||[];
            for (var j=0;j<auList.length;j++) {
                if (auList[j].bai===bai && auList[j].name) {
                    fullName=auList[j].name;
                    var ci=fullName.indexOf(", ");
                    lastName=ci>=0?fullName.substring(0,ci):fullName;
                    break;
                }
            }
        }
        GRAPH_DATA.authors[bai] = {num_papers:papers.length, weighted_citations:totalWC, categories:categories, papers:papers, full_name:fullName, last_name:lastName};
        GRAPH_DATA.network_bais.push(bai);

        // Compute edges with existing authors
        var nodeDS = network.body.data.nodes;
        var edgeDS = network.body.data.edges;
        var existing = Object.keys(GRAPH_DATA.authors);
        for (var i=0;i<existing.length;i++) {
            var other = existing[i];
            if (other === bai) continue;
            var op = GRAPH_DATA.authors[other].papers;
            var oById = {};
            for (var j=0;j<op.length;j++) oById[op[j].recid] = true;
            var shared = [];
            for (var j=0;j<papers.length;j++) { if(oById[papers[j].recid]) shared.push(papers[j]); }
            if (shared.length) {
                var ew = 0;
                for (var j=0;j<shared.length;j++) ew += shared[j].weighted_citation;
                GRAPH_DATA.edges[bai+"|"+other] = {author_a:bai, author_b:other, num_papers:shared.length, weighted_citations:ew, papers:shared};
            }
        }
        updateNetworkBais();
        addAuthorToVis(bai);
        updateGraphVisuals();
        renderAuthorTags();
        input.value = "";
    } catch(e) { alert("Error fetching "+bai+": "+e.message); }
    finally { addingAuthor = false; btn.disabled = false; btn.textContent = "Add"; }
}

function addAuthorToVis(bai) {
    var data = GRAPH_DATA.authors[bai];
    if (!data) return;
    var nodeDS = network.body.data.nodes;
    var edgeDS = network.body.data.edges;
    var wcr = getNodeWCRange();
    var sf = computeNodeSizeFont(data.weighted_citations, wcr.minLogWC, wcr.logWCRange);
    var nodeColor = {background:"#4C72B0",border:"#3a5a8c",highlight:{background:"#5c8fd6",border:"#3a5a8c"}};
    var nodeFont = {size:sf.fontSize,color:"#ffffff",face:"arial",multi:true,strokeWidth:2,strokeColor:"#2a4a7a"};
    var lastName = data.last_name || data.full_name || bai;
    var nodeLabel = hyphenateLabel(lastName);
    nodeDS.add({id:bai, label:nodeLabel, shape:"circle", margin:sf.margin,
        color:nodeColor, font:nodeFont, title:bai+"\nWeighted citations: "+data.weighted_citations.toFixed(1)
    });
    var maxLog = getMaxLog();
    var ekeys = Object.keys(GRAPH_DATA.edges);
    for (var i=0;i<ekeys.length;i++) {
        var e = GRAPH_DATA.edges[ekeys[i]];
        if (e.author_a === bai || e.author_b === bai) {
            var otherBai = e.author_a === bai ? e.author_b : e.author_a;
            if (!nodeDS.get(otherBai)) continue;
            var logW = Math.log1p(e.weighted_citations);
            edgeDS.add({from:e.author_a, to:e.author_b, value:1+8*logW/maxLog,
                color:edgeColor(e.weighted_citations, maxLog)
            });
        }
    }
}

function removeAuthor(bai) {
    if (!GRAPH_DATA.authors[bai]) return;
    delete GRAPH_DATA.authors[bai];
    GRAPH_DATA.network_bais = GRAPH_DATA.network_bais.filter(function(b){return b!==bai;});
    var ekeys = Object.keys(GRAPH_DATA.edges);
    for (var i=0;i<ekeys.length;i++) {
        var e = GRAPH_DATA.edges[ekeys[i]];
        if (e.author_a===bai || e.author_b===bai) delete GRAPH_DATA.edges[ekeys[i]];
    }
    updateNetworkBais();
    var connEdges = network.getConnectedEdges(bai);
    network.body.data.edges.remove(connEdges);
    network.body.data.nodes.remove(bai);
    updateGraphVisuals();
    renderAuthorTags();
    if (currentInfoType==="node" && currentInfoId===bai) { clearInfoPanel(); }
}

function clearAllAuthors() {
    var bais = GRAPH_DATA.network_bais.slice();
    for (var i=0;i<bais.length;i++) {
        var connEdges = network.getConnectedEdges(bais[i]);
        network.body.data.edges.remove(connEdges);
        network.body.data.nodes.remove(bais[i]);
    }
    GRAPH_DATA.authors = {};
    GRAPH_DATA.edges = {};
    GRAPH_DATA.network_bais = [];
    updateNetworkBais();
    renderAuthorTags();
    clearInfoPanel();
}

function clearInfoPanel() {
    document.getElementById("info-content").innerHTML =
        '<div class="welcome-msg">'
        +'<p style="font-size:16px;font-weight:600;color:#333;margin-bottom:12px">Collaboration Network</p>'
        +'<p><b>Nodes</b> represent authors. Size reflects weighted citations.</p>'
        +'<p><b>Edges</b> connect co-authors. Thickness and colour reflect the weighted citation score of their shared papers.</p>'
        +'<p style="margin-top:16px"><b>How to explore:</b></p>'
        +'<ul style="margin:4px 0 0 0;padding-left:20px"><li>Click a node to see an author\'s papers and categories</li>'
        +'<li>Click an edge to see shared publications</li><li>Hover over nodes to highlight connections</li>'
        +'<li>Use Tab/Shift+Tab to cycle through authors</li><li>Press Escape to deselect</li></ul></div>';
    currentInfoType = null; currentInfoId = null;
    navHistory = []; currentPapers = [];
}

// ── Info panel ─────────────────────────────────────
function refreshInfoPanel() {
    if (currentInfoType==="node") showNodeInfo(currentInfoId);
    else if (currentInfoType==="edge") {
        var p = currentInfoId.split("|");
        showEdgeInfo(p[0], p[1]);
    }
}

function categoriesHTML(cats) {
    if (!cats || !cats.length) return "";
    var mx = cats[0].count;
    var h = '<div class="info-section"><div class="info-label">arXiv Categories</div>';
    for (var i=0;i<cats.length;i++) {
        var ct = cats[i], bw = Math.max(4, Math.round(140*ct.count/mx));
        h += '<div class="cat-row"><span class="cat-name">'+esc(ct.category)+'</span>';
        h += '<div class="cat-bar" style="width:'+bw+'px"></div>';
        h += '<span class="cat-count">'+ct.count+'</span></div>';
    }
    return h + '</div>';
}

function edgeCategoriesFromPapers(papers) {
    var cc = {};
    for (var i=0;i<papers.length;i++) {
        var cats = papers[i].arxiv_categories||[];
        for (var j=0;j<cats.length;j++) cc[cats[j]] = (cc[cats[j]]||0)+1;
    }
    return Object.keys(cc).map(function(c){return{category:c,count:cc[c]};}).sort(function(a,b){return b.count-a.count;});
}

function pushNav() {
    if (currentInfoType !== null) {
        navHistory.push({type:currentInfoType, id:currentInfoId});
        if (navHistory.length > 50) navHistory.shift();
    }
}
function goBack() {
    if (!navHistory.length) return;
    var prev = navHistory.pop();
    if (prev.type === "node") showNodeInfo(prev.id, true);
    else if (prev.type === "edge") {
        var p = prev.id.split("|");
        showEdgeInfo(p[0], p[1], true);
    }
}
function backButtonHTML() {
    if (!navHistory.length) return "";
    return '<a class="back-link" onclick="goBack()">&larr; Back</a>';
}

function filterBarHTML(papers) {
    var cats = {};
    for (var i=0;i<papers.length;i++) {
        if (papers[i].primary_category) cats[papers[i].primary_category] = true;
    }
    var catKeys = Object.keys(cats).sort();
    var h = '<div class="filter-bar">';
    h += '<input id="paper-search" placeholder="Search papers..." />';
    h += '<div class="filter-row">';
    h += '<select id="cat-filter"><option value="">All categories</option>';
    for (var i=0;i<catKeys.length;i++) h += '<option value="'+esc(catKeys[i])+'">'+esc(catKeys[i])+'</option>';
    h += '</select>';
    h += ' Sort: <select id="sort-select">'
       +'<option value="recent"'+(globalSort==="recent"?" selected":"")+'>Most Recent</option>'
       +'<option value="weighted"'+(globalSort==="weighted"?" selected":"")+'>Most Weighted</option>'
       +'<option value="cited"'+(globalSort==="cited"?" selected":"")+'>Most Cited</option>'
       +'</select>';
    h += '</div></div>';
    return h;
}
function filterAndRenderPapers() {
    var query = (document.getElementById("paper-search")||{}).value||"";
    query = query.toLowerCase();
    var catFilter = (document.getElementById("cat-filter")||{}).value||"";
    var filtered = currentPapers.filter(function(p) {
        if (query && p.title.toLowerCase().indexOf(query) < 0) return false;
        if (catFilter && p.primary_category !== catFilter) return false;
        return true;
    });
    var el = document.getElementById("paper-list");
    if (el) el.innerHTML = renderPapers(filtered, globalSort);
}
function bindFilterBar() {
    var search = document.getElementById("paper-search");
    if (search) search.addEventListener("input", filterAndRenderPapers);
    var catSel = document.getElementById("cat-filter");
    if (catSel) catSel.addEventListener("change", filterAndRenderPapers);
    var sortSel = document.getElementById("sort-select");
    if (sortSel) sortSel.addEventListener("change", function() {
        globalSort = this.value; filterAndRenderPapers();
    });
}

function showNodeInfo(bai, isBack) {
    var data = GRAPH_DATA.authors[bai];
    if (!data) return;
    if (!isBack) pushNav();
    currentInfoType = "node"; currentInfoId = bai;
    currentPapers = data.papers;
    var el = document.getElementById("info-content");
    var dispName = data.full_name ? flipName(data.full_name) : bai;
    var inspireUrl = "https://inspirehep.net/literature?sort=mostrecent&q=a+"+encodeURIComponent(bai);
    var h = backButtonHTML();
    h += '<div class="info-title">'+esc(dispName)+'</div>';
    h += '<div style="font-size:12px;color:#666;margin-top:-8px;margin-bottom:12px">';
    h += esc(bai)+' &middot; <a href="'+inspireUrl+'" target="_blank" style="color:#1565c0;text-decoration:none">View on INSPIRE</a>';
    h += '</div>';
    h += '<div class="info-section"><div style="display:flex;gap:24px">';
    h += '<div><div class="info-label">Papers</div><div class="info-value">'+data.num_papers+'</div></div>';
    h += '<div><div class="info-label">Weighted Citations</div><div class="info-value">'+data.weighted_citations.toFixed(2)+'</div></div>';
    h += '</div></div>';
    h += categoriesHTML(data.categories);
    h += '<div class="info-section"><div class="info-label">Papers</div>';
    h += filterBarHTML(data.papers);
    h += '<div id="paper-list">'+renderPapers(data.papers, globalSort)+'</div></div>';
    el.innerHTML = h;
    bindFilterBar();
}

function showEdgeInfo(a, b, isBack) {
    var key = a+"|"+b;
    var data = GRAPH_DATA.edges[key];
    if (!data) { key = b+"|"+a; data = GRAPH_DATA.edges[key]; }
    if (!data) return;
    if (!isBack) pushNav();
    currentInfoType = "edge"; currentInfoId = key;
    currentPapers = data.papers;
    var el = document.getElementById("info-content");
    var nameA = GRAPH_DATA.authors[data.author_a] && GRAPH_DATA.authors[data.author_a].full_name
        ? flipName(GRAPH_DATA.authors[data.author_a].full_name) : data.author_a;
    var nameB = GRAPH_DATA.authors[data.author_b] && GRAPH_DATA.authors[data.author_b].full_name
        ? flipName(GRAPH_DATA.authors[data.author_b].full_name) : data.author_b;
    var h = backButtonHTML();
    h += '<div class="edge-title">';
    h += '<a class="net-author" onclick="selectNode(\''+escA(data.author_a)+'\')">'+esc(nameA)+'</a>';
    h += ' &harr; ';
    h += '<a class="net-author" onclick="selectNode(\''+escA(data.author_b)+'\')">'+esc(nameB)+'</a>';
    h += '</div>';
    h += '<div class="info-section"><div style="display:flex;gap:24px">';
    h += '<div><div class="info-label">Shared Papers</div><div class="info-value">'+data.num_papers+'</div></div>';
    h += '<div><div class="info-label">Weighted Citations</div><div class="info-value">'+data.weighted_citations.toFixed(2)+'</div></div>';
    h += '</div></div>';
    h += categoriesHTML(edgeCategoriesFromPapers(data.papers));
    h += '<div class="info-section"><div class="info-label">Papers</div>';
    h += filterBarHTML(data.papers);
    h += '<div id="paper-list">'+renderPapers(data.papers, globalSort)+'</div></div>';
    el.innerHTML = h;
    bindFilterBar();
}

function selectNode(bai) {
    network.selectNodes([bai]);
    showNodeInfo(bai);
    network.focus(bai, {scale:network.getScale(), animation:{duration:300}});
}

function renderPapers(papers, sortBy) {
    var sorted = papers.slice().sort(function(a,b) {
        if (sortBy==="recent") return (b.date||"0")>(a.date||"0")?1:-1;
        if (sortBy==="cited") return b.citation_count - a.citation_count;
        return b.weighted_citation - a.weighted_citation;
    });
    if (!sorted.length) return '<p style="color:#999;font-size:13px">No papers</p>';
    var h = "";
    for (var i=0;i<sorted.length;i++) {
        var p = sorted[i];
        h += '<div class="paper-entry"><div class="paper-meta">';
        if (p.primary_category) h += '<span class="paper-cat">'+esc(p.primary_category)+'</span>';
        if (p.arxiv_id) h += '<a class="paper-arxiv" href="https://arxiv.org/abs/'+encodeURIComponent(p.arxiv_id)+'" target="_blank">'+esc(p.arxiv_id)+'</a>';
        if (p.date) h += '<span class="paper-date">'+esc(p.date)+'</span>';
        h += '</div><div class="paper-title">'+esc(p.title)+'</div>';
        h += '<div class="paper-authors">';
        var parts=[], netSet={};
        for (var k=0;k<(p.network_bais||[]).length;k++) netSet[p.network_bais[k]]=true;
        for (var j=0;j<p.authors.length;j++) {
            var au=p.authors[j];
            var isNet = au.bai && (GRAPH_DATA.network_bais.indexOf(au.bai)>=0 || netSet[au.bai]);
            var dn = flipName(au.name);
            if (isNet) parts.push('<a class="net-author" onclick="selectNode(\''+escA(au.bai)+'\')">'+esc(dn)+'</a>');
            else parts.push(esc(dn));
        }
        h += parts.join(", ")+'</div>';
        h += '<div class="paper-stats">Cites: '+p.citation_count+' &middot; Weighted: '+p.weighted_citation.toFixed(2);
        if (p.journal) h += ' &middot; '+esc(p.journal);
        h += '</div></div>';
    }
    return h;
}

// ── Init ───────────────────────────────────────────
(function init() {
    if (typeof network === "undefined") { setTimeout(init, 100); return; }
    network.setOptions({interaction:{zoomView:false, hover:true, tooltipDelay:200}});
    // Populate network from pre-fetched GRAPH_DATA
    var initBais = GRAPH_DATA.network_bais.slice();
    for (var i=0;i<initBais.length;i++) {
        addAuthorToVis(initBais[i]);
    }
    updateGraphVisuals();
    renderAuthorTags();
    function fitNetwork(force) {
        if (!force && userHasZoomed) return;
        network.fit({animation:{duration:400,easingFunction:"easeInOutQuad"},maxZoomLevel:2.0,
                     minZoomLevel:0.3, nodes:network.body.data.nodes.getIds(),
                     padding:40});
    }
    // Fit after stabilization completes, with repeated calls to catch late layout shifts
    network.once("stabilizationIterationsDone", function() {
        fitNetwork();
        setTimeout(function(){ fitNetwork(); }, 500);
        setTimeout(function(){ fitNetwork(); }, 1500);
    });
    network.once("stabilized", function() {
        fitNetwork();
    });
    window.addEventListener("resize", function() { fitNetwork(true); });
    var container = document.getElementById("mynetwork");
    if (container) {
        container.addEventListener("wheel", function(e) {
            e.preventDefault(); e.stopPropagation();
            userHasZoomed = true;
            var scale = network.getScale();
            var delta = e.deltaY > 0 ? -0.015 : 0.015;
            network.moveTo({scale:scale*(1+delta), position:network.getViewPosition()});
        }, {passive:false, capture:true});
    }
    network.on("click", function(params) {
        if (params.nodes.length > 0) showNodeInfo(params.nodes[0]);
        else if (params.edges.length > 0) {
            var nodes = network.getConnectedNodes(params.edges[0]);
            if (nodes.length===2) showEdgeInfo(nodes[0], nodes[1]);
        }
    });

    // ── Node hover highlighting ─────────────────────
    network.on("hoverNode", function(params) {
        var hoveredId = params.node;
        var connNodes = network.getConnectedNodes(hoveredId);
        var connSet = {};
        connSet[hoveredId] = true;
        for (var i=0;i<connNodes.length;i++) connSet[connNodes[i]] = true;
        var allNodes = network.body.data.nodes.get();
        var nodeUpdates = [];
        for (var i=0;i<allNodes.length;i++) {
            var n = allNodes[i];
            if (!connSet[n.id]) {
                nodeUpdates.push({id:n.id,
                    color:{background:"#9ab2d4",border:"#8aa0c0",
                           highlight:{background:"#9ab2d4",border:"#8aa0c0"}},
                    font:{color:"rgba(255,255,255,0.6)",size:(n.font?n.font.size:14),face:"arial",multi:true,strokeWidth:1,strokeColor:"#7a94b8"}});
            }
        }
        network.body.data.nodes.update(nodeUpdates);
        var allEdges = network.body.data.edges.get();
        var edgeUpdates = [];
        for (var i=0;i<allEdges.length;i++) {
            var e = allEdges[i];
            if (!connSet[e.from] || !connSet[e.to]) {
                edgeUpdates.push({id:e.id, color:{color:"#ddd",opacity:0.5}});
            }
        }
        network.body.data.edges.update(edgeUpdates);
    });
    network.on("blurNode", function() {
        // Restore all nodes to standard blue style
        var allNodes = network.body.data.nodes.get();
        var nodeUpdates = [];
        for (var i=0;i<allNodes.length;i++) {
            var n = allNodes[i];
            nodeUpdates.push({id:n.id,
                color:{background:"#4C72B0",border:"#3a5a8c",
                       highlight:{background:"#5c8fd6",border:"#3a5a8c"}},
                font:{color:"#ffffff",size:(n.font?n.font.size:14),face:"arial",
                      multi:true,strokeWidth:2,strokeColor:"#2a4a7a"}});
        }
        network.body.data.nodes.update(nodeUpdates);
        updateGraphVisuals();
    });

    // ── Edge hover tooltip ──────────────────────────
    var edgeTip = document.getElementById("edge-tooltip");
    network.on("hoverEdge", function(params) {
        var nodes = network.getConnectedNodes(params.edge);
        if (nodes.length !== 2) return;
        var key = nodes[0]+"|"+nodes[1];
        var data = GRAPH_DATA.edges[key];
        if (!data) { key = nodes[1]+"|"+nodes[0]; data = GRAPH_DATA.edges[key]; }
        if (!data) return;
        var nA = GRAPH_DATA.authors[data.author_a]&&GRAPH_DATA.authors[data.author_a].full_name
            ? flipName(GRAPH_DATA.authors[data.author_a].full_name) : data.author_a;
        var nB = GRAPH_DATA.authors[data.author_b]&&GRAPH_DATA.authors[data.author_b].full_name
            ? flipName(GRAPH_DATA.authors[data.author_b].full_name) : data.author_b;
        edgeTip.innerHTML = '<b>'+esc(nA)+'</b> &harr; <b>'+esc(nB)+'</b><br>'
            +data.num_papers+' shared paper'+(data.num_papers!==1?'s':'');
        edgeTip.style.display = "block";
    });
    network.on("blurEdge", function() {
        edgeTip.style.display = "none";
    });
    if (container) {
        container.addEventListener("mousemove", function(e) {
            if (edgeTip.style.display !== "none") {
                edgeTip.style.left = (e.clientX + 12) + "px";
                edgeTip.style.top = (e.clientY + 12) + "px";
            }
        });
    }

    renderAuthorTags();
    document.getElementById("lambda-value").addEventListener("change", function() {
        var v = parseFloat(this.value);
        if (isNaN(v)||v<0) v=0;
        this.value = v;
        GRAPH_DATA.decay = v;
        recomputeAllWeights(); updateGraphVisuals(); refreshInfoPanel();
    });
    var authorInput = document.getElementById("add-author-input");
    var authorDropdown = document.getElementById("author-dropdown");
    document.getElementById("add-author-btn").addEventListener("click", function() {
        var v = authorInput.value.trim();
        if (v) { addAuthor(v); authorInput.value = ""; authorDropdown.style.display = "none"; }
    });
    document.getElementById("clear-all-btn").addEventListener("click", clearAllAuthors);
    authorInput.addEventListener("input", function() {
        var q = this.value.trim();
        clearTimeout(searchTimer);
        if (q.length < 2) { authorDropdown.style.display = "none"; return; }
        searchTimer = setTimeout(function(){ searchAuthors(q); }, 300);
    });
    authorInput.addEventListener("keydown", function(e) {
        if (e.key === "ArrowDown") { e.preventDefault(); navigateDropdown(1); }
        else if (e.key === "ArrowUp") { e.preventDefault(); navigateDropdown(-1); }
        else if (e.key === "Enter") {
            e.preventDefault();
            var options = authorDropdown.querySelectorAll(".author-option");
            if (dropdownIndex >= 0 && dropdownIndex < options.length) {
                selectDropdownAuthor(options[dropdownIndex].getAttribute("data-bai"));
            } else if (this.value.trim()) {
                addAuthor(this.value.trim()); this.value = ""; authorDropdown.style.display = "none";
            }
        } else if (e.key === "Escape") { authorDropdown.style.display = "none"; }
    });
    authorInput.addEventListener("blur", function() {
        clearTimeout(searchTimer);
        setTimeout(function(){ authorDropdown.style.display = "none"; }, 200);
    });

    // Formula explanation toggle
    var helpBtn = document.getElementById("formula-help");
    var explDiv = document.getElementById("formula-explanation");
    if (helpBtn && explDiv) {
        helpBtn.addEventListener("click", function() {
            explDiv.style.display = explDiv.style.display === "none" ? "block" : "none";
        });
    }

    // ── Zoom controls ────────────────────────────────
    document.getElementById("zoom-in").addEventListener("click", function() {
        userHasZoomed = true;
        var scale = network.getScale();
        network.moveTo({scale:scale*1.3, animation:{duration:200,easingFunction:"easeInOutQuad"}});
    });
    document.getElementById("zoom-out").addEventListener("click", function() {
        userHasZoomed = true;
        var scale = network.getScale();
        network.moveTo({scale:scale/1.3, animation:{duration:200,easingFunction:"easeInOutQuad"}});
    });
    document.getElementById("zoom-fit").addEventListener("click", function() {
        userHasZoomed = false;
        fitNetwork(true);
    });

    // ── Controls toggle ─────────────────────────────
    var ctrlToggle = document.getElementById("controls-toggle");
    if (ctrlToggle) {
        ctrlToggle.addEventListener("click", function() {
            var ctrl = document.getElementById("controls");
            var collapsed = ctrl.style.display === "none";
            ctrl.style.display = collapsed ? "" : "none";
            this.innerHTML = collapsed ? "&#9660; Controls" : "&#9654; Controls";
        });
    }

    // ── Keyboard navigation ─────────────────────────
    var kbSelectedIdx = -1;
    document.addEventListener("keydown", function(e) {
        if (e.target.tagName==="INPUT"||e.target.tagName==="SELECT"||e.target.tagName==="TEXTAREA") return;
        if (e.key === "Tab") {
            e.preventDefault();
            var ids = network.body.data.nodes.getIds();
            if (!ids.length) return;
            if (e.shiftKey) {
                kbSelectedIdx = kbSelectedIdx <= 0 ? ids.length - 1 : kbSelectedIdx - 1;
            } else {
                kbSelectedIdx = (kbSelectedIdx + 1) % ids.length;
            }
            selectNode(ids[kbSelectedIdx]);
        } else if (e.key === "Escape") {
            network.unselectAll();
            clearInfoPanel();
            kbSelectedIdx = -1;
        }
    });

    // Resizable info panel
    var handle = document.getElementById("resize-handle");
    var infoPanel = document.getElementById("info-panel");
    if (handle && infoPanel) {
        var dragging = false;
        handle.addEventListener("mousedown", function(e) {
            e.preventDefault(); dragging = true;
            handle.classList.add("active");
            document.body.style.cursor = "col-resize";
            document.body.style.userSelect = "none";
        });
        document.addEventListener("mousemove", function(e) {
            if (!dragging) return;
            var newWidth = window.innerWidth - e.clientX;
            if (newWidth < 200) newWidth = 200;
            if (newWidth > window.innerWidth * 0.6) newWidth = window.innerWidth * 0.6;
            infoPanel.style.width = newWidth + "px";
            infoPanel.style.minWidth = newWidth + "px";
        });
        document.addEventListener("mouseup", function() {
            if (dragging) {
                dragging = false;
                handle.classList.remove("active");
                document.body.style.cursor = "";
                document.body.style.userSelect = "";
            }
        });
    }
})();
'''

        custom_js = (
            '<script>\n'
            + js_template.replace('__GRAPH_DATA__', graph_data_json)
            + '\n</script>'
        )

        # ── Save and post-process HTML ─────────────────────────────────
        net.save_graph(save_path)
        with open(save_path, "r") as f:
            html = f.read()

        html = html.replace("<center>", "").replace("</center>", "")
        html = html.replace("</head>", f"{custom_css}\n</head>", 1)
        html = html.replace(
            "<body>",
            f"<body>\n{header_html}\n"
            f'<div id="controls-toggle-bar">'
            f'<button id="controls-toggle">&#9660; Controls</button>'
            f'</div>\n'
            f"{controls_html}\n"
            f'<div id="main-container">\n'
            f'<div id="graph-panel">\n'
            f'<div id="zoom-controls">'
            f'<button class="zoom-btn" id="zoom-in" title="Zoom in">+</button>'
            f'<button class="zoom-btn" id="zoom-out" title="Zoom out">&minus;</button>'
            f'<button class="zoom-btn" id="zoom-fit" title="Fit to view">&#8214;</button>'
            f'</div>\n',
            1,
        )
        html = html.replace(
            "</body>",
            f"{info_panel_html}\n"
            f'<div id="edge-tooltip"></div>\n'
            f"{custom_js}\n</body>",
            1,
        )

        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Saved to {save_path}")
        if show:
            webbrowser.open("file://" + os.path.abspath(save_path))


def _paper_age_years(date_str: str | None, current_year: int | None = None) -> float:
    """Return the age of a paper in years (floored to 0)."""
    if current_year is None:
        current_year = datetime.datetime.now().year
    if not date_str:
        return 30.0  # unknown date -> treat as old
    try:
        year = int(date_str[:4])
    except (ValueError, IndexError):
        return 30.0
    return max(current_year - year, 0.0)


def _weighted_citations_for_papers(
    papers: list[PaperInfo],
    decay: float,
) -> float:
    """Compute weighted_citations = Sum_i( c_i * e^{-decay * age_i} )."""
    total = 0.0
    for p in papers:
        age = _paper_age_years(p.date)
        total += p.citation_count * math.exp(-decay * age)
    return total


def build_collaboration_network(
    author_ids: list[str],
    *,
    decay: float = 1.0,
    max_authors: int = 10,
    client: InspireClient | None = None,
) -> CollabNetwork:
    """Build a weighted collaboration graph for the given list of authors.

    Each edge has a ``weighted_citations`` score computed as::

        W = Sum_i  c_i * e^{-decay * age_i}

    where ``c_i`` is the citation count and ``age_i`` is the number of
    years since the paper was published.

    Parameters
    ----------
    author_ids:
        List of INSPIRE BAIs.
    decay:
        Decay rate lambda. Higher values penalise older papers more.
    max_authors:
        Only include papers with at most this many authors.
    """
    client = client or InspireClient()

    # Fetch all authors concurrently (rate limiter is thread-safe)
    author_papers: dict[str, dict[str, PaperInfo]] = {}

    def _fetch(aid: str) -> tuple[str, dict[str, PaperInfo]]:
        papers = get_author_papers(aid, max_authors=max_authors, client=client)
        return aid, {p.recid: p for p in papers}

    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_fetch, aid): aid for aid in author_ids}
        for i, future in enumerate(as_completed(futures), 1):
            aid, papers_dict = future.result()
            author_papers[aid] = papers_dict
            print(
                f"\r  Fetched {aid} ({len(papers_dict)} papers) "
                f"[{i}/{len(author_ids)}]",
                end="", flush=True,
            )
    print()  # newline after progress

    G = nx.Graph()
    for aid in author_ids:
        G.add_node(aid)

    edges: list[CollabEdge] = []

    for a, b in combinations(author_ids, 2):
        shared_recids = set(author_papers[a]) & set(author_papers[b])
        if not shared_recids:
            continue
        shared = [author_papers[a][r] for r in shared_recids]
        wc = _weighted_citations_for_papers(shared, decay)

        G.add_edge(a, b, weighted_citations=wc, num_papers=len(shared))
        edges.append(CollabEdge(a, b, len(shared), wc))

    return CollabNetwork(
        graph=G, edges=edges, decay=decay, max_authors=max_authors,
        author_papers=author_papers, author_order=list(author_ids),
    )
