"""CLI entry points for inspire-tools."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict


def papers_main(argv: list[str] | None = None) -> None:
    """``inspire-papers`` — list papers for an author."""
    parser = argparse.ArgumentParser(
        description="Fetch papers by an INSPIRE author identifier (BAI)."
    )
    parser.add_argument("author_id", help="INSPIRE BAI, e.g. 'M.B.Green.1'")
    parser.add_argument(
        "--max-authors", type=int, default=10,
        help="Only include papers with fewer than this many authors (default: 10)",
    )
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    from inspire_tools.analysis import get_author_papers

    papers = get_author_papers(args.author_id, max_authors=args.max_authors)

    if args.as_json:
        print(json.dumps([asdict(p) for p in papers], indent=2))
        return

    print(f"Found {len(papers)} papers for {args.author_id} "
          f"(max {args.max_authors} authors)\n")
    for i, p in enumerate(papers, 1):
        arxiv = p.arxiv_id or "n/a"
        cats = ", ".join(p.arxiv_categories) if p.arxiv_categories else "n/a"
        print(f"{i:>4}. [{p.date or '????'}] {p.title}")
        print(f"       arXiv: {arxiv}  |  categories: {cats}")
        print(f"       citations: {p.citation_count}  |  authors: {p.author_count}")
        if p.doi:
            print(f"       doi: {p.doi}")
        print()


def categories_main(argv: list[str] | None = None) -> None:
    """``inspire-categories`` — count papers by arXiv category."""
    parser = argparse.ArgumentParser(
        description="Count an author's papers by arXiv category."
    )
    parser.add_argument("author_id", help="INSPIRE BAI, e.g. 'M.B.Green.1'")
    parser.add_argument(
        "--max-authors", type=int, default=10,
        help="Only include papers with fewer than this many authors (default: 10)",
    )
    args = parser.parse_args(argv)

    from inspire_tools.analysis import count_arxiv_categories

    counts = count_arxiv_categories(args.author_id, max_authors=args.max_authors)

    print(f"arXiv category breakdown for {args.author_id}:\n")
    total = sum(c.count for c in counts)
    for c in counts:
        bar = "█" * c.count
        print(f"  {c.category:<20s} {c.count:>4d}  ({100 * c.count / total:5.1f}%)  {bar}")
    print(f"\n  {'TOTAL':<20s} {total:>4d}")


def collab_main(argv: list[str] | None = None) -> None:
    """``inspire-collab`` — build and display a collaboration network."""
    parser = argparse.ArgumentParser(
        description="Build a collaboration network from a list of INSPIRE author IDs."
    )
    parser.add_argument(
        "author_ids", nargs="+",
        help="Two or more INSPIRE BAIs, e.g. 'M.B.Green.1 J.H.Schwarz.1'",
    )
    parser.add_argument(
        "--max-authors", type=int, default=10,
        help="Only include papers with at most this many authors (default: 10)",
    )
    parser.add_argument(
        "--decay", type=float, default=0.2,
        help="Lambda in W = Sum c_i * exp(-lambda * age_i) (default: 0.2)",
    )
    parser.add_argument("--plot", action="store_true", help="Open interactive HTML plot")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Output as JSON")
    args = parser.parse_args(argv)

    if len(args.author_ids) < 2:
        parser.error("Need at least 2 author IDs to build a network.")

    from inspire_tools.analysis import build_collaboration_network

    net = build_collaboration_network(
        args.author_ids,
        decay=args.decay,
        max_authors=args.max_authors,
    )

    if args.as_json:
        from dataclasses import asdict
        print(json.dumps([asdict(e) for e in net.edges], indent=2))
        return

    print(net.summary())

    if args.plot:
        net.plot()
