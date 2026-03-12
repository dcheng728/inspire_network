"""Sample script demonstrating inspire-tools usage."""

# I = Sum_i c_i * exp(-lambda * age_i)
decay = 0.2       # lambda: slower decay means older papers stay relevant

from inspire_tools import (
    InspireClient,
    get_author_papers,
    count_arxiv_categories,
    build_collaboration_network,
)

client = InspireClient()

# ── 1. Fetch papers for an author ────────────────────────────────────
author = "M.B.Green.1"
papers = get_author_papers(author, max_authors=10, client=client)
print(f"=== Papers by {author} (< 10 authors): {len(papers)} ===\n")
for p in papers[:5]:
    print(f"  {p.date}  {p.title}")
    print(f"            arXiv: {p.arxiv_id}  |  cites: {p.citation_count}")
print("  ...")

# ── 2. arXiv category breakdown ─────────────────────────────────────
print(f"\n=== arXiv categories for {author} ===\n")
categories = count_arxiv_categories(author, client=client)
for c in categories:
    print(f"  {c.category:<20s} {c.count:>4d}")

# ── 3. Collaboration network ────────────────────────────────────────
# authors = ["M.B.Green.1", "John.H.Schwarz.1", "Edward.Witten.1"]
imperial_authors = ["Arkady.A.Tseytlin.1",
                    "Shai.M.Chester.1",
                    "C.R.Contaldi.1",
                    "C.de.Rham.1",
                    "F.Dowker.1",
                    "Michael.J.Duff.1",
                    "J.P.Gauntlett.1",
                    "J.J.Halliwell.1",
                    "A.Hanany.1",
                    "C.M.Hull.1",
                    "S.Komatsu.1",
                    "J.Magueijo.1",
                    "A.Rajantie.1",
                    "A.J.Tolley.1",
                    "D.Waldram.1",
                    "Toby.Wiseman.1",
                    ]

other_authors = ["Edward.Witten.1",
                 "Stephen.W.Hawking.1",]

authors = imperial_authors + other_authors
print(f"\n=== Collaboration network: {authors} ===\n")

net = build_collaboration_network(
    authors,
    decay=decay,
    client=client,
)
print(net.summary())

# Visualize the network
net.plot(save_path="examples/collab_network.html")
