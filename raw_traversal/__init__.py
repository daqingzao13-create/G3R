"""
GeoSOT raw Neo4j traversal benchmarks.

This sub-package implements the same four operators evaluated in the main
manuscript (distance & direction reasoning, forward overlay, reverse
overlay, KNN), but executes them as *runtime graph traversals* against
Neo4j on every query, without the traversal-equivalent warm-cache
materializations described in Section 2.6 of the manuscript.

The CSVs produced here populate the appendix table that contrasts
"runtime graph traversal" against the warm-cache lookups reported in
the main text (Tables 7, 8, 9; Section 4.5). See README.md in this
folder for the protocol and command list.
"""
