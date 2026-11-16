# Program file manifest

This directory contains 44 selected core/reproduction programs and 4 optional tools.
Original module names and the raw_traversal subdirectory are retained.
Chinese comments/docstrings have been translated into English; functional data strings remain unchanged.

For the associated methodology and citation, see [PAPER.md](PAPER.md).
For the scope of local checks, see [VALIDATION.md](VALIDATION.md).

## Encoding and graph construction

| File | Role |
|---|---|
| [gbt40087_encoder.py](gbt40087_encoder.py) | GB/T 40087-2021 encoder for the northeastern hemisphere, with an optional China-range check. Points are encoded directly; lines and polygons use a centroid main code and an ordered vertex-code sequence. |
| [gbt40087_decode_main.py](gbt40087_decode_main.py) | GB/T 40087-2021 decoder for the northeastern hemisphere. Read encoded CSV/TXT input and export decoded grid polygons as GeoJSON. |
| [recover_encode_from_pg.py](recover_encode_from_pg.py) | Spatial graph preparation, step 1/3: encode PostGIS points, lines, and polygons as centroid codes (CC), ordered vertex codes (VC), and longest common prefix codes (LCPC). |
| [recover_build_levdigit_backbone.py](recover_build_levdigit_backbone.py) | Spatial graph preparation, step 2/3: construct the shared level-digit backbone used by the paper. |
| [recover_import_to_neo4j.py](recover_import_to_neo4j.py) | Spatial graph preparation, step 3/3: import encoded CSVs into Neo4j using the paper's schema. Run recover_encode_from_pg.py and recover_build_levdigit_backbone.py first. |
| [recover_import_semantics_to_neo4j.py](recover_import_semantics_to_neo4j.py) | Import the semantic entities and relations required by the paper's two urban applications. Spatial import alone supplies POI/Road/Parcel/Building, CC/VC/LCPC, and the level-digit backbone; applications also need Resident/House/CountyOrig... |
| [admin_norm.py](admin_norm.py) | Normalize county_origin administrative names for hierarchical origin resolution in the urban-village experiment. |

## Spatial operators and shared dependencies

| File | Role |
|---|---|
| [geosot_spatial_reasoning.py](geosot_spatial_reasoning.py) | GeoSOT point-to-point distance and direction reasoning through Neo4j grid adjacency. |
| [geosot_geometry_reasoning.py](geosot_geometry_reasoning.py) | Distance reasoning for all nine Point/LineString/Polygon pairs. |
| [geosot_overlay_query.py](geosot_overlay_query.py) | Overlay helpers using preloading, NumPy bbox filtering, and ray-casting PIP. |
| [geosot_overlay_query_v7.py](geosot_overlay_query_v7.py) | Forward overlay V7 using multiresolution polygon filling in discrete grid space. |
| [geosot_reverse_query.py](geosot_reverse_query.py) | Reverse overlay: find containing parcels for a feature main_code. Convert its code to level-32 integer Z-order coordinates, retrieve candidate parcels by GeoSOTCommonPrefixCode prefixes or LCPC_HAS_LEVEL_k_DIGIT chains, and verify contai... |
| [geosot_knn_v3.py](geosot_knn_v3.py) | Legacy KNN V3 helpers: adaptive grid pruning, prefix queries, and optional geometry reranking. Find the finest safe level meeting the margin (default 100 m), query its prefix, widen to a 3x3 neighborhood or coarsen when needed, and rank ... |
| [geosot_knn_v3_1.py](geosot_knn_v3_1.py) | KNN V3.1 search skeleton: planar L2 coverage, adaptive neighborhoods, and candidate-growth brakes. V3.2 inherits this search logic for the paper's default KNN. |
| [geosot_knn_v3_2.py](geosot_knn_v3_2.py) | KNN V3.2: planar window termination with local-ellipsoid ranking and final Vincenty refinement. Reuse V3.1's expansion/brakes. The default reranker selects the leading 8K planar candidates, ranks them using query-latitude meridian/prime-... |
| [bench_setup.py](bench_setup.py) | Shared PostGIS setup protocol for geometry, forward/reverse overlay, and KNN benchmarks. G3R preloads vertex/neighbor/code indexes into client memory; PG prewarms tables plus GiST/B-tree indexes into shared_buffers and runs disposable qu... |

## Urban applications

| File | Role |
|---|---|
| [run_mismatch_experiment.py](run_mismatch_experiment.py) | Public-service proximity under land-use mismatch: paired G3R and PostGIS pipelines. |
| [run_urban_village_experiment.py](run_urban_village_experiment.py) | Exploratory urban-village identification: paired G3R and PostGIS pipelines sharing the same normalization and equal-weight score. |
| [urban_village_g3r.py](urban_village_g3r.py) | Exploratory urban-village candidate identification through unsupervised multi-signal scoring. This demonstrates semantic/spatial integration rather than an authoritative village map. No official labels are assumed; known village names su... |
| [landuse_norm.py](landuse_norm.py) | Normalize LUFunctionGroup names for the mismatch experiment. classify(raw) returns (land_class,land_subtype), retaining original Chinese category values as functional data. Classes are residential, nonresidential, and mixed; nonresidenti... |
| [service_categories.py](service_categories.py) | Define/load six facility categories: education (schools); medical (hospitals/clinics); sports/recreation; scenic sites; public facilities; and transport excluding parking. |

## Benchmarks and appendix comparisons

| File | Role |
|---|---|
| [geometry_benchmark.py](geometry_benchmark.py) | Batch distance benchmark for all nine Point/LineString/Polygon type combinations. Record per-stage timings in nine detailed CSVs and a combined summary. |
| [knn_r14_overfetch_ablation.py](knn_r14_overfetch_ablation.py) | Exhaustive ellipsoid Top-K reference and PostGIS overfetch sensitivity. Sample unique query_codes from the existing paper CSV using --seed, reuse its G3R and 8x PG lists, and do not rerun Neo4j/G3R. Deduplicate POIs by main_code, then ra... |
| [knn_r15_probe_slices.py](knn_r15_probe_slices.py) | Report grid-boundary, sparse-candidate, and near-tie KNN slices from a compatible paper CSV without database access. Boundary margin is measured at level 16; sparse queries use the bottom 5% of gr_total_candidates; near ties have differe... |
| [geosot_knn_v3_3.py](geosot_knn_v3_3.py) | KNN V3.3 coverage-metric comparison, not the main paper default. Retain V3.2's starting level, expansion, and brakes, but compute r_K and r_safe with the same local-ellipsoid meters-per-degree factors at the query. Refine selected Top-K ... |
| [knn_csv_memory_baseline.py](knn_csv_memory_baseline.py) | No-graph V3.1 memory baseline and CSV helper module. Build POIPrefixCache.from_codes from exported centroid codes using sorted-prefix bisect and centroid arrays; query with session=None so an unavailable cache path cannot fall back to Ne... |
| [knn_csv_scan_baseline.py](knn_csv_scan_baseline.py) | Legacy V3.1 CSV linear-scan baseline without Neo4j or prefix materialization. Scan an unsorted code list with startswith, without bisect or precomputed centroids. Call knn_search independently for each K because coverage depends on K. Re... |
| [knn_csv_scan_v32.py](knn_csv_scan_v32.py) | CSV linear-scan KNN with the paper's V3.2 reranker: planar stop, leading 8K local-ellipsoid candidates, and Top-K Vincenty. Retrieve candidates by startswith over an unsorted list, without a prefix index or Neo4j. |
| [knn_raw_bolt_baseline.py](knn_raw_bolt_baseline.py) | Unmaterialized Bolt KNN using the V3.1 kernel. Do not construct POIPrefixCache or centroid arrays; query each neighborhood prefix with Neo4j STARTS WITH. Disable vertex prefetch and geometry reranking to match the corresponding V3.1/CSV ... |
| [export_neo4j_poi_codes.py](export_neo4j_poi_codes.py) | Read-only export of Point centroid codes for no-graph KNN comparisons. Match the same full GeoSOTCentroidCode/HAS_GEOMETRY_TYPE population as POIPrefixCache, not only the POI source in recover_out/poi_new_encoded.csv. --compare-csv audit... |

## Raw execution mode

| File | Role |
|---|---|
| [raw_traversal/__init__.py](raw_traversal/__init__.py) | GeoSOT raw Neo4j traversal benchmarks. |
| [raw_traversal/_path_setup.py](raw_traversal/_path_setup.py) | Add the parent ``./`` directory to ``sys.path`` so the raw-traversal drivers can import the operator primitives from the existing benchmark scripts (``geosot_overlay_query_v7``, ``geosot_reverse_query``, etc.) without packaging them as a... |
| [raw_traversal/raw_geometry_benchmark.py](raw_traversal/raw_geometry_benchmark.py) | Raw Neo4j traversal — distance & direction reasoning. |
| [raw_traversal/raw_overlay_forward.py](raw_traversal/raw_overlay_forward.py) | Raw Neo4j traversal — forward overlay (statutory plan parcel -> POI). |
| [raw_traversal/raw_overlay_reverse.py](raw_traversal/raw_overlay_reverse.py) | Raw Neo4j traversal — reverse overlay (POI -> containing parcel). |
| [raw_traversal/raw_knn.py](raw_traversal/raw_knn.py) | Raw Neo4j traversal — KNN retrieval. |
| [raw_traversal/raw_appendix_table.py](raw_traversal/raw_appendix_table.py) | Aggregate warm-cache vs raw-traversal CSVs into a single appendix table. |

## Statistics and plotting

| File | Role |
|---|---|
| [geometry_benchmark_analysis.py](geometry_benchmark_analysis.py) | GeoSOT Geometry Benchmark Result Analyzer |
| [overlay_v7_analysis.py](overlay_v7_analysis.py) | Forward Overlay V7 Paper Figure Generator |
| [reverse_query_analysis.py](reverse_query_analysis.py) | GeoSOT Reverse Overlay Query Result Analyzer |
| [knn_v32_join_plot_csv.py](knn_v32_join_plot_csv.py) | Join V3.2 results to the V3.1-compatible plotting schema for plot_knn_v3_sci.py and slice analyses. |
| [plot_knn_v3_sci.py](plot_knn_v3_sci.py) | G3R KNN — SCI-quality publication figures. |
| [knn_v32_paper_stats.py](knn_v32_paper_stats.py) | Summarize Recall, Top-1 agreement, MRR, and median latency from knn_v32_local8k_n10k.csv; join pgs_top1_dist_m from knn_v31_10k.csv for MAE. |
| [paper_figure_style.py](paper_figure_style.py) | Consistent standalone panels for the IJDE second revision. |

## Optional tools

| File | Role |
|---|---|
| [make_uv_figures.py](make_uv_figures.py) | Optional maps of exploratory urban-village candidate regions over satellite imagery. |
| [recover_verify.py](recover_verify.py) | Count nodes/relations under the paper's schema. Historical reference scale: approximately 1,660,949 CC, 7,768,659 VC, 237,805 LCPC, and 368 million relations. These are study-data reference counts, not expected counts for arbitrary or sy... |
| [dataset_experiment_statistics.py](dataset_experiment_statistics.py) | Collect PostGIS dataset counts, environment information, and optional legacy Neo4j graph counts. Spatial tables include buildings, POIs, statutory-plan parcels, and roads; semantic counts use oracle_src resident/house tables. The optiona... |
| [verify_pg_prewarm_cost.py](verify_pg_prewarm_cost.py) | Verify the setup-cost / amortization-crossover claims for the distance experiment against real data. |

## Version distinctions

- Main-paper KNN: `geosot_knn_v3_2.py`.
- `geosot_knn_v3.py` and `geosot_knn_v3_1.py` are retained as shared dependencies and planar references.
- V3.3 evaluates a different coverage metric; it is not the default paper entry.
- CSV/Bolt/raw arms retain their documented metric/version differences.
- Optional tools include mapping, graph counts, environment statistics, and setup-cost inspection.

No research datasets, map tiles, database dumps, or original credentials are included.

The machine-readable publication inventory is [release_file_manifest.csv](release_file_manifest.csv).
