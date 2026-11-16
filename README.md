# G3R

Research code accompanying **G3R: a GeoSOT grid-embedded graph reasoning
framework for integrating urban semantics and spatial analysis**, published in
*International Journal of Digital Earth* (2026).

For the methodology, graph schema, algorithms, experiments, and limitations,
please refer to the [paper and citation information](PAPER.md):
[https://doi.org/10.1080/17538947.2026.2761129](https://doi.org/10.1080/17538947.2026.2761129).

G3R combines urban semantic relations with GeoSOT centroid codes (CC), ordered
vertex codes (VC), and longest common prefix codes (LCPC) in a property graph.
It implements distance/direction reasoning, forward overlay, reverse overlay,
and K-nearest-neighbor retrieval, with PostGIS comparisons and two urban
application workflows.

## Programs

| Component | Main files |
|---|---|
| Encode and decode | `gbt40087_encoder.py`, `gbt40087_decode_main.py`, `recover_encode_from_pg.py` |
| Build the paper graph | `recover_build_levdigit_backbone.py`, `recover_import_to_neo4j.py`, `recover_import_semantics_to_neo4j.py` |
| Distance and direction | `geosot_spatial_reasoning.py`, `geosot_geometry_reasoning.py`, `geometry_benchmark.py` |
| Forward overlay | `geosot_overlay_query_v7.py` and `geosot_overlay_query.py` |
| Reverse overlay | `geosot_reverse_query.py` |
| Main-paper KNN | **`geosot_knn_v3_2.py`**, with V3/V3.1 shared search helpers |
| Public-service proximity | `run_mismatch_experiment.py` |
| Exploratory urban-village identification | `run_urban_village_experiment.py`, `urban_village_g3r.py` |
| Execution-mode and accuracy comparisons | `raw_traversal/`, KNN CSV/Bolt/overfetch/slice programs |
| Statistics and plotting | `geometry_benchmark_analysis.py`, `overlay_v7_analysis.py`, `reverse_query_analysis.py`, `plot_knn_v3_sci.py` |

The complete list of **48 selected Python files** is in
[FILE_MANIFEST.md](FILE_MANIFEST.md). Original module names and the
`raw_traversal/` layout are retained so existing imports and script commands
continue to resolve. Run commands from this directory.

V3.2 is the final main-paper KNN entry. V3 and V3.1 remain necessary dependencies
and planar references; V3.3 is a coverage-metric comparison. The raw Bolt KNN
arm uses V3.1, as distinguished from the V3.2 warm arm in the paper.

## Environment and configuration

The research environment reported in the paper used Python 3.13.5, Neo4j
2025.10.1, PostgreSQL 16.10, and PostGIS 3.5.3. Local verification of this copy
is reported separately in [VALIDATION.md](VALIDATION.md); it does not replace
the paper's experimental environment or reproduce its timing results.

```bash
python -m venv .venv
# Activate the virtual environment for your shell, then:
python -m pip install -r requirements.txt
# Optional satellite-map/HTML plotting:
python -m pip install -r requirements-maps.txt
```

No database passwords are included. Set credentials in the process environment.
[.env.example](.env.example) documents the variables; the scripts do **not**
automatically load a `.env` file.

PowerShell example:

```powershell
$env:NEO4J_URI = "bolt://127.0.0.1:7687"
$env:NEO4J_USER = "neo4j"
$env:NEO4J_DB = "geocode"
$env:NEO4J_PASSWORD = "<your-neo4j-password>"
$env:PGPASSWORD = "<your-postgres-password>"
python geosot_knn_v3_2.py --help
```

Some research scripts retain example database/table names such as
`postgis_35_sample`, `oracle_mig`, `poi_new`, and `statutory_plan`. Adapt their
configuration sections to the prepared schema. Environment overrides beyond
passwords are supported only where the individual script reads them.

## Running the research workflows

The following commands require the input tables, geometry CRS, graph schema,
and data access described in the paper. They are research-workflow examples,
not a bundled-data demonstration. Importers write to the configured graph.

```bash
# Spatial encoding and graph construction, in order:
python recover_encode_from_pg.py --out-dir recover_out
python recover_build_levdigit_backbone.py
python recover_import_to_neo4j.py --in-dir recover_out
# Semantic import also supplies aliases used by the urban application scripts:
python recover_import_semantics_to_neo4j.py

# Small operator runs on prepared databases:
python geometry_benchmark.py benchmark_demo -n 20 --seed 42
python geosot_overlay_query_v7.py --batch 20 --source POI --seed 42
python geosot_reverse_query.py --batch 20 --seed 42
python geosot_knn_v3_2.py --queries 20 --k 3,5,10,20

# Urban application samples:
python run_mismatch_experiment.py --limit 100
python run_urban_village_experiment.py --limit 100
```

The encoder implements the northeastern-hemisphere convention used by this
study. Distance units, boundary semantics, KNN stopping/reranking differences,
cache materialization, and setup-versus-query timing must be interpreted as
specified in the paper and source documentation.

## Data and reproducibility

This directory contains program files, configuration templates, and
documentation. It does not include resident/household records, building-level
population/origin tables, database exports, real code lists, or map-tile caches.
Chinese place/category strings used as database values remain unchanged;
Python comments and docstrings are in English.

Full urban-application reproduction requires controlled data that are not
distributed here. Shareable study data and request conditions are described in
the paper's data availability statement. A standalone synthetic-data demo is
not included in this copying/translation step.

The code has been copied from the selected final research sources, with English
documentation and credential cleanup. It is a local release candidate; the
license decision and full database workflow validation remain separate release
preparation tasks. No remote repository has been changed by this step.
