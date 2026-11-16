# Local verification

Date: 2026-11-16. Scope: copying and English documentation of the selected
research programs, with credential cleanup and two help-only CLI fixes.

## Checks completed

- All **48 Python files** parsed/compiled and imported successfully.
- All **709 Chinese comment tokens** and **402 Chinese docstrings** were
  translated. A final tokenizer/AST scan found no remaining Chinese comments
  or docstrings. An additional **51 help/output text fragments** were translated.
- Original source-file SHA256 values were rechecked: the research originals
  were not modified.
- An AST comparison checked each copy against its source while allowing only
  documentation changes, credential environment reads, translated display
  literals, and the two help-only early returns. Computational kernels and
  functional database category/place-name strings were preserved.
- Five full-depth sample coordinates passed encoding/integer-code round trips.
- Four Vincenty distances agreed with an independent PROJ/pyproj CGCS2000
  calculation within 0.01 m; the largest observed difference was approximately
  0.00000139 m. Coincident-point distance was also checked.
- Integer PIP/bbox tests matched known inside/outside answers on a simple
  square. Prefix-cache queries matched direct string filtering, including
  deduplication and self exclusion.
- Thirteen representative `--help` entry points exited successfully. The two
  urban application scripts originally ignored `--help`; the release copy now
  returns help before creating output directories or connecting to databases.

Network and database calls were guarded during these checks. No actual
database connections or research experiments were executed. Initial help
checks exposed the missing help handling and blocked its connection attempts;
both entry points passed after the help guards were added.

## Verification environment

These are the local verification packages, **not a reconstruction of the
paper's frozen experiment environment**.

| Component | Version |
|---|---|
| Python | 3.13.5 |
| NumPy | 2.4.4 |
| Numba | 0.65.1 |
| Neo4j Python driver | 6.2.0 |
| psycopg2-binary | 2.9.12 |
| pandas | 3.0.2 |
| Matplotlib | 3.10.9 |
| SciPy | 1.17.1 |
| GeoPandas | 1.1.3 |
| Shapely | 2.1.2 |
| Pillow | 12.2.0 |
| tqdm | 4.67.3 |
| psutil | 7.2.2 |
| contextily | 1.7.1 |
| Folium | 0.20.0 |
| pyproj | 3.7.2 |

## Limits

These are source, import, CLI-help, and small computational checks. They do not
verify every operator case, full database imports, citywide data alignment,
application results, map downloads, or the paper's performance measurements.
The optional dataset-statistics script retains legacy graph-count labels;
check them against the deployed graph before using its report.

No datasets, real credentials, downloaded imagery tiles, or database dumps
were copied. Complete citywide reproduction still requires the controlled
inputs and schema described in [PAPER.md](PAPER.md). A standalone synthetic
dataset/demo and a license file are not included in this preparation step.
