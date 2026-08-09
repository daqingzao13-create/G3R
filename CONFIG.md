# Configuration & Credentials / 配置与凭据

**English.** This open-source package does **not** ship real database passwords.
All secrets must be supplied via environment variables or CLI flags.

**中文。** 本开源包**不包含**真实数据库口令。所有机密须通过环境变量或命令行参数提供。

## Required environment variables / 必需环境变量

| Variable | Purpose | Default in code |
|----------|---------|-----------------|
| `NEO4J_URI` | Neo4j Bolt URI | `bolt://127.0.0.1:7687` |
| `NEO4J_USER` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | **empty** (must set) |
| `NEO4J_DB` | Neo4j database name | `geocode` |
| `PGPASSWORD` | PostgreSQL password | **empty** (must set) |

Optional PostgreSQL host settings (where a script reads them): `PGHOST`, `PGPORT`, `PGUSER`, `PGDATABASE`.

See also `.env.example`.

## Example (PowerShell) / 示例

```powershell
$env:NEO4J_PASSWORD = "your-neo4j-password"
$env:PGPASSWORD = "your-postgres-password"
python geosot_overlay_query_v7.py --source POI --help
```

## Example (bash) / 示例

```bash
export NEO4J_PASSWORD='your-neo4j-password'
export PGPASSWORD='your-postgres-password'
python geosot_knn_v3_1.py --queries 100 --k 5
```

## CLI overrides / 命令行覆盖

Importer and grid-relation builder accept `--password` (and `--uri` / `--user` / `--database`).
Prefer env vars in shared environments; never commit real passwords.

导入与网格关系构建脚本支持 `--password`（及 `--uri` / `--user` / `--database`）。
共享环境优先用环境变量；**切勿**把真实口令提交到仓库。

## Where passwords are read / 口令读取位置

| Module | Neo4j | PostgreSQL |
|--------|-------|------------|
| `geosot_spatial_reasoning.py` (`PG_CONFIG`) | `NEO4J_PASSWORD` | `PGPASSWORD` |
| Operator scripts (overlay / reverse / KNN / geometry) | `NEO4J_PASSWORD` | via `PG_CONFIG` |
| `geosot_neo4j_importer.py` / builders | `NEO4J_PASSWORD` or `--password` | — |
| `run_mismatch_experiment.py` / urban village apps | `NEO4J_PASSWORD` | `PGPASSWORD` (`PG_MAIN` / `PG_MIG`) |

## Notes / 说明

- Database names such as `postgis_35_sample` / `oracle_mig` / `geocode` are **local deployment examples**, not secrets. Change them to match your instance.
- 库名如 `postgis_35_sample` / `oracle_mig` / `geocode` 仅为本地部署示例，可按实例修改。
- Chinese comments in `.py` files are bilingual (`中文 / English` or Chinese line + English line).
- `.py` 中的中文注释已补英文（`中文 / English` 或中文行 + 英文行）。
