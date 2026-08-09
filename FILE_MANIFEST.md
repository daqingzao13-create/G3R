# GeoSOT Open Source Package

口径与论文 `main.tex` / `appendix.tex` 一致：**编码→导入 → 四个空间算子 → 两个城市应用**。

Scope matches the paper: **encode → import → four spatial operators → two urban applications**.

## 1. 编码到导入 / Encode & import

| 文件 / File | 作用 / Role |
|------|------|
| gbt40087_encoder.py | GeoSOT 编码 / encoding |
| gbt40087_decode_main.py | 解码 / decoding |
| geosot_neo4j_importer.py | 导入 Neo4j（CC/VC 等） |
| geosot_common_prefix_builder.py | LCPC 公共前缀 |
| geosot_grid_relation_builder.py | EAST_OF / NORTH_OF |

## 2. 四个正官算子 / Four operators（`生成配图命令行.md`）

| 算子 | 主程序 | 依赖 |
|------|--------|------|
| 距离/方位 | geosot_spatial_reasoning.py | geosot_geometry_reasoning.py |
| 正向叠置 | geosot_overlay_query_v7.py | geosot_overlay_query.py |
| 反向叠置 | geosot_reverse_query.py | （上列） |
| KNN | geosot_knn_v3_1.py | geosot_knn_v3.py |

## 3. 两个城市应用 / Two applications

| 论文应用 | 主程序 | 依赖 |
|----------|--------|------|
| 用地错配下的公共服务邻近 | run_mismatch_experiment.py | landuse_norm.py, service_categories.py |
| 城中村候选识别 | run_urban_village_experiment.py | urban_village_g3r.py |

## 4. 文档 / Docs

`docs/`：README 系列、SPATIAL_PROGRAMS.md、生成配图命令行.md、语义空间桥接实验_实验日志.md。

根目录另有：[`CONFIG.md`](CONFIG.md)、[`.env.example`](.env.example)。

## 5. 明确未纳入 / Excluded

- multiscale_kg_vs_postgis.py
- resident_distance / resident_hometown_knn_*
- 出图/分析脚本、历史版本、CSV/cache、论文/专利原稿、诊断探针

## 6. Open-source hygiene / 开源处理说明

- Python 源码中的**中文注释已补英文**（`中文 / English` 或中文行 + 英文行）。
  Chinese comments in `.py` files are bilingual.
- **真实账号口令已移除**；统一改为环境变量：
  - `NEO4J_PASSWORD`（Neo4j）
  - `PGPASSWORD`（PostgreSQL）
  默认值为空字符串，须由使用者自行设置。
  Real passwords removed; empty env defaults only.
- 配置说明见 [`CONFIG.md`](CONFIG.md)、[`.env.example`](.env.example)，
  以及 `README.md` / `docs/README*.md` 中的数据库配置节。
