# GeoSOT 空间计算程序总览

基于 **GB/T 40087-2021 (GeoSOT)** 编码与 **Neo4j 知识图谱**，本项目实现了六大类空间计算功能。所有计算均可与 **PostGIS** 基线对比。

---

## 目录

- [一、基础空间推理](#一基础空间推理)
- [二、叠置查询（正向：地块 → 要素）](#二叠置查询正向地块--要素)
- [三、反向叠置查询（要素 → 地块）](#三反向叠置查询要素--地块)
- [四、KNN 最近邻查询](#四knn-最近邻查询)
- [五、居民应用层](#五居民应用层)
- [六、基准测试与分析可视化](#六基准测试与分析可视化)
- [七、数据构建工具](#七数据构建工具)
- [附录：版本演进](#附录版本演进)
- [附录：文件清单](#附录文件清单)

---

## 一、基础空间推理

### 1. `geosot_spatial_reasoning.py` — 点与点距离 / 方位推理

通过 BFS 遍历 Neo4j 中 EAST_OF / NORTH_OF 方向关系，逐级累加网格偏移量，估算两个 GeoSOT 编码点之间的距离和方位角。

```bash
# 计算两个编码点的距离
python geosot_spatial_reasoning.py <编码1> <编码2>

# 不对比 PostGIS
python geosot_spatial_reasoning.py <编码1> <编码2> --no-postgis

# 验证图结构完整性
python geosot_spatial_reasoning.py --verify
```

**算法流程：**
1. 查询两个编码的 L1~L32 网格链
2. 逐级比较找到首个分异层级 Ld
3. BFS 遍历 EAST_OF / NORTH_OF 计算网格位置差
4. 各级偏移量转换为经纬度偏移
5. Vincenty 公式 (CGCS2000 椭球) 计算大地测量距离

### 2. `geosot_geometry_reasoning.py` — 通用几何要素距离推理

支持 Point / LineString / Polygon 两两组合（共 9 种），计算最短距离。

```bash
# 计算两个要素的距离
python geosot_geometry_reasoning.py <id_a> <id_b>

# 列出所有面要素
python geosot_geometry_reasoning.py --list --type Polygon --source Buildings

# 输出 JSON 格式
python geosot_geometry_reasoning.py <id_a> <id_b> --json
```

### 3. `geosot_polygon_reasoning.py` — 面与面距离推理

获取两个面要素的顶点编码，通过链式偏移估算顶点坐标，重建多边形后计算边到边最短距离。

```bash
# 计算两个面的距离
python geosot_polygon_reasoning.py <code_a> <code_b>

# 列出指定 source 的面要素
python geosot_polygon_reasoning.py --list --source Buildings
```

---

## 二、叠置查询（正向：地块 → 要素）

给定一个面状要素（如 Statutory Plan 地块），查找落在其多边形内的所有目标要素（如 POI）。

### 4. `geosot_overlay_query.py` — V1：预加载 + BBox + PIP

预加载全量目标要素坐标，BBox 过滤 + 射线法 PIP 精确判断。

```bash
python geosot_overlay_query.py --code G0xxx --source POI
python geosot_overlay_query.py --batch 50 --source POI --seed 42
python geosot_overlay_query.py --all --source POI --output result.csv
```

### 5. `geosot_overlay_query_v2.py` — V2：LCP 网格前缀过滤 + PIP

利用折点编码的最长公共前缀 (LCP) 缩小候选范围。

```bash
python geosot_overlay_query_v2.py --code G0xxx --source POI
python geosot_overlay_query_v2.py --all --source POI --output result.csv
```

### 6. `geosot_overlay_query_v3.py` — V3：网格覆盖 + 图推理

BBox(度) → 覆盖层级 → BFS 四叉树枚举 → Neo4j STARTS WITH 查询。

```bash
python geosot_overlay_query_v3.py --code G0xxx --source POI
python geosot_overlay_query_v3.py --all --source POI --strategy located_in
python geosot_overlay_query_v3.py --all --source POI --cover-method C --max-expand 2
```

| 参数 | 说明 |
|------|------|
| `--strategy` | `starts_with` (默认) / `located_in` |
| `--cover-method` | 覆盖网格生成方法 |
| `--max-expand` | 最大网格扩展层数 |

### 7. `geosot_overlay_query_v4.py` — V4：折点 LCP 公共前缀

折点 LCP → 前缀匹配 / 图遍历（相对 V3 省去 BBox(度) 和 BFS 枚举）。

```bash
python geosot_overlay_query_v4.py --code G0xxx --source POI
python geosot_overlay_query_v4.py --all --source POI --strategy common_prefix
```

### 8. `geosot_overlay_query_v5.py` — V5：Z-order 矩阵法 ⭐ 最快

将 GeoSOT 编码视为 Z-order (Morton code) 曲线，通过位运算拆解为 (x, y) 整数网格坐标，直接计算覆盖矩形，完全消除 BBox(度) 和 BFS 枚举。

```bash
# 单条查询
python geosot_overlay_query_v5.py --code G0xxx --source POI

# 全量测试 — Neo4j 查询策略
python geosot_overlay_query_v5.py --all --source POI --strategy matrix_neo4j

# 全量测试 — 纯内存策略 (最快)
python geosot_overlay_query_v5.py --all --source POI --strategy matrix_memory

# 控制覆盖矩形最大网格数
python geosot_overlay_query_v5.py --all --source POI --max-cells 64 --output result.csv
```

| 参数 | 说明 |
|------|------|
| `--strategy` | `matrix_neo4j` (默认) / `matrix_memory` (纯内存, 最快) |
| `--max-cells` | 覆盖矩形最大网格数 (默认 256) |
| `--query-source` | 查询面状要素来源 (默认 `Statutory Plan`) |

**V5 算法步骤：**
1. 获取查询多边形的折点编码和坐标
2. 折点编码 → Z-order 位运算拆解为 (x, y) 整数网格坐标，逐级计算覆盖矩形面积，选择最细层级
3. 候选筛选：
   - `matrix_neo4j`: 覆盖矩形 → GeoSOT 编码 → STARTS WITH 查询 Neo4j
   - `matrix_memory`: 预加载 POI (x,y) 数组 → NumPy 矩形范围筛选（不查 Neo4j）
4. 对候选要素做射线法 PIP 精确判断

### 9. `overlay_query_compare.py` — V3 / V4 / V5 跨版本对比

在同一组多边形上并行运行 V3、V4、V5，对比耗时、候选数、精度。

```bash
python overlay_query_compare.py --code G0xxx --source POI
python overlay_query_compare.py --batch 100 --source POI --output compare.csv
python overlay_query_compare.py --all --source POI
```

---

## 三、反向叠置查询（要素 → 地块）

给定一个要素（如 POI），查找其所属的地块（如 Statutory Plan）。

### 10. `geosot_reverse_query.py` — V1：前缀匹配 + 邻域投票

通过 GeoSOTCommonPrefix 前缀匹配筛选候选地块，射线法 PIP 精确判断。支持递进前缀、邻域投票、LOCATED_IN 网格链三种策略。

```bash
# 单条查询
python geosot_reverse_query.py --code G0xxx

# 批量测试
python geosot_reverse_query.py --batch 50 --seed 42

# 全量 + 邻域投票策略
python geosot_reverse_query.py --all --output result.csv --strategy neighbor

# 仅保留命中记录
python geosot_reverse_query.py --all --hit-only 500 --output result.csv
```

| 参数 | 说明 |
|------|------|
| `--strategy` | `prefix` (默认, 递进前缀) / `neighbor` (邻域投票) / `located_in` (网格链) |
| `--hit-only N` | 仅保留 PIP 命中记录, 补足至 N 条 |

### 11. `geosot_reverse_query_v2.py` — V2：Z-order 矩阵法 ⭐ 最快

预加载所有地块多边形的 Z-order 包围矩形 + 地理包围矩形为 NumPy 数组，用向量化范围筛选替代 Neo4j 前缀查询。

```bash
# 单条查询 (默认 matrix_memory 策略)
python geosot_reverse_query_v2.py --code G0xxx

# 全量测试 — 矩阵策略 (最快)
python geosot_reverse_query_v2.py --all --output result.csv --strategy matrix_memory

# 全量测试 — 对比旧前缀策略
python geosot_reverse_query_v2.py --all --output result.csv --strategy prefix

# 批量 + 仅命中
python geosot_reverse_query_v2.py --batch 50 --hit-only 200
```

| 参数 | 说明 |
|------|------|
| `--strategy` | `matrix_memory` (默认, 最快) / `prefix` (递进前缀) / `neighbor` (邻域投票) |
| `--hit-only N` | 仅保留命中记录 |

**V2 算法步骤：**
1. 预加载所有 Plan 多边形 → Z-order (x, y) bbox + 地理 bbox（一次性）
2. 解码 POI 坐标 + Z-order (x, y)
3. Z-order bbox 向量化筛选候选 Plan（NumPy, ~0.01 ms）
4. 地理 bbox 二次过滤
5. 射线法 PIP 精确判断

---

## 四、KNN 最近邻查询

### 12. `geosot_knn_v2.py` — 点要素 KNN

圆形多分辨率网格覆盖 + Neo4j `STARTS WITH` TEXT 索引查询 + Vincenty Top-K。

```bash
python geosot_knn_v2.py --queries 100
python geosot_knn_v2.py --queries 50 --k 10 --radius 200
python geosot_knn_v2.py --queries 50 --k 5 --seed 7
```

| 参数 | 说明 |
|------|------|
| `--queries N` | 查询次数 |
| `--k K` | Top-K (默认 5) |
| `--radius R` | 搜索半径 (米) |
| `--area-tol` | 面积容差 |

---

## 五、居民应用层

基于知识图谱中 Resident → House → Building → GeoSOTCode 关联关系。

### 13. `resident_distance.py` — 居民间距离

Graph / Hybrid / PostGIS 三种方法计算两个居民的居住距离。

```bash
python resident_distance.py <id_1> <id_2>
python resident_distance.py <id_1> <id_2> --rounds 5 --profile
python resident_distance.py <id_1> <id_2> --no-postgis
```

### 14~16. 老乡最近邻 (Hometown KNN)

查找距指定居民最近的 K 个同乡居民。

| 文件 | 版本 | 核心优化 |
|------|------|---------|
| `resident_hometown_knn.py` | V1 | 基础 Neo4j 图推理 |
| `resident_hometown_knn_v2.py` | V2 | 索引优化 + 并行计算 |
| `resident_hometown_knn_v3.py` | **V3** ⭐ | CHILD_OF 层级空间剪枝 |

```bash
# V3 (推荐)
python resident_hometown_knn_v3.py <resident_id> --k 5 --level 15

# V2
python resident_hometown_knn_v2.py <resident_id> --k 5 --workers 4

# V1
python resident_hometown_knn.py <resident_id> --k 10
```

| 参数 | 说明 | 适用版本 |
|------|------|---------|
| `--k K` | Top-K 数量 | 全部 |
| `--coarse-k N` | 粗筛候选数 | V1, V2 |
| `--level L` | 层级剪枝深度 | V3 |
| `--workers N` | 并行线程数 | V2, V3 |
| `--min-cand N` | 最小候选数 | V3 |
| `--no-postgis` | 不对比 PostGIS | 全部 |

---

## 六、基准测试与分析可视化

### 基准测试程序

| 程序 | 测试内容 | 典型用法 |
|------|---------|---------|
| `geosot_benchmark.py` | 点对点距离 3 种方法 | `--pairs 100000 --output out.csv` |
| `geosot_hierarchy_benchmark.py` | CHILD_OF 层级 6 种方法 | `--batch --limit 50 --csv result.csv` |
| `geosot_knn_benchmark.py` | KNN 图推理 vs PostGIS | `--queries 500 --k 10 --output knn.csv` |
| `geosot_overlay_benchmark.py` | 叠置 PIP vs PostGIS | `--queries 200 --output overlay.csv` |
| `geosot_polygon_benchmark.py` | 面面距离精度评估 | `--pairs 1000 --source Buildings` |
| `geometry_benchmark.py` | 通用几何 9 种组合 | `benchmark_v1 -n 100` |
| `resident_distance_benchmark.py` | 居民距离批量对比 | `--pairs 500 --seed 42 -o results.csv` |
| `resident_hometown_benchmark.py` | 老乡 Top-K 批量验证 | `--queries 100 --k 10` |
| `resident_hometown_benchmark_v2.py` | 老乡 KNN 6 种方法全对比 | `--queries 500 --workers 8` |

### 分析可视化程序

| 程序 | 分析对象 | 典型用法 |
|------|---------|---------|
| `overlay_query_v2_analysis.py` | 叠置查询 V2 CSV | `result.csv --dpi 600 --format pdf` |
| `overlay_query_v3_analysis.py` | 叠置查询 V3 CSV | `result.csv --out-dir figures` |
| `reverse_query_analysis.py` | 反向查询 CSV | `result.csv --format svg` |
| `hometown_knn_analysis.py` | 老乡 KNN CSV | `result.csv --dpi 600` |
| `benchmark_plot.py` | 基准测试 CSV | `--input results.csv --outdir figs` |
| `geosot_polygon_profile.py` | 面面推理分步耗时 | `<code_a> <code_b> --repeat 50` |
| `resident_hometown_profile.py` | 老乡 KNN 路径耗时 | `<resident_id> --repeat 3` |

---

## 七、数据构建工具

| 程序 | 功能 |
|------|------|
| `gbt40087_encoder.py` | GB/T 40087-2021 编码器：经纬度 → GeoSOT 编码 (主码 + 折点编码) |
| `gbt40087_decode_main.py` | GB/T 40087-2021 解码器：编码 → GeoJSON 网格多边形 |
| `geosot_neo4j_importer.py` | CSV 数据导入 Neo4j 图数据库 (GeoSOTCode 节点 + 层级关系) |
| `geosot_common_prefix_builder.py` | 构建 GeoSOTCommonPrefix 节点 + LOCATED_IN 关系 |
| `geosot_grid_relation_builder.py` | 构建网格节点 + EAST_OF / NORTH_OF 方向关系 |

```bash
# 编码
python gbt40087_encoder.py --input buildings.shp --level 32

# 解码
python gbt40087_decode_main.py encoded.csv --output grids.geojson

# 导入 Neo4j
python geosot_neo4j_importer.py --csv encoded.csv

# 构建方向关系
python geosot_grid_relation_builder.py --csv encoded.csv

# 构建公共前缀
python geosot_common_prefix_builder.py
```

---

## 附录：版本演进

### 叠置查询（正向：地块 → 要素）

```
V1 (预加载 BBox + PIP)
 → V2 (LCP 前缀过滤)
  → V3 (网格覆盖 + BFS 四叉树 + 图推理)
   → V4 (折点 LCP 公共前缀)
    → V5 (Z-order 矩阵法) ⭐ 最快
```

| 版本 | 候选筛选方式 | Neo4j 依赖 | 核心特点 |
|------|------------|-----------|---------|
| V1 | BBox 范围过滤 | 获取折点 | 简单直接 |
| V2 | LCP 前缀 STARTS WITH | 前缀查询 | 利用编码前缀 |
| V3 | BBox(度) → 覆盖层级 → BFS 枚举 | STARTS WITH / LOCATED_IN | 网格覆盖 |
| V4 | 折点 LCP → 前缀匹配 | 前缀查询 | 省去 BBox 计算 |
| **V5** | **Z-order 位运算 → 覆盖矩形** | **可选跳过** | **纯整数运算, matrix_memory 纯内存** |

### 反向叠置查询（要素 → 地块）

```
V1 (前缀匹配 + 邻域投票)
 → V2 (Z-order 矩阵法) ⭐ 最快
```

| 版本 | 候选筛选方式 | 特点 |
|------|------------|------|
| V1 | Neo4j CommonPrefix 前缀查询 | 支持三种策略 (prefix / neighbor / located_in) |
| **V2** | **Z-order bbox + 地理 bbox NumPy 向量化** | **跳过 Neo4j, ~0.01 ms/查询** |

### 老乡 KNN

```
V1 (基础图推理)
 → V2 (并行优化)
  → V3 (CHILD_OF 层级剪枝) ⭐ 最快
```

---

## 附录：文件清单

```
GeoSot/
│
├── 编码与解码
│   ├── gbt40087_encoder.py                  # GB/T 40087 编码器
│   └── gbt40087_decode_main.py              # GB/T 40087 解码器
│
├── 数据构建
│   ├── geosot_neo4j_importer.py             # Neo4j 数据导入
│   ├── geosot_common_prefix_builder.py      # 公共前缀节点构建
│   └── geosot_grid_relation_builder.py      # 网格方向关系构建
│
├── 基础空间推理
│   ├── geosot_spatial_reasoning.py          # 点到点 BFS 推理
│   ├── geosot_geometry_reasoning.py         # 通用几何距离 (9 种组合)
│   └── geosot_polygon_reasoning.py          # 面到面多边形推理
│
├── 叠置查询 (正向: 地块 → 要素)
│   ├── geosot_overlay_query.py              # V1: 预加载 BBox + PIP
│   ├── geosot_overlay_query_v2.py           # V2: LCP 前缀过滤
│   ├── geosot_overlay_query_v3.py           # V3: 网格覆盖 + 图推理
│   ├── geosot_overlay_query_v4.py           # V4: 折点 LCP 公共前缀
│   ├── geosot_overlay_query_v5.py           # V5: Z-order 矩阵法 ⭐
│   └── overlay_query_compare.py             # V3/V4/V5 跨版本对比
│
├── 反向叠置查询 (要素 → 地块)
│   ├── geosot_reverse_query.py              # V1: 前缀匹配 + 邻域投票
│   └── geosot_reverse_query_v2.py           # V2: Z-order 矩阵法 ⭐
│
├── KNN 最近邻
│   ├── geosot_knn_v2.py                     # 点要素 KNN (圆形网格覆盖)
│   └── geosot_knn_diagnose.py               # KNN 召回率诊断
│
├── 居民应用层
│   ├── resident_distance.py                 # 居民间距离 (三方法)
│   ├── resident_hometown_knn.py             # 老乡 KNN V1
│   ├── resident_hometown_knn_v2.py          # 老乡 KNN V2 (并行)
│   └── resident_hometown_knn_v3.py          # 老乡 KNN V3 (层级剪枝) ⭐
│
├── 基准测试
│   ├── geosot_benchmark.py                  # 点对点距离精度评估
│   ├── geosot_hierarchy_benchmark.py        # 层级图推理方法对比
│   ├── geosot_knn_benchmark.py              # KNN 图推理 vs PostGIS
│   ├── geosot_overlay_benchmark.py          # 叠置查询精度评估
│   ├── geosot_polygon_benchmark.py          # 面面距离精度评估
│   ├── geometry_benchmark.py                # 通用几何批量基准
│   ├── resident_distance_benchmark.py       # 居民距离批量对比
│   ├── resident_hometown_benchmark.py       # 老乡 KNN 批量验证
│   └── resident_hometown_benchmark_v2.py    # 老乡 KNN 6 方法全对比
│
├── 分析与可视化
│   ├── overlay_query_v2_analysis.py         # 叠置 V2 结果分析
│   ├── overlay_query_v3_analysis.py         # 叠置 V3 结果分析
│   ├── reverse_query_analysis.py            # 反向查询结果分析
│   ├── hometown_knn_analysis.py             # 老乡 KNN 结果分析
│   ├── benchmark_plot.py                    # 基准测试图表生成
│   ├── geosot_polygon_profile.py            # 面面推理耗时剖析
│   └── resident_hometown_profile.py         # 老乡 KNN 路径耗时剖析
│
├── 诊断与辅助
│   ├── check_missing_buildings.py           # 缺失建筑检查
│   ├── knn_perf_diagnostic.py               # KNN 性能诊断
│   ├── overlay_benchmark_analysis.py        # 叠置基准分析
│   ├── overlay_diagnose.py                  # 叠置查询诊断
│   ├── overlay_level_study.py               # 叠置层级研究
│   └── overlay_v2_outlier_analysis.py       # 叠置 V2 离群分析
│
└── 文档
    ├── README.md                            # 项目总览
    ├── SPATIAL_PROGRAMS.md                  # 空间计算程序总览 (本文件)
    ├── README_GBT40087_ENCODER.md           # 编码器详细文档
    ├── README_GEOSOT_NEO4J_IMPORTER.md      # 导入工具详细文档
    ├── README_GEOSOT_GRID_RELATION_BUILDER.md # 关系构建详细文档
    └── README_KNN.md                        # KNN 详细文档
```
