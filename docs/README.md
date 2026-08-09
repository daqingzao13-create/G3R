# GeoSOT 知识图谱空间推理系统

基于 **GB/T 40087-2021 (GeoSOT)** 编码体系和 **Neo4j 知识图谱**，实现空间距离推理、最近邻查询 (KNN)、居民距离计算等功能，并与 **PostGIS** 基线进行性能与精度对比。

---

## 目录

- [系统架构](#系统架构)
- [环境依赖](#环境依赖)
- [数据库配置](#数据库配置)
- [知识图谱数据模型](#知识图谱数据模型)
- [模块总览](#模块总览)
  - [编码与解码](#1-编码与解码)
  - [Neo4j 数据导入](#2-neo4j-数据导入)
  - [空间推理](#3-空间推理)
  - [KNN 最近邻查询](#4-knn-最近邻查询)
  - [居民应用层](#5-居民应用层)
  - [批量评估与可视化](#6-批量评估与可视化)
- [快速开始](#快速开始)
- [典型实验结果](#典型实验结果)
- [文件清单](#文件清单)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     应用层 (Application)                     │
│  resident_distance.py  resident_hometown_knn.py  ...        │
├─────────────────────────────────────────────────────────────┤
│                     推理层 (Reasoning)                       │
│  geosot_spatial_reasoning.py   geosot_polygon_reasoning.py  │
│  (点到点 BFS 推理)              (面到面多边形推理)              │
├────────────────────────┬────────────────────────────────────┤
│    Neo4j 知识图谱       │         PostGIS 空间数据库          │
│  GeoSOTCode 节点        │    postgis_35_sample (building_new)│
│  Grid 网格节点          │    oracle_mig (resident/house/     │
│  方向关系 EAST/NORTH_OF │         building)                  │
├────────────────────────┴────────────────────────────────────┤
│                 编码层 (Encoding / Decoding)                  │
│         gbt40087_encoder.py    gbt40087_decode_main.py       │
└─────────────────────────────────────────────────────────────┘
```

**三种计算路径对比：**

| 路径 | 描述 | 数据流 |
|------|------|--------|
| **Graph** | 纯知识图谱推理 | Neo4j 图遍历 → GeoSOT 编码解码 → Vincenty 距离 |
| **Hybrid** | 图谱发现 + 空间计算 | Neo4j 获取建筑 ID → PostGIS `ST_Distance` |
| **PostGIS** | 纯关系型空间查询 | SQL JOIN → PostGIS `ST_Distance` |

---

## 环境依赖

### Python 包

```
neo4j
psycopg2          # 或 psycopg2-binary
pandas
geopandas
shapely
tqdm
matplotlib
numpy
```

安装：

```bash
pip install neo4j psycopg2-binary pandas geopandas shapely tqdm matplotlib numpy
```

### 外部服务

| 服务 | 版本要求 | 用途 |
|------|---------|------|
| **Neo4j** | 4.x+ | 知识图谱存储，GeoSOT 节点、网格关系、方向关系 |
| **PostgreSQL + PostGIS** | PG 12+ / PostGIS 3.x | 空间基线查询，`ST_Distance(geography)` |

---

## 数据库配置 / Database configuration

> **No real passwords are shipped.** Set secrets via environment variables.
> 本仓库**不附带真实口令**。详见仓库根目录 [`CONFIG.md`](../CONFIG.md) 与 [`.env.example`](../.env.example)。

### Neo4j

| 环境变量 / Variable | 默认值 / Default | 说明 / Notes |
|---------|--------|------|
| `NEO4J_URI` | `bolt://127.0.0.1:7687` | Bolt URI |
| `NEO4J_USER` | `neo4j` | Username |
| `NEO4J_PASSWORD` | *(empty — required)* | Password / 须自行设置 |
| `NEO4J_DB` | `geocode` | Database name |

### PostgreSQL

`PG_CONFIG` in `geosot_spatial_reasoning.py` is shared by operator modules:

| 参数 / Key | 默认值 / Default | 说明 / Notes |
|------|---|------|
| host | `127.0.0.1` | |
| port | `5432` | |
| dbname | `postgis_35_sample` | Example geometry DB name |
| user | `postgres` | |
| password | `os.getenv("PGPASSWORD", "")` | **No hardcoded secret** |

Urban applications also use a second DB name `oracle_mig` on the same instance for resident/house tables (override as needed).

---

## 知识图谱数据模型

```
(:Resident {id})
  -[:LIVE_IN]-> (:House {id})
    -[:LOCATED_IN]-> (:Building {id})
      -[:HAS_GEOSOT]-> (:GeoSOTCode {main_code})
        -[:HAS_VERTEX {vertex_index}]-> (:GeoSOTVertexCode {code})
        -[:LOCATED_IN_01GRID]-> (:grid01Code)
        -[:LOCATED_IN_02GRID]-> (:grid02Code)
        ...
        -[:LOCATED_IN_32GRID]-> (:grid32Code)

(:Resident) -[:HAS_COUNTY_ORIGIN]-> (:CountyOrigin {name})

(:gridXXCode) -[:EAST_OF]-> (:gridXXCode)
(:gridXXCode) -[:NORTH_OF]-> (:gridXXCode)
```

**关键概念：**
- **main_code**: GeoSOT 32 级编码（如 `G00113023020113303211022033330112`），标识建筑质心
- **vertex_code**: 顶点编码，描述建筑多边形的每个折点
- **Grid 网格**: L1\~L32 层级网格节点，通过 `LOCATED_IN_xxGRID` 建立编码与网格的归属关系
- **方向关系**: `EAST_OF` / `NORTH_OF` 用于 BFS 推理两个网格之间的相对位置偏移

---

## 模块总览

### 1. 编码与解码

| 文件 | 功能 |
|------|------|
| `gbt40087_encoder.py` | GB/T 40087-2021 编码器。支持 Shapefile/GeoJSON 输入，将点/线/面要素编码为 GeoSOT 编码（主码 + 折点编码），输出 CSV |
| `gbt40087_decode_main.py` | GB/T 40087-2021 解码器。读取编码 CSV/TXT，解码为 GeoJSON 网格多边形用于可视化 |

```bash
# 编码 Shapefile
python gbt40087_encoder.py --input buildings.shp --level 32

# 解码为 GeoJSON
python gbt40087_decode_main.py encoded.csv --output grids.geojson
```

### 2. Neo4j 数据导入

| 文件 | 功能 |
|------|------|
| `geosot_neo4j_importer.py` | 将编码 CSV 导入 Neo4j：创建 GeoSOTCode 节点、层级 Grid 节点、HAS_VERTEX 顶点关系 |
| `geosot_grid_relation_builder.py` | 构建网格方向关系：从编码 CSV 提取各层网格前缀，自动创建 EAST_OF / NORTH_OF 关系 |

```bash
# 导入编码数据
python geosot_neo4j_importer.py --csv encoded.csv

# 构建方向关系
python geosot_grid_relation_builder.py --csv encoded.csv
```

### 3. 空间推理

| 文件 | 功能 |
|------|------|
| `geosot_spatial_reasoning.py` | **点到点空间推理**。通过 BFS 遍历 EAST_OF / NORTH_OF 关系，逐级累加网格偏移量，估算两个 GeoSOT 编码点之间的距离和方位角。可与 PostGIS 实际坐标对比验证精度 |
| `geosot_polygon_reasoning.py` | **面到面空间推理**。获取两个面要素的顶点编码，通过链式偏移估算顶点坐标，重建多边形后计算边到边最短距离 (Vincenty)。支持网格解码、图推理、PostGIS 三种方法对比 |

```bash
# 点到点推理
python geosot_spatial_reasoning.py <code1> <code2>

# 面到面推理
python geosot_polygon_reasoning.py <code_a> <code_b>

# 列出面要素
python geosot_polygon_reasoning.py --list --source Buildings
```

**推理算法核心流程（点到点）：**
1. 从 Neo4j 查询两个编码的 L1\~L32 网格链（`LOCATED_IN_xxGRID`）
2. 逐级比较找到首个分异层级 Ld
3. 在 Ld 及更深层级，通过 BFS 遍历 EAST_OF / NORTH_OF 关系计算网格位置差
4. 将各级偏移量转换为经纬度偏移（基于 DMS 编码的网格尺寸）
5. 使用 Vincenty 公式（CGCS2000 椭球）计算大地测量距离

**推理算法核心流程（面到面）：**
1. 从 Neo4j 获取两个面要素的顶点编码（HAS_VERTEX，按 vertex_index 排序）
2. 链式偏移：以首个顶点为基准，利用 BFS 估算其余顶点的坐标
3. 重建两个多边形，逐边计算 segment-to-segment 最短距离

### 4. KNN 最近邻查询

| 文件 | 版本 | 算法 |
|------|------|------|
| `geosot_knn_benchmark.py` | V1 | 多层级 `LOCATED_IN` 图遍历：从查询点出发，逐级扩大搜索范围直到找到 K 个候选 |
| `geosot_knn_v2.py` | **V2** | 圆形多分辨率网格覆盖 + `STARTS WITH` TEXT 索引查询 |
| `geosot_knn_diagnose.py` | 诊断 | KNN 召回率诊断：检查未命中节点的层级关系完整性 |

```bash
# V2 KNN 评估（推荐）
python geosot_knn_v2.py --queries 100 --k 5

# V1 KNN 评估
python geosot_knn_benchmark.py --queries 100

# 诊断未命中
python geosot_knn_diagnose.py <query_code> <missed_code>
```

### 5. 居民应用层

基于知识图谱中的居民-房屋-建筑-GeoSOT 关联关系，实现两大应用场景。

#### 5.1 居民间距离计算

| 文件 | 功能 |
|------|------|
| `resident_distance.py` | 计算两个居民的居住距离，三种方法 (Graph / Hybrid / PostGIS) 对比，显示逐步耗时 |
| `resident_distance_benchmark.py` | 批量计算 N 对居民的距离，输出 CSV，支持断点续跑 |

```bash
# 单次查询
python resident_distance.py <resident_id_1> <resident_id_2>

# 批量评估 10000 对
python resident_distance_benchmark.py --pairs 10000 --seed 42
```

**Graph 方法数据流：**
```
Resident → LIVE_IN → House → LOCATED_IN → Building → HAS_GEOSOT → GeoSOTCode
GeoSOTCode → HAS_VERTEX → GeoSOTVertexCode (获取多边形顶点)
→ 链式偏移估算顶点坐标 → polygon_min_distance_m (边到边最短距离)
```

#### 5.2 老乡最近邻查询 (Hometown KNN)

| 文件 | 功能 |
|------|------|
| `resident_hometown_knn.py` | 查找距指定居民最近的 K 个同乡，包含核心 Neo4j 查询函数和距离计算逻辑 |
| `resident_hometown_profile.py` | 单次查询四种路径的逐步耗时剖析（Graph / Graph v2 / Hybrid / PostGIS） |
| `resident_hometown_benchmark.py` | 批量老乡 KNN 评估，流式获取居民，三路径对比 Recall/误差/耗时，增量写入 CSV |

```bash
# 单次查询
python resident_hometown_knn.py <resident_id> --k 5

# 耗时剖析
python resident_hometown_profile.py <resident_id> --coarse-k 200

# 批量评估（流式，即开即跑）
python resident_hometown_benchmark.py --queries 10000 --k 5
```

**两阶段策略 (Graph)：**
1. **粗筛**：用 GeoSOT main_code 解码质心坐标，Vincenty 距离筛选 coarse-K 候选
2. **精排**：对候选建筑获取顶点编码，面到面多边形距离排序取 Top-K

### 6. 批量评估与可视化

| 文件 | 功能 |
|------|------|
| `geosot_benchmark.py` | 点到点距离三种方法批量精度评估（10 万对），统计误差分布和耗时 |
| `geosot_polygon_benchmark.py` | 面到面距离三种方法批量精度评估（1000 对） |
| `geosot_polygon_profile.py` | 面到面推理逐步耗时剖析，定位性能瓶颈 |
| `benchmark_plot.py` | 读取 benchmark CSV，生成 SCI 论文级别的对比图（误差分布、CDF、时间 box plot 等） |

```bash
# 点到点精度评估
python geosot_benchmark.py --pairs 100000

# 面到面精度评估
python geosot_polygon_benchmark.py --pairs 1000

# 生成论文图表
python benchmark_plot.py
```

---

## 快速开始

### 1. 编码空间数据

```bash
python gbt40087_encoder.py --input your_buildings.shp --level 32
```

### 2. 导入 Neo4j

```bash
python geosot_neo4j_importer.py --csv output/gbt40087_encoded_xxx.csv
python geosot_grid_relation_builder.py --csv output/gbt40087_encoded_xxx.csv
```

### 3. 验证空间推理

```bash
# 验证图结构
python geosot_spatial_reasoning.py --verify

# 点到点推理
python geosot_spatial_reasoning.py <code1> <code2>

# 面到面推理
python geosot_polygon_reasoning.py <code_a> <code_b>
```

### 4. 运行居民应用

```bash
# 居民距离
python resident_distance.py <id_1> <id_2>

# 老乡 KNN
python resident_hometown_knn.py <resident_id> --k 5

# 批量评估
python resident_distance_benchmark.py --pairs 10000
python resident_hometown_benchmark.py --queries 10000
```

### 5. 生成论文图表

```bash
python benchmark_plot.py
```

---

## 典型实验结果

### 居民距离（N=8,666 对）

| 方法 | 平均耗时 (ms) | 中位数 (ms) | P95 (ms) | 相对加速比 |
|------|-------------|-----------|---------|-----------|
| **Graph** | 41.6 | 23.2 | 81.5 | **13.3x** |
| Hybrid | 292.2 | 275.1 | 365.1 | 1.9x |
| PostGIS | 551.3 | 531.3 | 688.6 | 1.0x (基线) |

**Graph 精度：** 平均误差 0.007 m，最大误差 8.2 m，平均误差率 0.00006%

### 分步耗时瓶颈分析

| 步骤 | 平均耗时 (ms) |
|------|-------------|
| Graph: Neo4j 查询 | 36.5 |
| Graph: 顶点估算 | 1.5 |
| Graph: 多边形距离 | 3.5 |
| PostGIS: 几何查找 | 260.7 |
| PostGIS: ST_Distance | 264.6 |

---

## 文件清单

```
GeoSot/
├── gbt40087_encoder.py              # GB/T 40087-2021 编码器
├── gbt40087_decode_main.py          # GB/T 40087-2021 解码器
├── geosot_neo4j_importer.py         # Neo4j 数据导入
├── geosot_grid_relation_builder.py  # 网格方向关系构建
├── geosot_spatial_reasoning.py      # 点到点空间推理 (BFS)
├── geosot_polygon_reasoning.py      # 面到面多边形推理
├── geosot_knn_benchmark.py          # KNN V1 评估
├── geosot_knn_v2.py                 # KNN V2 评估
├── geosot_knn_diagnose.py           # KNN 召回率诊断
├── resident_hometown_knn.py         # 老乡 KNN 核心逻辑
├── resident_hometown_profile.py     # 老乡 KNN 耗时剖析
├── resident_hometown_benchmark.py   # 老乡 KNN 批量评估
├── resident_distance.py             # 居民距离三方法对比
├── resident_distance_benchmark.py   # 居民距离批量评估
├── geosot_benchmark.py              # 点到点精度评估
├── geosot_polygon_benchmark.py      # 面到面精度评估
├── geosot_polygon_profile.py        # 面到面耗时剖析
├── benchmark_plot.py                # 论文图表生成
├── output/                          # 编码/解码输出目录
├── README.md                        # 本文件
├── README_GBT40087_ENCODER.md       # 编码器详细文档
├── README_GEOSOT_NEO4J_IMPORTER.md  # 导入工具详细文档
├── README_GEOSOT_GRID_RELATION_BUILDER.md  # 关系构建详细文档
└── README_KNN.md                    # KNN 详细文档
```

---

## 核心算法说明

### Vincenty 距离公式

所有距离计算均使用 Vincenty 反算公式，基于 CGCS2000 椭球参数（与 WGS84 一致），保证大地测量级精度。

### GeoSOT 编码解码

- **编码**：经纬度 → DMS (度/分/秒) → 四进制编码，32 级分辨率（约 cm 级）
- **解码**：编码 → DMS → 经纬度中心坐标
- 编码遵循 GB/T 40087-2021 国家标准（东北半球，G0 前缀）

### 链式偏移顶点估算

面要素的顶点坐标估算采用链式偏移法：
1. 解码首个顶点到参考坐标
2. 后续顶点与前一顶点做 BFS 偏移推理
3. 累加偏移量得到每个顶点的估计坐标

该方法利用相邻顶点通常位于相近网格的特性，减少 BFS 搜索层数，提高效率。

### 多边形最短距离

重建两个多边形后，计算所有边对之间的最短 Vincenty 距离，取全局最小值作为面到面距离。
