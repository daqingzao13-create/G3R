## GeoSOT 编码 CSV 导入 Neo4j 工具

`geosot_neo4j_importer.py` 用于将 **GeoSOT 主码及其属性、顶点信息** 从 CSV 文件批量导入 Neo4j 图数据库，并与已有的 GeoSOT 网格节点建立关联关系。

### 功能概述

- **主码节点导入**：将 CSV 中的 `main_code` 导入为 `(:GeoSOTCode {main_code, geosot_code_id, seq_code, ...})`。
- **主码-网格关系**：基于主码逐级建立 `LOCATED_IN_01GRID ~ LOCATED_IN_32GRID` 关系，连接到已有的 `:grid01Code ~ :grid32Code` 节点。
- **顶点节点与关系**：
  - 解析 `vertex_codes_json` 字段，创建 `(:GeoSOTVertexCode {vertex_code})` 节点；
  - 建立 `(:GeoSOTCode)-[r:HAS_VERTEX]->(:GeoSOTVertexCode)`，关系属性 **`r.vertex_index`** 表示该顶点在该要素中的顺序（0=第 1 个，1=第 2 个，…）；
  - 为顶点建立逐级 `VERTEX_IN_01GRID ~ VERTEX_IN_32GRID` 关系。

> **顶点顺序说明**：同一顶点可能被多个要素共享，在不同要素中顺序不同，因此顺序存在 `HAS_VERTEX` 关系上（`vertex_index`），而非顶点节点上。查询时按 `r.vertex_index` 排序即可得到正确顺序。
- **属性与标签映射**：
  - 支持将选定字段作为 `GeoSOTCode` 节点属性写入；
  - 支持将字段映射为单独的标签节点（如数据源、要素类型等），并建立 `HAS_xxx` 关系。

> **前置条件**：需要先在 Neo4j 中构建好 GeoSOT 网格层级节点（`grid01Code ~ grid32Code`），可参考 `README_GEOSOT_GRID_RELATION_BUILDER.md`。

---

### 环境依赖

- **Python**：推荐 Python 3.8+
- **Python 包**：

```bash
pip install neo4j tqdm
```

（如果 CSV 很大，建议启用虚拟环境，并确保有足够内存与磁盘 IO 性能。）

---

### CSV 数据要求

- **必需列**
  - `main_code`：GeoSOT 主码，形如 `G0...`，必须是东北半球合法编码（内部会校验）。
- **推荐列**
  - `vertex_codes_json`：顶点编码数组的 JSON 字符串，如：

    ```json
    ["G001310321", "G001310322", "G001310323", "G001310324"]
    ```

- **可选属性列**
  - 任意业务字段（如 `feature_name`, `geometry_type`, `source_name` 等），可通过 `--props` 及 `--label-map` 控制如何写入 Neo4j。

> **注意**：脚本会自动尝试多种编码（`utf-8-sig` / `utf-8` / `gb18030` / `gbk`）读取 CSV，并放宽单字段长度限制以兼容长的 `vertex_codes_json`。

---

### 命令行参数

在 `GeoSot` 目录下运行：

```bash
python geosot_neo4j_importer.py --input <输入CSV路径> [其它参数]
```

- **核心参数**
  - `--input`：必选，输入 CSV 文件路径。
  - `--code-column`：主码列名，默认 `main_code`。
  - `--vertex-column`：顶点编码列名，默认 `vertex_codes_json`。
  - `--props`：作为 `GeoSOTCode` 节点属性写入的字段名，逗号分隔。
  - `--label-map`：字段映射为独立标签节点及关系的规则。
- **Neo4j 连接参数**
  - `--uri`：Neo4j URI，默认 `bolt://localhost:7687`。
  - `--user`：用户名，默认 `neo4j`。
  - `--password`：Neo4j 密码 / Neo4j password。无内置默认口令；也可设环境变量 `NEO4J_PASSWORD`（见 `CONFIG.md`）。 / No built-in default secret; set `--password` or `NEO4J_PASSWORD`.
  - `--database`：数据库名，默认 `geocode`。
- **其它控制参数**
  - `--batch`：批量大小，默认 `5000`。
  - `--skip-constraints`：跳过约束/索引创建（适用于约束已提前创建的场景）。
  - `--list-columns`：仅列出 CSV 列名，不执行导入。

---

### 字段到标签节点映射（--label-map）

`--label-map` 的格式为：

- `field`  
- `field:Label`  
- `field:Label:prop`

多个规则用逗号分隔，例如：

```bash
--label-map "geometry_type:GeoFeatureType:type_name,source_name:DataSource:source_name"
```

含义：

- 对每一条记录：
  - 从字段 `geometry_type` 取值，在 Neo4j 中创建 `(:GeoFeatureType {type_name: 值})` 节点；
  - 从字段 `source_name` 取值，在 Neo4j 中创建 `(:DataSource {source_name: 值})` 节点；
  - 当前 `GeoSOTCode` 节点分别通过 `HAS_GEOMETRY_TYPE`、`HAS_SOURCE_NAME` 关系指向对应标签节点。

如果只写 `field`，则：

- Label 自动从字段名推断（首字母大写，非字母数字转 `_`），属性名默认为 `value`。

---

### 常用示例

#### 1. 仅导入主码与少量属性

```bash
python geosot_neo4j_importer.py \
  --input encoded_out.csv \
  --props feature_name,geometry_type
```

- `GeoSOTCode` 节点将包含：
  - `geosot_code_id`（自动生成，`main_code+顺序码`）
  - `main_code`, `seq_code`
  - `feature_name`, `geometry_type` 等属性
- 顶点信息会根据 `vertex_codes_json` 自动解析导入。

#### 2. 同时创建业务标签节点

```bash
python geosot_neo4j_importer.py \
  --input encoded_out.csv \
  --props feature_name,geometry_type,source_name \
  --label-map "geometry_type:GeoFeatureType:type_name,source_name:DataSource:source_name"
```

- 为不同 `geometry_type` 创建 `:GeoFeatureType` 节点。
- 为不同 `source_name` 创建 `:DataSource` 节点。
- `GeoSOTCode` 与这些节点之间建立 `HAS_GEOMETRY_TYPE`、`HAS_SOURCE_NAME` 关系。

#### 2b. 指向固定数据源节点（如 Buildings）

```bash
python geosot_neo4j_importer.py \
  --input encoded_out.csv \
  --props vertex_count,BEREGION,BLDADDR \
  --label-map "geometry_type:GeoFeatureType:type_name" \
  --source Buildings
```

- 创建 `(:Source {name: "Buildings"})` 节点；
- 所有导入的 `GeoSOTCode` 建立 `(GeoSOTCode)-[:FROM_SOURCE]->(:Source {name: "Buildings"})` 关系。

#### 3. 仅查看 CSV 列名

```bash
python geosot_neo4j_importer.py \
  --input encoded_out.csv \
  --list-columns
```

用于快速确认有哪些字段可以放进 `--props` 或 `--label-map`。

#### 4. 在已有约束的数据库中导入

```bash
python geosot_neo4j_importer.py \
  --input encoded_out.csv \
  --skip-constraints
```

适用于生产环境中，约束/索引已由 DBA 提前创建的情况。

---

### 注意事项

- **主码合法性**：仅导入形如 `G0...` 且长度在 2~33 字符（含 `G` 前缀）的东北半球编码，且仅接受 `0/1/2/3` 四进制码元。
- **顺序码上限**：同一个 `main_code` 下，顺序号上限为 `9999`，超过后脚本会抛出异常。
- **网格节点前置**：`LOCATED_IN_xxGRID` / `VERTEX_IN_xxGRID` 关系依赖已有的 `:gridXXCode` 节点，请确保已通过相关构建脚本或导入流程预置好网格节点。
- **连接失败排查**：如果启动时报 `Neo4j 连接失败`，请检查：
  - Neo4j 服务是否启动；
  - `--uri`、`--user`、`--password`、`--database` 是否正确；
  - 防火墙 / 网络连通性。

如需与其他 GeoSOT 工具（编码器、网格关系构建器等）联动使用，可结合 `README_GBT40087_ENCODER.md` 与 `README_GEOSOT_GRID_RELATION_BUILDER.md` 一起查看整体流程。

---

### 顶点顺序查询示例

顶点顺序存储在 `HAS_VERTEX` 关系的 `vertex_index` 上，查询时按该属性排序：

```cypher
MATCH (g:GeoSOTCode {geosot_code_id: 'G0013103210001'})-[r:HAS_VERTEX]->(v:GeoSOTVertexCode)
RETURN v.vertex_code, r.vertex_index
ORDER BY r.vertex_index
```

---

### 完整用法

```bash
python geosot_neo4j_importer.py --input <输入CSV路径> [参数...]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | 是 | - | 输入 CSV 文件路径 |
| `--code-column` | 否 | main_code | 主码列名 |
| `--vertex-column` | 否 | vertex_codes_json | 顶点编码列名 |
| `--props` | 否 | (空) | 作为 GeoSOTCode 属性的字段，逗号分隔 |
| `--label-map` | 否 | (空) | 字段→标签节点映射，如 `field:Label:prop` |
| `--source` | 否 | (空) | 数据源节点：使所有 GeoSOTCode 指向固定的 Source 节点，如 `Buildings` |
| `--uri` | 否 | bolt://localhost:7687 | Neo4j URI |
| `--user` | 否 | neo4j | 用户名 |
| `--password` | 否 / no | *(env / CLI, no default secret)* | Neo4j 密码 / password |
| `--database` | 否 | geocode | 数据库名 |
| `--batch` | 否 | 5000 | 每批写入条数 |
| `--skip-constraints` | 否 | false | 跳过约束创建 |
| `--list-columns` | 否 | false | 仅列出 CSV 列名后退出 |

**完整命令示例**（含业务属性与标签映射）：

```bash
python geosot_neo4j_importer.py \
  --input output/gbt40087_encoded_20260306_204449.csv \
  --props "vertex_count,BEREGION,BLDADDR,BLDCOND,BLDG_HEIGH,BLDG_NO,BLDG_NUM,BLDG_USAGE,BLDSTRU,NOWNAME" \
  --label-map "geometry_type:GeoFeatureType:type_name" \
  --uri bolt://localhost:7687 \
  --user neo4j \
  --password YOUR_PASSWORD \
  --database geocode \
  --batch 2000
```

