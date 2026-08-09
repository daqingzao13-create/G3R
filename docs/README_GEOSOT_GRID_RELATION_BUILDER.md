# GeoSOT 网格节点与方向关系一键构建

脚本：`GeoSot/geosot_grid_relation_builder.py`

## 目标

按 `grid01Code ~ grid32Code` 形式创建网格节点，并自动建立每层：

- `EAST_OF`
- `NORTH_OF`

关系定义与现有 QASystem 保持一致：

- A 在 B 东边：`(A)-[:EAST_OF]->(B)`
- A 在 B 北边：`(A)-[:NORTH_OF]->(B)`

## 输入

使用 `gbt40087_encoder.py` 生成的编码 CSV：

- 主码列：默认 `main_code`
- 折点列：`vertex_codes_json`（脚本会自动读取并提取层级前缀）

## 一键运行

```bash
python d:\project\geosot\GeoSot\geosot_grid_relation_builder.py ^
  --input d:\project\geosot\GeoSot\encoded_out.csv
```

## 先建“空载”网格结构（推荐先检查）

不依赖业务 CSV，直接按中国范围生成指定层级网格并建立 `EAST_OF/NORTH_OF`：

```bash
python d:\project\geosot\GeoSot\geosot_grid_relation_builder.py ^
  --seed-region-china ^
  --min-level 1 ^
  --max-level 12
```

## 按你要的“每层 0123”模板方式建网格（约 4*层级数）

这会在每个层级 `gridXXCode` 下仅创建 `0/1/2/3` 四个节点，并建立该层内部方向关系：

```bash
python d:\project\geosot\GeoSot\geosot_grid_relation_builder.py ^
  --template-level-grids ^
  --min-level 1 ^
  --max-level 32
```

## 可选参数

- `--code-column main_code`：主码列名
- `--seed-region-china`：按中国范围预生成网格（不依赖输入 CSV）
- `--template-level-grids`：每层仅创建 `0/1/2/3` 四个码元节点
- `--min-level / --max-level`：仅处理指定层级范围
- `--force-large`：允许高层级生成（`max-level>16`）
- `--uri` / `--user` / `--password` / `--database`：Neo4j 连接参数
- `--batch 5000`：批处理大小
- `--skip-constraints`：跳过唯一约束创建
- `--nodes-only`：仅建节点，不建方向关系

## 默认连接 / Default connection

- URI：`bolt://localhost:7687`
- User：`neo4j`
- Password：via `NEO4J_PASSWORD` or `--password`（**无内置默认口令** / **no built-in default secret**）
- Database：`geocode`

详见 / See [`CONFIG.md`](../CONFIG.md)。
