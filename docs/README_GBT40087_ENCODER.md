# GB/T 40087-2021 编码工具（东北半球 / 中国范围）

脚本路径：`GeoSot/gbt40087_encoder.py`

## 功能

- 按 GB/T 40087-2021 的 `G + 四进制` 规则生成编码（东北半球前缀 `G0`）
- 支持点、线、面（含 Multi 几何）
- 点：主码=点坐标编码
- 线/面：主码=质心编码，附带折点编码
- 编码结果 CSV 会保留输入 SHP/GeoJSON 的原始属性字段

## 输入输出

- 输入：任意 `geopandas` 可读空间文件（如 `.shp`、`.geojson`、`.gpkg`）
- 输出：CSV（包含主码与折点编码 JSON）

## 命令示例

```bash
python d:\project\geosot\GeoSot\gbt40087_encoder.py ^
  --input d:\project\geosot\your_data.geojson ^
  --output d:\project\geosot\GeoSot\encoded_out.csv ^
  --level 32 ^
  --strict-china ^
  --name-field NAME
```

## 参数说明

- `--input`：输入空间文件路径（必填）
- `--output`：输出 CSV 路径（必填）
- `--level`：编码级别，`0~32`，默认 `32`
- `--strict-china`：启用中国范围校验（经度 `73~136`，纬度 `3~54`）
- `--name-field`：要素名字段（可选）
- `--quiet`：关闭进度条显示

## 备注

- 当前实现按你的要求仅覆盖东北半球、中国范围应用场景。
- 中国范围场景不涉及极区（88°N~90°N）合并编码分支。

## 解码工具

脚本路径：`GeoSot/gbt40087_decode_main.py`

### 功能

- 从 CSV/TXT 读取 `G...` 编码，解码成网格 GeoJSON 多边形
- 默认读取 CSV 的 `main_code` 列（可通过 `--code-column` 指定）
- 支持 `--max-features` 抽样解码
- 若输入为编码器生成的 CSV，会将原始属性一并写入解码后 GeoJSON 的 `properties`
- 支持 `--decode-mode geometry|grid`：
  - `geometry`：按 `vertex_codes_json` 重建点/线/面（面自动闭合）
  - `grid`：按 `main_code` 输出网格多边形（旧模式）

### 命令示例

```bash
python d:\project\geosot\GeoSot\gbt40087_decode_main.py ^
  d:\project\geosot\GeoSot\encoded_out.csv ^
  d:\project\geosot\GeoSot\decoded_grid.geojson ^
  --code-column main_code ^
  --decode-mode geometry ^
  --max-features 200
```

可选静默模式（关闭进度条与大部分日志）：

```bash
python d:\project\geosot\GeoSot\gbt40087_decode_main.py ^
  d:\project\geosot\GeoSot\encoded_out.csv ^
  d:\project\geosot\GeoSot\decoded_grid.geojson ^
  --quiet
```
