# Related paper and citation

This repository contains the research implementation associated with our paper:

> **G3R: a GeoSOT grid-embedded graph reasoning framework for integrating urban
> semantics and spatial analysis**

**Authors:** Rui Shen, Ye Zheng, Ding Ma, Jiaming Huang, Wuxin Zheng, Zhe Zhang,
Weixi Wang, Renzhong Guo, and Xiaoming Li.

**Journal:** *International Journal of Digital Earth* (2026).

**DOI:** [10.1080/17538947.2026.2761129](https://doi.org/10.1080/17538947.2026.2761129).

For a detailed description of the G3R framework, GeoSOT representations,
graph schema, spatial reasoning operators, experimental design, and study
limitations, please refer to the paper and its appendices.

## Relationship between the paper and the code

| Paper component | Implementation |
|---|---|
| CC/VC/LCPC representations and shared level-digit backbone | `recover_encode_from_pg.py`, `recover_build_levdigit_backbone.py`, `recover_import_to_neo4j.py` |
| Urban semantic relations and classification hierarchies | `recover_import_semantics_to_neo4j.py`, `admin_norm.py`, `landuse_norm.py`, `service_categories.py` |
| Distance and direction reasoning | `geosot_spatial_reasoning.py`, `geosot_geometry_reasoning.py`, `geometry_benchmark.py` |
| Forward overlay | `geosot_overlay_query_v7.py` |
| Reverse overlay | `geosot_reverse_query.py` |
| Main-paper KNN retrieval | `geosot_knn_v3_2.py`; V3/V3.1 provide shared components |
| Public-service proximity under land-use mismatch | `run_mismatch_experiment.py` |
| Exploratory urban-village identification | `run_urban_village_experiment.py`, `urban_village_g3r.py` |
| Raw traversal, materialization, and KNN accuracy comparisons | `raw_traversal/`, CSV/Bolt KNN baselines, overfetch and slice analyses |

The paper's KNN method uses planar window termination, local-ellipsoid
reranking of the leading 8K candidates, and Vincenty refinement of Top-K.
V3.3 is retained as a coverage-metric comparison, not as a replacement for the
main-paper entry. Read the appendix distinctions when comparing warm and raw
execution modes.

## Citation

If you use this code or build on the G3R framework in research, please cite:

Shen, R., Zheng, Y., Ma, D., Huang, J., Zheng, W., Zhang, Z., Wang, W., Guo, R.,
and Li, X. (2026). G3R: a GeoSOT grid-embedded graph reasoning framework for
integrating urban semantics and spatial analysis. *International Journal of
Digital Earth*. [https://doi.org/10.1080/17538947.2026.2761129](https://doi.org/10.1080/17538947.2026.2761129).

```bibtex
@article{shen2026g3r,
  author = {Shen, Rui and Zheng, Ye and Ma, Ding and Huang, Jiaming
            and Zheng, Wuxin and Zhang, Zhe and Wang, Weixi
            and Guo, Renzhong and Li, Xiaoming},
  title = {{G3R}: a {GeoSOT} grid-embedded graph reasoning framework
           for integrating urban semantics and spatial analysis},
  journal = {International Journal of Digital Earth},
  year = {2026},
  doi = {10.1080/17538947.2026.2761129},
  url = {https://doi.org/10.1080/17538947.2026.2761129}
}
```

## Reproduction scope

The code documents the implementation and evaluation workflows. Original
resident-level records are not publicly distributed. Complete reproduction of
the citywide applications depends on the controlled inputs and restrictions
stated in the paper. Historical timing results are tied to the reported
hardware, databases, materialization, and warmup protocol; timings on another
system must be measured again.
