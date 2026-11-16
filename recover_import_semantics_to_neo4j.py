#!/usr/bin/env python3
# coding: utf-8
"""
Import the semantic entities and relations required by the paper's two urban applications.
Spatial import alone supplies POI/Road/Parcel/Building, CC/VC/LCPC, and the level-digit
backbone; applications also need Resident/House/CountyOrigin/Province/AdminCity,
POIType/POIType2/POIType3, and LUFunctionGroup/LUClass/LUSubtype.

Read oracle_mig.oracle_src resident/house/building/county_origin_dict and
postgis_35_sample.public poi_new/statutory_plan/building_new. Align (:Building).BLDG_NO with
oracle_src.building.id and house.buildingid. The feature_id/main_code is the spatial
identifier, not this administrative join key. Use synthetic or appropriately controlled
semantic records.

Selectable, idempotent stages: 0 constraints/indexes; 1 type dictionaries; 2 building
attributes; 3 House-LOCATED_IN->Building; 4 Resident-LIVE_IN->House and HAS_COUNTY_ORIGIN; 5
POI category edges; 6 parcel/CC land-use edges; 7 SUBCLASS_OF, IN_LU_CLASS, HAS_LU_SUBTYPE,
IN_PROVINCE, IN_CITY; 8 compatibility aliases HAS_GEOSOT, :GeoSOTCode, :Source; 9
verification counts.

Usage:
    python recover_import_semantics_to_neo4j.py --stages 0,1
    python recover_import_semantics_to_neo4j.py --stages 0,1,2,3,4 --limit-resident 50000
    python recover_import_semantics_to_neo4j.py
    python recover_import_semantics_to_neo4j.py --stages 9

Dependencies: neo4j, psycopg2, tqdm; landuse_norm/admin_norm support hierarchy construction.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import psycopg2
from neo4j import GraphDatabase
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from landuse_norm import classify, normalize
except ImportError:  # pragma: no cover
    classify = None
    normalize = None

try:
    from admin_norm import parse_admin
except ImportError:  # pragma: no cover
    parse_admin = None


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DB", "geocode")

PG_MAIN = {
    "host": os.getenv("PGHOST", "127.0.0.1"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgis_35_sample"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}
PG_MIG = {**PG_MAIN, "dbname": os.getenv("PG_MIG_DB", "oracle_mig")}

BATCH = 2000
RESIDENT_BATCH = 5000

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _t() -> float:
    return time.perf_counter()


def _chunks(seq: Sequence[Any], n: int) -> Iterator[Sequence[Any]]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _pg(cfg: Dict) -> Any:
    return psycopg2.connect(**cfg)


class SemanticImporter:
    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str,
        batch: int = BATCH,
        limit_resident: Optional[int] = None,
        limit_house: Optional[int] = None,
        only_linked_buildings: bool = True,
    ):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        self.batch = batch
        self.limit_resident = limit_resident
        self.limit_house = limit_house
        self.only_linked_buildings = only_linked_buildings

    def close(self) -> None:
        self.driver.close()

    def _session(self):
        return self.driver.session(database=self.database)

    def run(self, cypher: str, **params) -> Any:
        with self._session() as s:
            return s.run(cypher, **params).consume()

    def rows(self, cypher: str, **params) -> List[Dict]:
        with self._session() as s:
            return [dict(r) for r in s.run(cypher, **params)]

    def scalar(self, cypher: str, **params) -> Any:
        with self._session() as s:
            rec = s.run(cypher, **params).single()
            return None if rec is None else rec[0]

    def unwind(self, cypher: str, key: str, items: Sequence[Any],
               desc: str) -> None:
        if not items:
            print(f"  [{desc}] empty, skip")
            return
        with self._session() as s:
            for chunk in tqdm(
                list(_chunks(items, self.batch)),
                desc=f"  {desc}", unit="batch",
            ):
                s.run(cypher, **{key: list(chunk)})

    # ------------------------------------------------------------------
    # Stage 0
    # ------------------------------------------------------------------
    def stage0_constraints(self) -> None:
        print("\n[0] Constraints / indexes")
        stmts = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Resident) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:House) REQUIRE n.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:CountyOrigin) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Province) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:AdminCity) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:POIType) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:POIType2) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:POIType3) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:LUFunctionGroup) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:LUClass) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:LUSubtype) REQUIRE n.name IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (n:Building) ON (n.BLDG_NO)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Building) ON (n.id)",
            "CREATE INDEX IF NOT EXISTS FOR (n:House) ON (n.buildingid)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Resident) ON (n.housecode)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Resident) ON (n.county_origin)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Resident) ON (n.birthday)",
            "CREATE INDEX IF NOT EXISTS FOR (n:POI) ON (n.f1)",
            "CREATE INDEX IF NOT EXISTS FOR (n:POI) ON (n.f2)",
            "CREATE INDEX IF NOT EXISTS FOR (n:POI) ON (n.f3)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Parcel) ON (n.lu_functio)",
        ]
        for s in stmts:
            try:
                self.run(s)
                print("  ✓", s.split("FOR")[1].strip()[:60] if "FOR" in s else s[:60])
            except Exception as e:
                print("  !", e)

    # ------------------------------------------------------------------
    # Stage 1 — type dictionaries
    # ------------------------------------------------------------------
    def stage1_type_nodes(self) -> None:
        print("\n[1] Type dictionary nodes from PG")
        t0 = _t()
        m = _pg(PG_MAIN)
        g = _pg(PG_MIG)

        with m.cursor() as c:
            c.execute(
                "SELECT DISTINCT f1 FROM public.poi_new "
                "WHERE f1 IS NOT NULL AND btrim(f1) <> ''"
            )
            f1 = [r[0] for r in c.fetchall()]
            c.execute(
                "SELECT DISTINCT f2 FROM public.poi_new "
                "WHERE f2 IS NOT NULL AND btrim(f2) <> ''"
            )
            f2 = [r[0] for r in c.fetchall()]
            c.execute(
                "SELECT DISTINCT f3 FROM public.poi_new "
                "WHERE f3 IS NOT NULL AND btrim(f3) <> ''"
            )
            f3 = [r[0] for r in c.fetchall()]
            c.execute(
                "SELECT DISTINCT lu_functio FROM public.statutory_plan "
                "WHERE lu_functio IS NOT NULL"
            )
            lu_raw = [r[0] for r in c.fetchall()]

        lu_names: set = set()
        for lu in lu_raw:
            text = normalize(lu) if normalize else str(lu).strip()
            for part in text.split("+"):
                part = part.strip()
                if part:
                    lu_names.add(part)

        with g.cursor() as c:
            c.execute(
                "SELECT name, COALESCE(province,''), COALESCE(city,'') "
                "FROM oracle_src.county_origin_dict"
            )
            origins = [
                {"name": n, "province": p or "", "city": ci or ""}
                for n, p, ci in c.fetchall() if n
            ]

        m.close()
        g.close()

        self.unwind(
            "UNWIND $rows AS n MERGE (:POIType {name:n})",
            "rows", f1, "POIType",
        )
        self.unwind(
            "UNWIND $rows AS n MERGE (:POIType2 {name:n})",
            "rows", f2, "POIType2",
        )
        self.unwind(
            "UNWIND $rows AS n MERGE (:POIType3 {name:n})",
            "rows", f3, "POIType3",
        )
        self.unwind(
            "UNWIND $rows AS n MERGE (:LUFunctionGroup {name:n})",
            "rows", sorted(lu_names), "LUFunctionGroup",
        )
        self.unwind(
            "UNWIND $rows AS r "
            "MERGE (c:CountyOrigin {name:r.name}) "
            "SET c.province = r.province, c.city = r.city",
            "rows", origins, "CountyOrigin",
        )
        self.unwind(
            "UNWIND $rows AS r "
            "MATCH (c:CountyOrigin {name:r.name}) "
            "FOREACH (_ IN CASE WHEN r.province <> '' THEN [1] ELSE [] END | "
            "  MERGE (p:Province {name:r.province}) "
            "  MERGE (c)-[:IN_PROVINCE]->(p)) "
            "FOREACH (_ IN CASE WHEN r.city <> '' THEN [1] ELSE [] END | "
            "  MERGE (ct:AdminCity {name:r.city}) "
            "  MERGE (c)-[:IN_CITY]->(ct))",
            "rows", origins, "CountyOrigin admin edges",
        )
        print(
            f"  POIType {len(f1)} / POIType2 {len(f2)} / POIType3 {len(f3)} | "
            f"LUFunctionGroup {len(lu_names)} | CountyOrigin {len(origins)} "
            f"({_t() - t0:.1f}s)"
        )

    # ------------------------------------------------------------------
    # Stage 2 — enrich Building from oracle
    # ------------------------------------------------------------------
    def stage2_enrich_buildings(self) -> None:
        print("\n[2] Enrich Building from oracle_src.building (by BLDG_NO)")
        t0 = _t()
        g = _pg(PG_MIG)
        sql = """
            SELECT id::text,
                   COALESCE(layers::text, ''),
                   COALESCE(buildingarea::text, ''),
                   COALESCE(buildtype, ''),
                   COALESCE(buildingusage, ''),
                   COALESCE(address, '')
              FROM oracle_src.building
             WHERE id IS NOT NULL
        """
        with g.cursor(name="bld_cur") as c:
            c.itersize = 10000
            c.execute(sql)
            buf: List[Dict] = []
            n = 0
            matched = 0
            with self._session() as s:
                for row in tqdm(c, desc="  oracle building", unit="row"):
                    bid, layers, area, btype, usage, addr = row
                    item = {
                        "bno": str(bid),
                        "layers": layers,
                        "buildingarea": area,
                        "buildtype": btype or "",
                        "buildingusage": usage or "",
                        "address": addr or "",
                    }
                    buf.append(item)
                    if len(buf) >= self.batch:
                        matched += self._flush_building_enrich(s, buf)
                        n += len(buf)
                        buf.clear()
                if buf:
                    matched += self._flush_building_enrich(s, buf)
                    n += len(buf)
        g.close()
        print(f"  scanned {n:,} oracle buildings; "
              f"SET props on matched Neo4j nodes "
              f"(approx via counters) {_t() - t0:.1f}s")

    @staticmethod
    def _flush_building_enrich(session, buf: List[Dict]) -> int:
        # layers / buildingarea stored as numbers when parseable
        cy = """
        UNWIND $rows AS r
        MATCH (b:Building {BLDG_NO: r.bno})
        SET b.id = r.bno,
            b.layers = CASE WHEN r.layers <> '' THEN toInteger(r.layers)
                            ELSE b.layers END,
            b.buildingarea = CASE WHEN r.buildingarea <> ''
                                  THEN toFloat(r.buildingarea)
                                  ELSE b.buildingarea END,
            b.buildtype = CASE WHEN r.buildtype <> '' THEN r.buildtype
                               ELSE b.buildtype END,
            b.buildingusage = CASE WHEN r.buildingusage <> ''
                                   THEN r.buildingusage
                                   ELSE b.buildingusage END,
            b.address = CASE WHEN r.address <> '' THEN r.address
                             ELSE b.address END
        RETURN count(b) AS n
        """
        rec = session.run(cy, rows=buf).single()
        return int(rec["n"]) if rec else 0

    # ------------------------------------------------------------------
    # Stage 3 — House
    # ------------------------------------------------------------------
    def _neo4j_building_bnos(self) -> List[str]:
        return [
            r["bno"] for r in self.rows(
                "MATCH (b:Building) WHERE b.BLDG_NO IS NOT NULL "
                "RETURN b.BLDG_NO AS bno"
            ) if r.get("bno")
        ]

    def stage3_houses(self) -> None:
        print("\n[3] House nodes + LOCATED_IN → Building")
        t0 = _t()
        g = _pg(PG_MIG)
        bnos = self._neo4j_building_bnos() if self.only_linked_buildings else None
        if bnos is not None:
            print(f"  Neo4j buildings with BLDG_NO: {len(bnos):,}")

        with g.cursor() as c:
            c.execute("DROP TABLE IF EXISTS tmp_neo4j_bno")
            c.execute("CREATE TEMP TABLE tmp_neo4j_bno (bno text PRIMARY KEY)")
            if bnos is not None:
                from psycopg2.extras import execute_values
                execute_values(
                    c, "INSERT INTO tmp_neo4j_bno(bno) VALUES %s ON CONFLICT DO NOTHING",
                    [(x,) for x in bnos], page_size=10000,
                )
            g.commit()

        if bnos is not None:
            sql = (
                "SELECT h.id::text, h.buildingid::text "
                "  FROM oracle_src.house h "
                "  JOIN tmp_neo4j_bno t ON t.bno = h.buildingid::text "
                " WHERE h.id IS NOT NULL"
            )
        else:
            sql = (
                "SELECT id::text, buildingid::text "
                "  FROM oracle_src.house WHERE id IS NOT NULL"
            )
        if self.limit_house:
            sql += f" LIMIT {int(self.limit_house)}"

        with g.cursor(name="house_cur") as c:
            c.itersize = 10000
            c.execute(sql)
            buf: List[Dict] = []
            n_in = n_link = 0
            with self._session() as s:
                for hid, bid in tqdm(c, desc="  houses", unit="row"):
                    if not hid:
                        continue
                    buf.append({
                        "id": str(hid),
                        "buildingid": str(bid) if bid else "",
                    })
                    if len(buf) >= self.batch:
                        a, b = self._flush_houses(s, buf)
                        n_in += a
                        n_link += b
                        buf.clear()
                if buf:
                    a, b = self._flush_houses(s, buf)
                    n_in += a
                    n_link += b
        g.close()
        print(f"  House upserts ~{n_in:,}, LOCATED_IN ~{n_link:,} "
              f"({_t() - t0:.1f}s)")

    @staticmethod
    def _flush_houses(session, buf: List[Dict]) -> Tuple[int, int]:
        cy = """
        UNWIND $rows AS r
        MERGE (h:House {id: r.id})
        SET h.buildingid = r.buildingid
        WITH h, r
        WHERE r.buildingid <> ''
        MATCH (b:Building {BLDG_NO: r.buildingid})
        MERGE (h)-[:LOCATED_IN]->(b)
        RETURN count(DISTINCT h) AS nh, count(*) AS ne
        """
        rec = session.run(cy, rows=buf).single()
        if not rec:
            return 0, 0
        return int(rec["nh"]), int(rec["ne"])

    # ------------------------------------------------------------------
    # Stage 4 — Resident
    # ------------------------------------------------------------------
    def stage4_residents(self) -> None:
        print("\n[4] Resident nodes + LIVE_IN + HAS_COUNTY_ORIGIN")
        t0 = _t()
        g = _pg(PG_MIG)

        # Restrict to residents living in houses already linked to Neo4j
        # Buildings (via house.buildingid = Building.BLDG_NO).
        if self.only_linked_buildings:
            bnos = self._neo4j_building_bnos()
            print(f"  filter by Neo4j BLDG_NO: {len(bnos):,}")
            with g.cursor() as c:
                c.execute("DROP TABLE IF EXISTS tmp_neo4j_bno")
                c.execute(
                    "CREATE TEMP TABLE tmp_neo4j_bno (bno text PRIMARY KEY)"
                )
                from psycopg2.extras import execute_values
                execute_values(
                    c,
                    "INSERT INTO tmp_neo4j_bno(bno) VALUES %s "
                    "ON CONFLICT DO NOTHING",
                    [(x,) for x in bnos], page_size=10000,
                )
            g.commit()
            sql = (
                "SELECT r.id::text, "
                "       COALESCE(r.housecode::text, ''), "
                "       COALESCE(r.county_origin, ''), "
                "       COALESCE(r.birthday::text, ''), "
                "       COALESCE(r.sex::text, ''), "
                "       COALESCE(r.domiciletype::text, '') "
                "  FROM oracle_src.resident r "
                "  JOIN oracle_src.house h ON h.id = r.housecode "
                "  JOIN tmp_neo4j_bno t ON t.bno = h.buildingid::text "
                " WHERE r.id IS NOT NULL"
            )
        else:
            sql = (
                "SELECT id::text, "
                "       COALESCE(housecode::text, ''), "
                "       COALESCE(county_origin, ''), "
                "       COALESCE(birthday::text, ''), "
                "       COALESCE(sex::text, ''), "
                "       COALESCE(domiciletype::text, '') "
                "  FROM oracle_src.resident "
                " WHERE id IS NOT NULL"
            )
        if self.limit_resident:
            sql += f" LIMIT {int(self.limit_resident)}"

        with g.cursor(name="res_cur") as c:
            c.itersize = 20000
            c.execute(sql)
            buf: List[Dict] = []
            n = 0
            with self._session() as s:
                for row in tqdm(c, desc="  residents", unit="row"):
                    rid, hcode, origin, bday, sex, dtype = row
                    if not rid:
                        continue
                    buf.append({
                        "id": str(rid),
                        "housecode": str(hcode) if hcode else "",
                        "county_origin": origin or "",
                        "birthday": bday or "",
                        "sex": sex or "",
                        "domiciletype": dtype or "",
                    })
                    if len(buf) >= RESIDENT_BATCH:
                        self._flush_residents(s, buf)
                        n += len(buf)
                        buf.clear()
                if buf:
                    self._flush_residents(s, buf)
                    n += len(buf)
        g.close()
        total = self.scalar("MATCH (r:Resident) RETURN count(r)")
        print(f"  processed rows {n:,}; Resident nodes now {total:,} "
              f"({_t() - t0:.1f}s)")

    @staticmethod
    def _flush_residents(session, buf: List[Dict]) -> None:
        cy = """
        UNWIND $rows AS r
        MERGE (res:Resident {id: r.id})
        SET res.housecode = r.housecode,
            res.county_origin = r.county_origin,
            res.birthday = r.birthday,
            res.sex = r.sex,
            res.domiciletype = r.domiciletype
        WITH res, r
        FOREACH (_ IN CASE WHEN r.housecode <> '' THEN [1] ELSE [] END |
          MERGE (h:House {id: r.housecode})
          MERGE (res)-[:LIVE_IN]->(h)
        )
        FOREACH (_ IN CASE WHEN r.county_origin <> '' THEN [1] ELSE [] END |
          MERGE (co:CountyOrigin {name: r.county_origin})
          MERGE (res)-[:HAS_COUNTY_ORIGIN]->(co)
        )
        """
        session.run(cy, rows=buf)

    # ------------------------------------------------------------------
    # Stage 5 — POI type edges
    # ------------------------------------------------------------------
    def stage5_poi_type_edges(self) -> None:
        print("\n[5] POI → HAS_POI_TYPE* edges (from node props f1/f2/f3)")
        t0 = _t()
        # Stream POI props from Neo4j in batches by feature_id
        fids = [r["id"] for r in self.rows(
            "MATCH (p:POI) RETURN p.feature_id AS id ORDER BY id")]
        print(f"  POI nodes: {len(fids):,}")
        with self._session() as s:
            for chunk in tqdm(list(_chunks(fids, self.batch)),
                              desc="  POI type edges", unit="batch"):
                s.run(
                    """
                    UNWIND $ids AS fid
                    MATCH (p:POI {feature_id: fid})
                    FOREACH (_ IN CASE WHEN p.f1 IS NOT NULL AND p.f1 <> ''
                                      THEN [1] ELSE [] END |
                      MERGE (t:POIType {name: p.f1})
                      MERGE (p)-[:HAS_POI_TYPE]->(t)
                    )
                    FOREACH (_ IN CASE WHEN p.f2 IS NOT NULL AND p.f2 <> ''
                                      THEN [1] ELSE [] END |
                      MERGE (t2:POIType2 {name: p.f2})
                      MERGE (p)-[:HAS_POI_TYPE_2]->(t2)
                    )
                    FOREACH (_ IN CASE WHEN p.f3 IS NOT NULL AND p.f3 <> ''
                                      THEN [1] ELSE [] END |
                      MERGE (t3:POIType3 {name: p.f3})
                      MERGE (p)-[:HAS_POI_TYPE_3]->(t3)
                    )
                    """,
                    ids=list(chunk),
                )
        print(f"  done {_t() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Stage 6 — Parcel LU edges
    # ------------------------------------------------------------------
    def stage6_lu_edges(self) -> None:
        print("\n[6] Parcel (+ CC) → HAS_LU_FUNCTION_GROUP")
        t0 = _t()
        parcels = self.rows(
            "MATCH (p:Parcel) WHERE p.lu_functio IS NOT NULL "
            "RETURN p.feature_id AS fid, p.lu_functio AS lu"
        )
        items: List[Dict] = []
        for r in parcels:
            lu = r.get("lu") or ""
            text = normalize(lu) if normalize else str(lu).strip()
            parts = [x.strip() for x in text.split("+") if x.strip()]
            for part in parts:
                items.append({"fid": r["fid"], "name": part})
        print(f"  parcel-LU pairs: {len(items):,}")
        self.unwind(
            """
            UNWIND $rows AS r
            MATCH (p:Parcel {feature_id: r.fid})
            MERGE (g:LUFunctionGroup {name: r.name})
            MERGE (p)-[:HAS_LU_FUNCTION_GROUP]->(g)
            WITH p, g
            MATCH (p)-[:HAS_CENTROID_CODE]->(c:GeoSOTCentroidCode)
            MERGE (c)-[:HAS_LU_FUNCTION_GROUP]->(g)
            """,
            "rows", items, "LU edges",
        )
        print(f"  done {_t() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Stage 7 — hierarchy edges
    # ------------------------------------------------------------------
    def stage7_hierarchy(self) -> None:
        print("\n[7] Hierarchy: SUBCLASS_OF + LU class/subtype")
        t0 = _t()
        # POIType3 → POIType2 → POIType from co-occurrence on POI
        pairs32 = self.rows(
            """
            MATCH (p:POI)-[:HAS_POI_TYPE_3]->(c:POIType3)
            MATCH (p)-[:HAS_POI_TYPE_2]->(par:POIType2)
            RETURN c.name AS child, par.name AS parent, count(*) AS n
            ORDER BY child, n DESC
            """
        )
        pairs21 = self.rows(
            """
            MATCH (p:POI)-[:HAS_POI_TYPE_2]->(c:POIType2)
            MATCH (p)-[:HAS_POI_TYPE]->(par:POIType)
            RETURN c.name AS child, par.name AS parent, count(*) AS n
            ORDER BY child, n DESC
            """
        )

        def _dedupe(pairs: List[Dict]) -> List[Dict]:
            # keep all parents (DAG), unique (child,parent)
            seen = set()
            out = []
            for r in pairs:
                key = (r["child"], r["parent"])
                if key in seen or not r["child"] or not r["parent"]:
                    continue
                seen.add(key)
                out.append({"child": r["child"], "parent": r["parent"]})
            return out

        e32 = _dedupe(pairs32)
        e21 = _dedupe(pairs21)
        self.unwind(
            """
            UNWIND $rows AS r
            MATCH (c:POIType3 {name:r.child})
            MATCH (p:POIType2 {name:r.parent})
            MERGE (c)-[:SUBCLASS_OF]->(p)
            """,
            "rows", e32, "SUBCLASS_OF L3→L2",
        )
        self.unwind(
            """
            UNWIND $rows AS r
            MATCH (c:POIType2 {name:r.child})
            MATCH (p:POIType {name:r.parent})
            MERGE (c)-[:SUBCLASS_OF]->(p)
            """,
            "rows", e21, "SUBCLASS_OF L2→L1",
        )

        # LUFunctionGroup → LUClass / LUSubtype via landuse_norm
        if classify is None:
            print("  ! landuse_norm unavailable, skip LUClass/LUSubtype")
        else:
            names = [r["n"] for r in self.rows(
                "MATCH (g:LUFunctionGroup) RETURN g.name AS n")]
            rows = []
            for nm in names:
                cls, sub = classify(nm)
                rows.append({"nm": nm, "cls": cls, "sub": sub})
            self.unwind(
                """
                UNWIND $rows AS r
                MATCH (g:LUFunctionGroup {name:r.nm})
                MERGE (lc:LUClass {name:r.cls})
                MERGE (ls:LUSubtype {name:r.sub})
                MERGE (g)-[:IN_LU_CLASS]->(lc)
                MERGE (g)-[:HAS_LU_SUBTYPE]->(ls)
                """,
                "rows", rows, "LU class/subtype",
            )
        print(f"  done {_t() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Stage 8 — experiment compatibility
    # ------------------------------------------------------------------
    def _batched_write(
        self, cypher: str, desc: str, page: int = 2000,
    ) -> int:
        """Repeat LIMIT-page writes until a batch returns 0 (avoids OOM)."""
        total = 0
        with self._session() as s:
            with tqdm(desc=desc, unit="row") as bar:
                while True:
                    rec = s.run(cypher, page=page).single()
                    n = int(rec["n"]) if rec and rec.get("n") is not None else 0
                    total += n
                    bar.update(n)
                    if n < page:
                        break
        return total

    def stage8_compat(self) -> None:
        print("\n[8] Experiment compatibility shim "
              "(HAS_GEOSOT / :GeoSOTCode / :Source)")
        t0 = _t()
        # Secondary labels — must be batched; a single SET over ~1.6M CC
        # nodes blows dbms.memory.transaction.total.max.
        n_cc = self._batched_write(
            """
            MATCH (c:GeoSOTCentroidCode)
            WHERE NOT c:GeoSOTCode
            WITH c LIMIT $page
            SET c:GeoSOTCode
            RETURN count(*) AS n
            """,
            "  label GeoSOTCode on CC",
            page=max(500, min(self.batch, 2000)),
        )
        # DataSource count is tiny
        self.run("MATCH (d:DataSource) WHERE NOT d:Source SET d:Source")
        # Alias HAS_GEOSOT (same endpoints as HAS_CENTROID_CODE)
        n_rel = self._batched_write(
            """
            MATCH (f:SpatialFeature)-[:HAS_CENTROID_CODE]->(c:GeoSOTCentroidCode)
            WHERE NOT (f)-[:HAS_GEOSOT]->(c)
            WITH f, c LIMIT $page
            MERGE (f)-[:HAS_GEOSOT]->(c)
            RETURN count(*) AS n
            """,
            "  HAS_GEOSOT",
            page=max(500, min(self.batch, 2000)),
        )
        print(f"  labeled CC→GeoSOTCode ~{n_cc:,}; DataSource→Source; "
              f"HAS_GEOSOT ~{n_rel:,} ({_t() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Stage 9 — verify
    # ------------------------------------------------------------------
    def stage9_verify(self) -> None:
        print("\n[9] Verify counts")
        checks = [
            ("Resident", "MATCH (n:Resident) RETURN count(n)"),
            ("House", "MATCH (n:House) RETURN count(n)"),
            ("CountyOrigin", "MATCH (n:CountyOrigin) RETURN count(n)"),
            ("Province", "MATCH (n:Province) RETURN count(n)"),
            ("AdminCity", "MATCH (n:AdminCity) RETURN count(n)"),
            ("POIType", "MATCH (n:POIType) RETURN count(n)"),
            ("POIType2", "MATCH (n:POIType2) RETURN count(n)"),
            ("POIType3", "MATCH (n:POIType3) RETURN count(n)"),
            ("LUFunctionGroup", "MATCH (n:LUFunctionGroup) RETURN count(n)"),
            ("LUClass", "MATCH (n:LUClass) RETURN count(n)"),
            ("LUSubtype", "MATCH (n:LUSubtype) RETURN count(n)"),
            ("LIVE_IN", "MATCH ()-[r:LIVE_IN]->() RETURN count(r)"),
            ("LOCATED_IN", "MATCH ()-[r:LOCATED_IN]->() RETURN count(r)"),
            ("HAS_COUNTY_ORIGIN",
             "MATCH ()-[r:HAS_COUNTY_ORIGIN]->() RETURN count(r)"),
            ("HAS_POI_TYPE_3",
             "MATCH ()-[r:HAS_POI_TYPE_3]->() RETURN count(r)"),
            ("SUBCLASS_OF", "MATCH ()-[r:SUBCLASS_OF]->() RETURN count(r)"),
            ("HAS_LU_FUNCTION_GROUP",
             "MATCH ()-[r:HAS_LU_FUNCTION_GROUP]->() RETURN count(r)"),
            ("HAS_GEOSOT", "MATCH ()-[r:HAS_GEOSOT]->() RETURN count(r)"),
            ("Building.layers set",
             "MATCH (b:Building) WHERE b.layers IS NOT NULL "
             "RETURN count(b)"),
            # Prove Resident→House→Building exists without scanning all 13M+.
            ("chain sample ok",
             "MATCH (b:Building)<-[:LOCATED_IN]-(:House)"
             "<-[:LIVE_IN]-(:Resident) "
             "WITH b LIMIT 1 RETURN CASE WHEN b IS NULL THEN 0 ELSE 1 END"),
        ]
        for name, cy in checks:
            try:
                n = self.scalar(cy)
                print(f"  {name:<28} {n:>12,}")
            except Exception as e:
                print(f"  {name:<28} ERROR {e}")


STAGE_FUNCS = {
    0: "stage0_constraints",
    1: "stage1_type_nodes",
    2: "stage2_enrich_buildings",
    3: "stage3_houses",
    4: "stage4_residents",
    5: "stage5_poi_type_edges",
    6: "stage6_lu_edges",
    7: "stage7_hierarchy",
    8: "stage8_compat",
    9: "stage9_verify",
}


def parse_stages(text: str) -> List[int]:
    if not text.strip():
        return list(STAGE_FUNCS.keys())
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Import semantic entities/relations into Neo4j (post-recovery)")
    ap.add_argument(
        "--stages", default="",
        help="comma stages 0-9 (default: all). "
             "0=constraints 1=types 2=building 3=house 4=resident "
             "5=poi-edges 6=lu-edges 7=hierarchy 8=compat 9=verify",
    )
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--limit-resident", type=int, default=None,
                    help="cap residents read from oracle (smoke)")
    ap.add_argument("--limit-house", type=int, default=None,
                    help="cap houses read from oracle (smoke)")
    ap.add_argument(
        "--include-unlinked-houses", action="store_true",
        help="also import houses/residents whose building has no Neo4j geometry",
    )
    args = ap.parse_args()

    stages = parse_stages(args.stages)
    print("=" * 72)
    print("Semantic import → Neo4j")
    print(f"  Neo4j  {NEO4J_URI} / {NEO4J_DATABASE}")
    print(f"  PG     {PG_MAIN['dbname']} + {PG_MIG['dbname']}")
    print(f"  stages {stages}")
    print("=" * 72)

    imp = SemanticImporter(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE,
        batch=args.batch,
        limit_resident=args.limit_resident,
        limit_house=args.limit_house,
        only_linked_buildings=not args.include_unlinked_houses,
    )
    try:
        if not imp.scalar("RETURN 1"):
            print("Neo4j connection failed")
            return 1
        for st in stages:
            fn = getattr(imp, STAGE_FUNCS[st])
            fn()
    finally:
        imp.close()
    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
