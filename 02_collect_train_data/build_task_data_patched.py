"""
Patched wrapper around build_task_data.py.

Why this exists:
- The original script looks up embeddings with key (code, value, unit).
- Our new embedding pipeline builds unique events from
  extract_unique_events.py, where deduplication is based on:
    (omop_table, code, value, unit)
- So lookup must include omop_table to avoid collisions and to match the new
  event_index.parquet schema.

This wrapper keeps the original script untouched and only monkey-patches the
embedding-index-related functions before delegating to the original main().
"""

import json
import math
import os
import sys

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import build_task_data as orig


def normalise_event_key(omop_table, code, value, unit):
    return (
        str(omop_table if omop_table is not None else "").strip(),
        str(code if code is not None else "").strip(),
        str(value if value is not None else "").strip(),
        str(unit if unit is not None else "").strip(),
    )


def load_embed_idx(embed_dir):
    idx_path = os.path.join(embed_dir, "event_index.parquet")
    df = pd.read_parquet(
        idx_path,
        columns=["event_id", "omop_table", "code", "value", "unit"],
    )
    result = {}
    for row in df.itertuples(index=False):
        key = normalise_event_key(row.omop_table, row.code, row.value, row.unit)
        result[key] = int(row.event_id)
    print("  Loaded {} embedding index entries from {}".format(len(result), idx_path))
    return result


def _build_timeline_from_df(events_df):
    n_hits = 0
    misses = {}
    lines = []

    for _, ev in events_df.iterrows():
        if ev["omop_table"] == "condition_occurrence":
            continue

        code = ev["code"]
        if code in orig._CODE_2_DESC:
            description = orig._CODE_2_DESC[code]
            n_hits += 1
        else:
            description = None
            misses[code] = misses.get(code, 0) + 1

        raw_value = ev["value"]
        raw_unit = ev["unit"]

        has_value = isinstance(raw_value, str) and raw_value.strip() != ""
        if has_value:
            numeric = pd.to_numeric(raw_value, errors="coerce")
            value = None if (isinstance(numeric, float) and math.isnan(numeric)) else float(numeric)
        else:
            value = None

        unit = raw_unit.strip() if isinstance(raw_unit, str) and raw_unit.strip() != "" else None

        embed_val = raw_value if isinstance(raw_value, str) else ""
        embed_unit = raw_unit if isinstance(raw_unit, str) else ""
        norm_key = normalise_event_key(ev["omop_table"], code, embed_val, embed_unit)
        if norm_key not in orig._EMBED_IDX:
            raise KeyError(
                "No embedding found for event key {!r} "
                "(omop_table={!r}, code={!r}, raw_value={!r}, raw_unit={!r}). "
                "Expected event_index.parquet built from unique_event_rows.parquet.".format(
                    norm_key, ev["omop_table"], code, raw_value, raw_unit
                )
            )
        embedding_idx = orig._EMBED_IDX[norm_key]

        event = {
            "time": str(ev["start"])[:16],
            "type": orig.OMOP_TABLE_PREFIX.get(ev["omop_table"], ev["omop_table"]),
            "table": ev["omop_table"],
            "code": code,
            "description": description,
            "value": value,
            "unit": unit,
            "embedding_idx": embedding_idx,
        }
        lines.append(json.dumps(event, ensure_ascii=False))

    return "\n".join(lines), n_hits, misses


orig.normalise_event_key = normalise_event_key
orig.load_embed_idx = load_embed_idx
orig._build_timeline_from_df = _build_timeline_from_df


if __name__ == "__main__":
    orig.main()
