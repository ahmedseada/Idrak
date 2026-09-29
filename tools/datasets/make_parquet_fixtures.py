"""Writes the Parquet test fixtures of Idrak.Datasets and, next to each, the rows as pyarrow reads them (JSON Lines).

    python tools/datasets/make_parquet_fixtures.py tests/Idrak.Tests/data/parquet

Needs pyarrow. The expected rows use the JSON forms the C# reader produces: dates as yyyy-MM-dd, timestamps as ISO 8601,
decimals as numbers, maps as [{"key", "value"}] lists.
"""
import datetime as dt
import decimal
import json
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def rows(n):
    out = []
    for i in range(n):
        messages = [{"role": "user", "content": f"question {i}"}]
        if i % 3 != 0:
            messages.append({"role": "assistant", "content": None if i % 7 == 0 else f"answer {i} " + "é" * (i % 4)})
        out.append({
            "id": i * 1000003 - 5,
            "text": f"row {i % 17} of the data set",
            "text2": ("prefix-shared-" + str(i)) if i % 5 else None,
            "score": None if i % 4 == 0 else i / 8,
            "half": float(i % 3) * 0.5,
            "flag": i % 2 == 0,
            "messages": messages,
            "tags": None if i % 6 == 0 else [f"t{j}" for j in range(i % 3)],
            "grid": [[i, i + 1], [], [None, 2]] if i % 2 else [],
            "meta": None if i % 5 == 0 else {"source": f"s{i % 2}", "rank": i % 4},
            "attrs": [("a", i), ("b", None)] if i % 2 else [],
            "day": dt.date(2024, 1, 1) + dt.timedelta(days=i),
            "at": dt.datetime(2024, 1, 2, 3, 4, 5, 123456) + dt.timedelta(seconds=i),
            "price": decimal.Decimal(i * 125) / 100,
            "small": i % 120,
        })
    return out


SCHEMA = pa.schema([
    ("id", pa.int64()),
    ("text", pa.string()),
    ("text2", pa.string()),
    ("score", pa.float64()),
    ("half", pa.float32()),
    ("flag", pa.bool_()),
    ("messages", pa.list_(pa.struct([("role", pa.string()), ("content", pa.string())]))),
    ("tags", pa.list_(pa.string())),
    ("grid", pa.list_(pa.list_(pa.int32()))),
    ("meta", pa.struct([("source", pa.string()), ("rank", pa.int32())])),
    ("attrs", pa.map_(pa.string(), pa.int32())),
    ("day", pa.date32()),
    ("at", pa.timestamp("us")),
    ("price", pa.decimal128(10, 2)),
    ("small", pa.int8()),
])


def expected(row):
    def fix(v):
        if isinstance(v, dt.datetime):
            return v.strftime("%Y-%m-%dT%H:%M:%S") + (("." + f"{v.microsecond:06d}".rstrip("0")) if v.microsecond else "")
        if isinstance(v, dt.date):
            return v.isoformat()
        if isinstance(v, decimal.Decimal):
            return float(v) if v != v.to_integral() else int(v)
        return v

    out = {k: fix(v) for k, v in row.items()}
    out["attrs"] = [{"key": k, "value": v} for k, v in row["attrs"]]
    return out


def write(folder, name, n, **options):
    data = rows(n)
    table = pa.Table.from_pylist(data, schema=SCHEMA)
    path = os.path.join(folder, name + ".parquet")
    pq.write_table(table, path, **options)
    back = pq.read_table(path).to_pylist()
    with open(os.path.join(folder, name + ".jsonl"), "w", encoding="utf-8") as f:
        for row in back:
            f.write(json.dumps(expected(row), ensure_ascii=False) + "\n")


def main():
    folder = sys.argv[1]
    os.makedirs(folder, exist_ok=True)
    write(folder, "snappy-v1", 40, compression="snappy")
    write(folder, "gzip-v2", 40, compression="gzip", data_page_version="2.0")
    write(folder, "brotli-groups", 300, compression="brotli", row_group_size=64)
    write(folder, "lz4-plain", 40, compression="lz4", use_dictionary=False)
    write(folder, "delta", 200, compression="none", use_dictionary=False, data_page_version="2.0",
          column_encoding={"id": "DELTA_BINARY_PACKED", "text": "DELTA_BYTE_ARRAY", "text2": "DELTA_LENGTH_BYTE_ARRAY",
                           "score": "BYTE_STREAM_SPLIT", "small": "DELTA_BINARY_PACKED"})
    write(folder, "zstd", 5, compression="zstd")


if __name__ == "__main__":
    main()
