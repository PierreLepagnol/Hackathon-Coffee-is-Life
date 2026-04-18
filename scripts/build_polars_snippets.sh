#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/build_polars_snippets.sh [source_dir] [output_file]

Defaults:
  source_dir  = polars-code
  output_file = artifacts/polars-snippets.json

The script extracts snippet blocks delimited with:
  --8<-- [start:<name>]
  --8<-- [end:<name>]

Metadata is refined in 3 passes:
  1. Path metadata     -> category, topic, source lines
  2. Snippet metadata  -> role/title from snippet name
  3. Code metadata     -> inferred operations, tags, purpose
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

resolve_path() {
  local path="$1"
  python3 - "$REPO_ROOT" "$path" <<'PY'
import os
import sys

repo_root, raw_path = sys.argv[1:3]
if os.path.isabs(raw_path):
    print(os.path.abspath(raw_path))
else:
    print(os.path.abspath(os.path.join(repo_root, raw_path)))
PY
}

SOURCE_DIR="$(resolve_path "${1:-polars-code}")"
OUTPUT_FILE="$(resolve_path "${2:-artifacts/polars-snippets.json}")"

if [[ ! -d "$SOURCE_DIR" ]]; then
  printf 'Source directory not found: %s\n' "$SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname -- "$OUTPUT_FILE")"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SNIPPET_DIR="$TMP_DIR/snippets"
MANIFEST_FILE="$TMP_DIR/manifest.tsv"
mkdir -p "$SNIPPET_DIR"

extract_marked_snippets() {
  local file="$1"
  local rel_path="$2"
  local file_key="$3"

  awk \
    -v rel_path="$rel_path" \
    -v file_key="$file_key" \
    -v snippets_dir="$SNIPPET_DIR" \
    -v manifest_file="$MANIFEST_FILE" \
    '
function sanitize(text,    out) {
  out = text
  gsub(/[^[:alnum:]_.-]+/, "_", out)
  return out
}

function fail(message) {
  print message > "/dev/stderr"
  exit 2
}

BEGIN {
  in_block = 0
  snippet_count = 0
}

{
  if (match($0, /--8<--? \[start:([^]]+)\]/, parts)) {
    if (in_block) {
      fail("Nested snippet start in " rel_path " at line " NR)
    }

    in_block = 1
    snippet_count += 1
    snippet_name = parts[1]
    marker_start_line = NR
    content_start_line = NR + 1
    out_file = sprintf("%s/%s__%03d__%s.txt", snippets_dir, file_key, snippet_count, sanitize(snippet_name))
    next
  }

  if (match($0, /--8<--? \[end:([^]]+)\]/, parts)) {
    if (!in_block) {
      fail("Unexpected snippet end in " rel_path " at line " NR)
    }
    if (parts[1] != snippet_name) {
      fail("Mismatched snippet markers in " rel_path ": expected " snippet_name ", got " parts[1] " at line " NR)
    }

    content_end_line = NR - 1
    printf("%s\t%s\t%d\t%d\t%d\t%d\t%d\t%s\n", rel_path, snippet_name, snippet_count, marker_start_line, content_start_line, content_end_line, NR, out_file) >> manifest_file
    close(out_file)

    in_block = 0
    snippet_name = ""
    out_file = ""
    next
  }

  if (in_block) {
    print $0 >> out_file
  }
}

END {
  if (in_block) {
    fail("Unclosed snippet " snippet_name " in " rel_path)
  }
  exit(snippet_count > 0 ? 0 : 3)
}
' "$file"
}

extract_whole_file() {
  local file="$1"
  local rel_path="$2"
  local file_key="$3"
  local snippet_file="$SNIPPET_DIR/${file_key}__001__file.txt"
  local line_count

  cp -- "$file" "$snippet_file"
  line_count="$(python3 - "$file" <<'PY'
from pathlib import Path
import sys

print(len(Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()))
PY
)"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$rel_path" \
    "file" \
    "1" \
    "0" \
    "1" \
    "$line_count" \
    "0" \
    "$snippet_file" >> "$MANIFEST_FILE"
}

while IFS= read -r -d '' file; do
  rel_path="${file#"$SOURCE_DIR"/}"
  file_key="$(python3 - "$rel_path" <<'PY'
import re
import sys

print(re.sub(r'[^A-Za-z0-9_.-]+', '_', sys.argv[1]))
PY
)"

  if extract_marked_snippets "$file" "$rel_path" "$file_key"; then
    :
  else
    status=$?
    if [[ "$status" -eq 3 ]]; then
      extract_whole_file "$file" "$rel_path" "$file_key"
    else
      exit "$status"
    fi
  fi
done < <(find "$SOURCE_DIR" -type f -print0 | sort -z)

if [[ ! -s "$MANIFEST_FILE" ]]; then
  printf 'No snippets were extracted from %s\n' "$SOURCE_DIR" >&2
  exit 1
fi

python3 - "$MANIFEST_FILE" "$SOURCE_DIR" "$OUTPUT_FILE" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

manifest_path = Path(sys.argv[1])
source_dir = Path(sys.argv[2]).resolve()
output_file = Path(sys.argv[3]).resolve()
source_root_name = source_dir.name

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".sh": "bash",
    ".md": "markdown",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sql": "sql",
}

OPERATION_PATTERNS = [
    ("sql", r"\bSQLContext\b|\.sql\("),
    ("join_asof", r"\bjoin_asof\b"),
    ("join_where", r"\bjoin_where\b"),
    ("join", r"\bjoin\("),
    ("group_by_dynamic", r"\bgroup_by_dynamic\b"),
    ("group_by_rolling", r"\bgroup_by_rolling\b|\.rolling\("),
    ("group_by", r"\bgroup_by\("),
    ("aggregation", r"\.agg\("),
    ("select", r"\.select\("),
    ("with_columns", r"\.with_columns\("),
    ("filter", r"\.filter\("),
    ("pivot", r"\bpivot\("),
    ("unpivot", r"\bunpivot\(|\bmelt\("),
    ("window", r"\.over\("),
    ("lazy", r"\.lazy\("),
    ("csv", r"\bread_csv\b|\bscan_csv\b|\bwrite_csv\b"),
    ("json", r"\bread_json\b|\bscan_ndjson\b|\bwrite_json\b|\bwrite_ndjson\b"),
    ("parquet", r"\bread_parquet\b|\bscan_parquet\b|\bwrite_parquet\b"),
    ("excel", r"\bread_excel\b|\bwrite_excel\b"),
    ("database", r"\bread_database\b|\bwrite_database\b|\bread_sql\b"),
    ("cloud", r"s3://|gs://|azure"),
    ("datetime", r"\.dt\."),
    ("strings", r"\.str\."),
    ("lists", r"\.list\."),
    ("structs", r"\.struct\."),
    ("categoricals", r"Categorical|Enum"),
    ("visualization", r"hvplot|matplotlib|plotnine|seaborn|plotly|altair"),
    ("arrow", r"\barrow\b|pycapsule"),
    ("numpy", r"\bnumpy\b|\bnp\."),
    ("user_defined_function", r"\bmap_elements\b|\bapply\b|def "),
]

OPERATION_LABELS = {
    "sql": "Polars SQL",
    "join_asof": "as-of joins",
    "join_where": "non-equi joins",
    "join": "joins",
    "group_by_dynamic": "dynamic group-bys",
    "group_by_rolling": "rolling windows",
    "group_by": "group-bys",
    "aggregation": "aggregations",
    "select": "column selection",
    "with_columns": "column creation",
    "filter": "filtering",
    "pivot": "pivoting",
    "unpivot": "unpivoting",
    "window": "window expressions",
    "lazy": "lazy execution",
    "csv": "CSV I/O",
    "json": "JSON I/O",
    "parquet": "Parquet I/O",
    "excel": "Excel I/O",
    "database": "database I/O",
    "cloud": "cloud storage I/O",
    "datetime": "datetime operations",
    "strings": "string expressions",
    "lists": "list expressions",
    "structs": "struct expressions",
    "categoricals": "categorical data",
    "visualization": "visualization",
    "arrow": "Arrow interoperability",
    "numpy": "NumPy integration",
    "user_defined_function": "user-defined functions",
}


def slug_to_label(value: str) -> str:
    cleaned = value.replace("-", " ").replace("_", " ").strip()
    return cleaned.title() if cleaned else "Root"


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def detect_language(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "text")


def classify_role(snippet_name: str, code: str) -> str:
    name = snippet_name.lower()

    if name == "file":
        return "file"
    if name in {"setup", "context"}:
        return "setup"
    if name.startswith(("setup", "prep", "prepare", "clean", "register")):
        return "setup"
    if re.fullmatch(r"df\d*|dataframe|dataset|players|tokens|props_[a-z0-9_]+|df_[a-z0-9_]+", name):
        return "data"
    return "example"


def detect_operations(code: str) -> list[str]:
    return [name for name, pattern in OPERATION_PATTERNS if re.search(pattern, code, flags=re.IGNORECASE | re.MULTILINE)]


def format_context(category_label: str, subcategory_labels: list[str], topic_label: str) -> str:
    return " / ".join([category_label, *subcategory_labels, topic_label])


def build_purpose(role: str, context_label: str, operations: list[str]) -> str:
    primary_operation = operations[0] if operations else None
    primary_label = OPERATION_LABELS.get(primary_operation, primary_operation.replace("_", " ") if primary_operation else "example")

    if role == "setup":
        return f"Setup code for the {context_label} snippets."
    if role == "data":
        if primary_operation:
            return f"Builds example input data for {primary_label} in {context_label}."
        return f"Builds example input data for the {context_label} snippets."
    if role == "file":
        return f"Full source file from {context_label} when no explicit snippet markers are available."
    if primary_operation:
        return f"Demonstrates {primary_label} in {context_label}."
    return f"Example snippet from {context_label}."


def make_title(topic_label: str, snippet_label: str) -> str:
    if snippet_label.lower() in {"setup", "file"}:
        return f"{topic_label} {snippet_label}"
    return f"{topic_label}: {snippet_label}"


snippets: list[dict] = []
files_seen: set[str] = set()

with manifest_path.open("r", encoding="utf-8", newline="") as handle:
    reader = csv.reader(handle, delimiter="\t")
    for row in reader:
        rel_path, snippet_name, ordinal, marker_start, content_start, content_end, marker_end, snippet_file = row

        source_path = Path(rel_path)
        code = Path(snippet_file).read_text(encoding="utf-8")
        parts = list(source_path.parts)
        file_stem = source_path.stem
        category = parts[0] if len(parts) > 1 else "root"
        subcategories = parts[1:-1] if len(parts) > 2 else []
        topic = file_stem
        language = detect_language(source_path)
        role = classify_role(snippet_name, code)
        operations = detect_operations(code)

        category_label = slug_to_label(category)
        subcategory_labels = [slug_to_label(value) for value in subcategories]
        topic_label = slug_to_label(topic)
        snippet_label = slug_to_label(snippet_name)
        context_label = format_context(category_label, subcategory_labels, topic_label)
        purpose = build_purpose(role, context_label, operations)

        tags = unique(
            [
                category,
                *subcategories,
                topic,
                snippet_name,
                role,
                language,
                *operations,
            ]
        )

        line_count = len(code.splitlines())
        files_seen.add(rel_path)
        snippets.append(
            {
                "id": f"{rel_path}#{snippet_name}",
                "language": language,
                "source": {
                    "root": source_root_name,
                    "path": rel_path,
                    "file_stem": file_stem,
                    "category": category,
                    "subcategories": subcategories,
                    "snippet_name": snippet_name,
                    "snippet_ordinal": int(ordinal),
                    "marker_start_line": int(marker_start),
                    "content_start_line": int(content_start),
                    "content_end_line": int(content_end),
                    "marker_end_line": int(marker_end),
                    "sha1": hashlib.sha1(code.encode("utf-8")).hexdigest(),
                },
                "metadata": {
                    "path": {
                        "category_label": category_label,
                        "subcategory_labels": subcategory_labels,
                        "topic": topic,
                        "topic_label": topic_label,
                        "context_label": context_label,
                    },
                    "snippet": {
                        "label": snippet_label,
                        "role": role,
                        "title": make_title(topic_label, snippet_label),
                    },
                    "inferred": {
                        "operations": operations,
                        "primary_operation": operations[0] if operations else None,
                        "purpose": purpose,
                        "tags": tags,
                    },
                },
                "stats": {
                    "line_count": line_count,
                    "char_count": len(code),
                    "has_import": "import " in code,
                    "has_print": "print(" in code,
                    "has_polars": "pl." in code or "import polars" in code,
                    "is_empty": code.strip() == "",
                },
                "code": code,
            }
        )

snippets.sort(key=lambda item: (item["source"]["path"], item["source"]["content_start_line"], item["source"]["snippet_ordinal"]))

payload = {
    "schema_version": 1,
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "source_root": source_root_name,
    "source_dir": str(source_dir),
    "file_count": len(files_seen),
    "snippet_count": len(snippets),
    "categories": dict(sorted(Counter(item["source"]["category"] for item in snippets).items())),
    "snippets": snippets,
}

output_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY

printf 'Wrote %s\n' "$OUTPUT_FILE"
