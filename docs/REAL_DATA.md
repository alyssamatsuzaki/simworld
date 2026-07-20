# Real-data adapter

Stage 1 accepts a real firm panel without changing downstream code. Set
`data.source=real` and point `data.real_panel_path` at a Parquet (`.parquet` or `.pq`)
or CSV file. The adapter validates the file before it builds
`artifacts/data/panel_analysis.parquet`.

## Required firm-panel contract

The input must contain exactly these columns and types:

| Column | Polars type | Meaning |
|---|---|---|
| `firm_id` | `Int64` | Stable firm identifier |
| `quarter` | `Int64` | One-based reporting quarter |
| `region` | `Int64` | Rollout region |
| `treatment_quarter` | `Int64` | First treated quarter; use `-1` if never treated |
| `reported_compliant` | `Boolean` | Lagged, reported compliance status |
| `revenue_noisy` | `Float64` | Nonnegative observed revenue proxy |
| `audited` | `Boolean` | Observed audit indicator |
| `fined` | `Boolean` | Observed fine indicator |
| `alive` | `Boolean` | Firm active-status indicator |

Keys `(firm_id, quarter)` must be unique. Required identifiers cannot be null, float
columns must be finite, and extra columns are rejected. Cast CSV columns explicitly
before export if automatic CSV inference does not produce these types.

## Supporting observed tables

The minimal adapter replaces only `firm_panel.parquet`. It derives regressors from
the standard observed support files under `${paths.data}/observed/`:

- `firm_registry.parquet`
- `consumer_survey.parquet`
- `graphs/supply_edges.parquet`
- `graphs/market_edges.parquet`

These files use the contracts in `regworld.data.schema`. Supply-edge endpoints may
be serialized as `0` or `firm_0`. Provide only observed/degraded edges, never a true
or answer-key graph.

## Run

```bash
uv run python scripts/make_data.py \
  data=real \
  data.real_panel_path=/absolute/path/to/firm_panel.parquet
```

The resulting analysis panel keeps the established downstream field names, but its
decision-time regressors are estimates derived only from available observations:
lagged reports, observed supply neighbors, registry and association proxies, surveyed
privacy over observed firm–segment market links, and the published treatment schedule.
Adapt the documented
nine-quarter rollout phase in `data/ingest.py` if the real program used a different
published schedule.
