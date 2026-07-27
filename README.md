# GABench

A benchmark for evaluating graph analysis agents on multi-step reasoning tasks over real-world graph datasets.

---

## Code Structure

```
code/
├── new_genrator_ml/          # Task generation pipeline
│   ├── run_pipeline.sh       # Entry point: runs all 4 generation stages
│   ├── step0_precalc.py      # Computes global graph metrics and LLM-generated fuzzy tool descriptions
│   ├── step0_5_generate_anchors.py  # Selects anchor nodes for task construction
│   ├── step1_compute_gt.py   # Computes ground truth by running graph operators
│   ├── step2_generate_questions.py  # Generates natural-language questions via LLM
│   ├── mcp/                  # MCP tool definitions used during generation
│   └── ml/                   # LLM provider abstraction (provider.py)
│
├── sandbox/                  # Agent evaluation harness
│   ├── simple_eval.py        # Main evaluation runner (Docker-based)
│   ├── run_all.sh            # Master orchestration script (all datasets x tasks)
│   ├── insight.sh            # Thin wrapper for insight task evaluation
│   ├── ml.sh                 # Thin wrapper for ML task evaluation
│   ├── prediction.sh         # Thin wrapper for prediction task evaluation
│   ├── retrieval.sh          # Thin wrapper for retrieval task evaluation
│   ├── theory.sh             # Thin wrapper for theory task evaluation
│   ├── Dockerfile            # Docker image for isolated agent execution
│   ├── graph_analysis_skills/  # Graph tools injected into Docker for agents
│   │   ├── graph_io_tools.py
│   │   ├── graph_retrieval_tools.py
│   │   ├── graph_theory_tools.py
│   │   ├── graph_insight_tools.py
│   │   └── graph_ml_tools.py   # ML tools with ground-truth label injection
│   └── src/
│       ├── agents/           # Agent runner implementations (claudecode, codex, hermes, openclaw)
│       └── utils/            # Grading, Docker, and CLI utilities
│
├── metrics/                  # Post-evaluation metric computation
│   ├── compute_pass_rate.py  # Pass rate by dataset and sub-category
│   ├── compute_tool_metrics.py  # Tool-use precision / recall / F1
│   ├── llm_eval.py           # LLM-as-judge evaluation for open-ended answers
│   ├── compute_usage.py      # Token usage and cost aggregation
│   ├── extract_tools.py      # Extract tool call sequences from trajectories
│   └── *.sh                  # Batch wrappers for the above scripts
│
└── build_task/               # Legacy task-building scripts (superseded by new_genrator_ml)
```

---

## Task Generation

The generation pipeline has 4 sequential stages, all orchestrated by `run_pipeline.sh`.

### Configuration

Edit the variables at the top of `run_pipeline.sh` before running:

| Variable | Description |
|---|---|
| `DATASET_DIR` | Directory containing `.pt` graph dataset files |
| `GRAPH_FORMAT` | Graph format: `TAG`, `TPG`, or `PG` |
| `GRAPH_TYPE` | Optional domain hint: `citation`, `social`, `recommendation` |
| `ENABLE_ML` | `true` to include ML tasks (node/graph classification, link prediction) |
| `NUM_TASKS` | Number of tasks to generate per dataset |
| `NODE_CLS_DESC` | Natural-language description of the node classification task |

### Run the full pipeline

```bash
cd new_genrator_ml
bash run_pipeline.sh /path/to/dataset_dir
```

This runs the following stages for every `.pt` file found in `DATASET_DIR`:

**Stage 0** — Compute global graph metrics and generate fuzzy tool descriptions via LLM:
```bash
python step0_precalc.py \
    --dataset <dataset.pt> \
    --output global_metrics_cache.json \
    --graph-format PG \
    --mode ego \
    --enable-ml
```

**Stage 0.5** — Select anchor nodes:
```bash
python step0_5_generate_anchors.py \
    --dataset <dataset.pt> \
    --output anchor_nodes_cache.json \
    --max-size 500 \
    --graph-format PG
```

**Stage 1** — Compute ground truth by running graph operators:
```bash
python step1_compute_gt.py \
    --dataset <dataset.pt> \
    --num-tasks 200 \
    --output intermediate_ground_truth.json \
    --mode ego \
    --graph-format PG \
    --enable-ml
```

**Stage 2** — Generate natural-language questions via LLM:
```bash
python step2_generate_questions.py \
    --input intermediate_ground_truth.json \
    --output graph-prediction.json \
    --metrics-cache global_metrics_cache.json \
    --graph-format PG \
    --enable-ml
```

Output files per dataset:
- `graph-insight.json` — insight tasks (when `ENABLE_ML=false`)
- `graph-prediction.json` — prediction tasks (when `ENABLE_ML=true`)
- `pure_ml_bench.json` — pure ML tasks

---

## Sandbox (Agent Evaluation)

The sandbox runs each agent inside an isolated Docker container with the graph analysis tools injected.

### Prerequisites

Build the Docker image:
```bash
cd sandbox
docker build -t sandbox_eval:latest .
```

### Run a single task type

```bash
cd sandbox
python simple_eval.py \
    --pt_file /path/to/dataset.pt \
    --json_file /path/to/graph-insight.json \
    --agent claudecode \
    --model <model_name> \
    --api_key <your_api_key> \
    --api_url <api_base_url> \
    --skills_dir graph_analysis_skills \
    --output_dir output/insight/my_run \
    --parallel 4 \
    --limit 50
```

For ML/prediction tasks, also pass the masked data file:
```bash
python simple_eval.py \
    --pt_file /path/to/dataset.pt \
    --json_file /path/to/graph-prediction.json \
    --intermediate_pt_file /path/to/intermediate_ground_truth_masked_data.pt \
    --agent claudecode \
    --model <model_name> \
    --api_key <your_api_key> \
    --skills_dir graph_analysis_skills \
    --output_dir output/prediction/my_run \
    --parallel 2
```

Key arguments:

| Argument | Description |
|---|---|
| `--pt_file` | Path to the `.pt` graph dataset |
| `--json_file` | Path to the benchmark questions JSON |
| `--intermediate_pt_file` | Masked `.pt` file for ML/prediction tasks |
| `--agent` | Agent harness: `claudecode`, `codex`, `hermes`, `openclaw` |
| `--model` | Model name to evaluate |
| `--skills_dir` | Directory of graph tools to inject into Docker |
| `--parallel` | Number of concurrent Docker containers |
| `--limit` | Max number of questions to evaluate (0 = no limit) |
| `--port` | Base port for container bindings (default: 18789) |

### Run all datasets and task types

Configure `DATASETS`, `TASKS`, `MODEL`, and `LIMIT` at the top of `run_all.sh`, then:

```bash
cd sandbox
bash run_all.sh
```

This iterates over all datasets. For each dataset:
- Stage 1: runs `retrieval`, `theory`, `insight` in parallel
- Stage 2: runs `ml` then `prediction` sequentially

Existing output directories are skipped automatically (resume-safe).

---

## Evaluation Metrics

After evaluation, compute metrics from the output directories.

### Pass rate
```bash
cd metrics
python compute_pass_rate.py --output_dir /path/to/output
```

### Tool-use precision / recall / F1
```bash
python compute_tool_metrics.py --output_dir /path/to/output
```

### LLM-as-judge evaluation (open-ended answers)
```bash
python llm_eval.py --output_dir /path/to/output --model <judge_model>
```

### Token usage and cost
```bash
python compute_usage.py --output_dir /path/to/output
```

### Batch evaluation (all datasets)
```bash
bash metrics/quality_cal.sh
bash metrics/calculate_tools.sh
```
