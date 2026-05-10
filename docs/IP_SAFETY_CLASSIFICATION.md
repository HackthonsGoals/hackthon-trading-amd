# Step 1 Repository Classification

Classification was performed from the workspace root `D:\WEBSITE\open-env-nifty500`.
The new project uses no copied code or data from the original trading system.

| file_path | classification | reason |
|---|---|---|
| `project_history.md` | SENSITIVE | Narrative project history can reveal architecture, decisions, and implementation sequence. |
| `logs/**` | CRITICAL | Live-trader logs and runtime outputs can expose decisions, symbols, order flow, and operational behavior. |
| `data/portfolio.json` | CRITICAL | Portfolio/capital state is private business data. |
| `files (1)/llm_config.py` | CRITICAL | LLM configuration may include private endpoints, prompt wiring, or credentials. |
| `files (1)/*.ps1` | SENSITIVE | Maintenance scripts may reveal operational workflow. |
| `scripts/dhan_diagnostic.py` | CRITICAL | Broker diagnostic path; never copy. |
| `assets/*.pdf` | SAFE | Public/index reference documents only; not used in the demo. |
| `assets/*.png` | SAFE | Presentation images only; not used in the demo. |
| `assets/forex/**` | SENSITIVE | External trading articles and downloaded page assets; not needed and may carry copyright or strategy text. |
| `openenv-nifty500/.venv/**` | CRITICAL | Third-party dependency dump and possible local environment state; never copy. |
| `openenv-nifty500/.tmp/**` | CRITICAL | Temporary runtime state, inaccessible in part; never copy. |
| `openenv-nifty500/.pytest_cache/**` | SENSITIVE | Generated test cache; no reuse value. |
| `openenv-nifty500/**/__pycache__/**` | CRITICAL | Compiled code artifacts; never copy. |
| `openenv-nifty500/.streamlit/**` | SENSITIVE | Local dashboard configuration; not portable. |
| `openenv-nifty500/.agent/**` | CRITICAL | Agent memory/skills may reveal private repo-specific instructions. |
| `openenv-nifty500/.codex/**` | CRITICAL | Codex skills and project instructions may reveal private logic. |
| `openenv-nifty500/app.py` | CRITICAL | Production dashboard imports live feed, LLM, simulator, signals, execution, and prompts. |
| `openenv-nifty500/local_dashboard.py` | CRITICAL | Dashboard logic over production trading outputs. |
| `openenv-nifty500/inference.py` | CRITICAL | LLM inference and decision prompt surface. |
| `openenv-nifty500/client.py` | SENSITIVE | Client integration code; not needed for standalone demo. |
| `openenv-nifty500/openenv.yaml` | CRITICAL | OpenEnv/RL system configuration. |
| `openenv-nifty500/Dockerfile` | SENSITIVE | Deployment shape of original system. |
| `openenv-nifty500/requirements.txt` | SAFE | Dependency names only, but not copied to avoid coupling. |
| `openenv-nifty500/pyproject.toml` | SAFE | Packaging metadata only, but not copied to avoid coupling. |
| `openenv-nifty500/uv.lock` | SENSITIVE | Full dependency lock can reveal environment and is unnecessary. |
| `openenv-nifty500/README.md` | SENSITIVE | Describes original system architecture and workflows. |
| `openenv-nifty500/SYSTEM_AUDIT.md` | CRITICAL | Audit details can expose weaknesses and internal logic. |
| `openenv-nifty500/SYSTEM_RUNBOOK.md` | CRITICAL | Operational runbook for original system. |
| `openenv-nifty500/TODO.md` | SENSITIVE | Roadmap and unresolved private logic. |
| `openenv-nifty500/*Analysis.ipynb` | CRITICAL | Analysis notebooks may include strategy outputs and trade behavior. |
| `openenv-nifty500/*Dashboard.ipynb` | CRITICAL | Dashboard notebooks expose trading outputs and visualization logic. |
| `openenv-nifty500/train.py` | CRITICAL | Training entrypoint for private AI/RL workflow. |
| `openenv-nifty500/train/**` | CRITICAL | SFT/GRPO/reward functions/templates and dataset builders are explicitly prohibited. |
| `openenv-nifty500/eval/**` | CRITICAL | Evaluation logic for signals, execution, LLM endpoints, and trade plans. |
| `openenv-nifty500/env/**` | CRITICAL | Trading environment/OpenEnv logic. |
| `openenv-nifty500/server/**` | CRITICAL | Original trading server environment. |
| `openenv-nifty500/tasks/**` | SENSITIVE | Task wrappers around original system. |
| `openenv-nifty500/scratch/**` | SENSITIVE | Experimental tests may expose private pipeline behavior. |
| `openenv-nifty500/tests/**` | SENSITIVE | Tests encode strategy, risk, signal, execution, and policy expectations. |
| `openenv-nifty500/scripts/**` | CRITICAL | Contains broker auth, live trading, RL loop, audits, fees, inference, and data collection scripts. |
| `openenv-nifty500/openenv_nifty500/data/schemas.py` | SAFE | Generic OHLCV schema concepts are safe, but code was not copied. |
| `openenv-nifty500/openenv_nifty500/data/loaders.py` | SENSITIVE | Data loading may encode private source assumptions. |
| `openenv-nifty500/openenv_nifty500/data/feeds.py` | CRITICAL | Live/local feed routing and broker-related data paths. |
| `openenv-nifty500/openenv_nifty500/data/news.py` | SENSITIVE | News ingestion/intelligence workflow. |
| `openenv-nifty500/openenv_nifty500/data/intelligence.py` | CRITICAL | Intelligence features can leak signal context. |
| `openenv-nifty500/openenv_nifty500/data/preprocess.py` | SENSITIVE | Feature preparation may encode private assumptions. |
| `openenv-nifty500/openenv_nifty500/data/candle_aggregator.py` | SAFE | Generic aggregation concept only; code not copied. |
| `openenv-nifty500/openenv_nifty500/engine/trading_logic.py` | CRITICAL | Core trading logic. |
| `openenv-nifty500/openenv_nifty500/engine/index_provider.py` | SENSITIVE | Original universe/provider logic. |
| `openenv-nifty500/openenv_nifty500/environment/**` | CRITICAL | RL/trading environment. |
| `openenv-nifty500/openenv_nifty500/execution/**` | CRITICAL | Trade policy and broker/order execution. |
| `openenv-nifty500/openenv_nifty500/models/**` | CRITICAL | Order models and production domain objects. |
| `openenv-nifty500/openenv_nifty500/monitoring/**` | CRITICAL | Watchlist, alerts, paper trades, signal logging, and market state. |
| `openenv-nifty500/openenv_nifty500/outcomes/**` | CRITICAL | Outcome simulation tied to trading lifecycle. |
| `openenv-nifty500/openenv_nifty500/patterns/**` | CRITICAL | Proprietary pattern and signal generation logic. |
| `openenv-nifty500/openenv_nifty500/recommendations/**` | CRITICAL | Trade recommender/business logic. |
| `openenv-nifty500/openenv_nifty500/risk_reward/**` | CRITICAL | Risk/reward formulas and sizing logic. |
| `openenv-nifty500/openenv_nifty500/signals/**` | CRITICAL | Signal generation logic. |
| `openenv-nifty500/openenv_nifty500/state/**` | CRITICAL | Order/global state. |
| `openenv-nifty500/openenv_nifty500/trade_plan/**` | CRITICAL | Trade-plan schema and lifecycle domain logic. |
| `openenv-nifty500/openenv_nifty500/dhan_client.py` | CRITICAL | Broker API client. |
| `openenv-nifty500/openenv_nifty500/llm_config.py` | CRITICAL | LLM endpoint/client config and prompt-adjacent code. |
| `openenv-nifty500/openenv_nifty500/live_simulator.py` | CRITICAL | Original simulator tied to production domain. |
| `openenv-nifty500/openenv_nifty500/simulator.py` | CRITICAL | Original execution simulator. |
| `openenv-nifty500/openenv_nifty500/portfolio.py` | CRITICAL | Portfolio/capital management. |
| `openenv-nifty500/openenv_nifty500/graders.py` | CRITICAL | Reward/grading logic. |
| `openenv-nifty500/openenv_nifty500/journal.py` | SENSITIVE | Trade journal persistence. |
| `openenv-nifty500/openenv_nifty500/settings.py` | CRITICAL | Environment variables and operational toggles. |
| `openenv-nifty500/openenv_nifty500/client.py` | SENSITIVE | Package client helper. |
| `openenv-nifty500/openenv_nifty500/streaming.py` | CRITICAL | Live streaming integration. |
| `openenv-nifty500/openenv_nifty500/tasks.py` | SENSITIVE | Task wrapper around original package. |
| `openenv-nifty500/data/*_audit.csv` | CRITICAL | Historical symbol audit data used by original system. |
| `openenv-nifty500/data/*sft*.jsonl` | CRITICAL | Training datasets and prompt/decision examples. |
| `openenv-nifty500/data/live_journal.json` | CRITICAL | Trade journal/runtime output. |
| `openenv-nifty500/data/portfolio.json` | CRITICAL | Portfolio/capital data. |
| `openenv-nifty500/data/watchlist.json` | CRITICAL | Active trading universe/watchlist. |
| `openenv-nifty500/data/instrument_cache.json` | CRITICAL | Broker/instrument mapping cache. |
| `openenv-nifty500/data/nifty500*.csv` | SENSITIVE | Original universe/data inputs; not copied. |
| `openenv-nifty500/data/replay/**` | SENSITIVE | Historical OHLCV replay corpus; not copied. |
| `openenv-nifty500/logs/**` | CRITICAL | Production and audit logs. |
| `openenv-nifty500/artifacts/**` | CRITICAL | Training/eval/model artifacts. |
| `openenv-nifty500/models/**` | CRITICAL | Original model artifacts. |
| `openenv-nifty500/notebooks/**` | CRITICAL | Training and analysis notebooks. |
| `openenv-nifty500/docs/**` | SENSITIVE | Operational docs and screenshots. |
| `openenv-nifty500/docs/screenshots/**` | SENSITIVE | Demo screenshots may expose original UI and outputs. |

## Component Map

| component | original locations | safety decision |
|---|---|---|
| Signal generation | `openenv_nifty500/signals`, `patterns`, `recommendations`, `engine/trading_logic.py` | CRITICAL, not copied. |
| Execution | `execution`, `dhan_client.py`, `state/order_state.py`, live scripts | CRITICAL, not copied. |
| Logging/monitoring | `logs`, `monitoring`, `journal.py`, dashboards | CRITICAL/SENSITIVE, not copied. |
| Visualization | `app.py`, `local_dashboard.py`, notebooks, screenshots | CRITICAL/SENSITIVE, rebuilt from scratch. |
| Data | `data`, `data/replay`, loaders/feeds | SENSITIVE/CRITICAL, replaced with tiny mock CSV. |
| Training/RL/LLM | `train`, `env`, `inference.py`, `llm_config.py`, artifacts | CRITICAL, not copied. |

