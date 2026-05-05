# Hackathon Execution Task Plan

This file is the step-by-step operating procedure for completing and presenting
the AMD GPU-Accelerated AI Signal Pipeline hackathon project.

The goal is to keep the project simple, fast, visually compelling, and safe.
This is a demo of an AI pipeline and GPU acceleration. It is not a trading bot.

## Core Rules

Follow these rules throughout the hackathon:

- Do not add real trading strategy logic.
- Do not add broker APIs or exchange integrations.
- Do not add `.env` files, credentials, API keys, tokens, or private configs.
- Do not copy code from the real trading system.
- Do not add RL/OpenEnv code.
- Do not add private prompts or private model weights.
- Keep signal generation visibly dummy and non-realistic.
- Keep all data public, synthetic, or clearly demo-only.
- Make GPU performance easy to see in the dashboard and README.

## Final Submission Target

The finished submission should answer these questions in under 10 seconds:

1. What is this?
   - An AMD GPU-accelerated AI signal pipeline demo.
2. What does AMD GPU acceleration do?
   - It speeds up batch inference and improves throughput.
3. What AI is included?
   - Market-style batch inference plus open-weight sentiment analysis.
4. Is this safe?
   - Yes. It uses mock data, fake signals, and simulated execution only.
5. Can judges run it quickly?
   - Yes. `pip install -r requirements.txt` and `streamlit run app.py`.

## Phase 0: Workspace Separation

Purpose: keep the hackathon repo isolated from the actual trading system.

### Tasks

1. Clone or move the hackathon repo into a clean folder outside the production workspace.
2. Use this repo as the only hackathon working directory:

```bash
git clone https://github.com/HackthonsGoals/hackthon-trading-amd.git
cd hackthon-trading-amd
```

3. Confirm the project has this structure:

```text
data/
models/
engine/
simulator/
dashboard/
utils/
sentiment/
scripts/
assets/screenshots/
app.py
requirements.txt
README.md
HACKATHON_AUDIT.md
IP_SAFETY_CLASSIFICATION.md
HACKATHON_TASK_PLAN.md
```

### Acceptance Criteria

- The repo is outside the actual trading system.
- No production files are present.
- `git status` is clean before starting new work.

## Phase 1: Environment Setup

Purpose: make the project runnable in under two minutes on a clean machine.

### Tasks

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate it.

Windows:

```bash
.venv\Scripts\activate
```

Linux or macOS:

```bash
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the dashboard:

```bash
streamlit run app.py
```

5. Open the local URL shown by Streamlit.

### Acceptance Criteria

- App starts without Python import errors.
- Dashboard loads in the browser.
- Screenshots in README render on GitHub.
- If GPU is unavailable, dashboard clearly says GPU is pending/unavailable.

## Phase 2: Data Gathering

Purpose: collect safe demo data without leaking real trading data.

### Approved Data Types

- Synthetic OHLCV rows.
- Public sample headlines.
- Synthetic financial-style news headlines.
- Small public datasets with permissive licenses.

### Prohibited Data Types

- Real trade logs.
- Broker exports.
- Portfolio files.
- Watchlists from the actual system.
- Historical production replay data.
- Private research notes.
- Private model outputs.
- Private prompts.

### Tasks

1. Review existing mock market data:

```text
data/sample_ohlcv.csv
```

2. Review sample headlines:

```text
data/sample_headlines.csv
```

3. Generate or refresh synthetic sentiment data:

```bash
python scripts/generate_sentiment_dataset.py
```

4. Confirm the dataset exists:

```text
data/sentiment_dataset.csv
```

5. Confirm it has columns:

```text
text,label
```

### Acceptance Criteria

- Dataset is synthetic or public.
- No private symbols, trade IDs, broker IDs, or account data.
- Sentiment dataset has balanced labels.

## Phase 3: Labeling Procedure

Purpose: produce clean sentiment labels for lightweight fine-tuning.

### Label Definitions

Use exactly three labels:

```text
POSITIVE
NEGATIVE
NEUTRAL
```

### POSITIVE Examples

- Company beats estimates.
- Company raises guidance.
- Demand growth is reported.
- Analyst upgrade is announced.
- Major contract is won.

### NEGATIVE Examples

- Company misses estimates.
- Company cuts guidance.
- Demand slowdown is reported.
- Analyst downgrade is announced.
- Margin pressure or uncertainty is reported.

### NEUTRAL Examples

- Board meeting announced.
- Product roadmap updated.
- Office opened.
- Stable operations reported.
- Existing guidance maintained.

### Tasks

1. Inspect class balance.
2. Remove duplicate headline rows.
3. Remove ambiguous labels if they are too subjective.
4. Keep label names uppercase.
5. Keep each row short, headline-like, and demo-safe.

### Recommended Dataset Size

- Minimum demo: 700 rows.
- Stronger demo: 1,500 to 2,000 rows.
- Keep classes balanced.

### Acceptance Criteria

- Labels are balanced.
- No private data is present.
- All labels are one of `POSITIVE`, `NEGATIVE`, or `NEUTRAL`.

## Phase 4: Data Preparation

Purpose: validate, clean, and split data before training.

### Tasks

1. Confirm required columns:

```text
text,label
```

2. Validate labels:

```text
POSITIVE
NEGATIVE
NEUTRAL
```

3. Normalize text:

- trim whitespace
- remove empty rows
- remove duplicate rows
- keep punctuation simple

4. Split data:

```text
train: 80%
validation: 10-15%
test: 5-10%
```

5. Record dataset summary in the README or a small report:

```text
total rows
class counts
train/validation/test counts
example rows
```

### Optional Script To Add

If time allows, create:

```text
scripts/validate_dataset.py
```

It should print:

- row count
- missing values
- duplicate count
- class balance
- invalid labels

### Acceptance Criteria

- Dataset passes validation.
- Class distribution is understandable.
- Training script can read the dataset without manual edits.

## Phase 5: Sentiment Fine-Tuning

Purpose: train a lightweight open-weight sentiment model.

### Model Choice

Default model:

```text
distilbert-base-uncased
```

Reason:

- open-weight
- lightweight
- fast enough for hackathon demo
- supported by Hugging Face Transformers
- GPU-compatible through PyTorch
- easy to fine-tune

### Training Command

```bash
python scripts/train_sentiment_model.py
```

### Default Training Settings

```text
epochs: 3
batch size: 16
max length: 96
optimizer: AdamW
device: cuda if available, otherwise cpu
```

### Output Path

```text
models/sentiment-distilbert/
```

### Important Notes

- Do not commit large model checkpoints unless the hackathon requires it.
- If checkpoints are not committed, explain how to train them.
- Keep fallback sentiment visibly marked as demo fallback.

### Acceptance Criteria

- Training completes.
- Validation accuracy is printed.
- Model/tokenizer save successfully.
- Inference module loads the saved model if present.

## Phase 6: Sentiment Inference

Purpose: provide fast headline sentiment predictions in the app.

### Input

```text
news headline text
```

### Output

```json
{
  "sentiment": "POSITIVE",
  "score": 0.91
}
```

### Tasks

1. Use:

```text
sentiment/sentiment_inference.py
```

2. Confirm it supports:

- single headline inference
- batch headline inference
- GPU when available
- CPU fallback

3. Confirm dashboard displays:

- headline text
- sentiment label
- score
- signed score
- backend
- device

### Acceptance Criteria

- Batch inference works.
- Sentiment panel renders.
- No proprietary APIs are used.

## Phase 7: Market Inference Benchmark

Purpose: make AMD GPU acceleration visible and measurable.

### Benchmark Requirements

Benchmark:

```text
CPU vs GPU
batch size 100
batch size 1000
latency
throughput
speedup ratio
```

### Tasks

1. Use:

```text
utils/benchmark.py
```

2. Confirm it returns:

```text
records[]
speedups[]
best_speedup
```

3. Confirm dashboard plots:

- CPU throughput
- GPU throughput
- batch sizes
- speedup

### Acceptance Criteria

- CPU results always show.
- GPU results show when ROCm/CUDA PyTorch is available.
- Dashboard does not fake GPU numbers.

## Phase 8: Sentiment Benchmark

Purpose: show single vs batch sentiment inference performance.

### Metrics

Track:

- single headline latency
- batch headline latency
- batch throughput
- CPU vs GPU
- speedup ratio

### Tasks

1. Use:

```text
sentiment/benchmark.py
```

2. Confirm dashboard shows:

- sentiment benchmark table
- sentiment throughput chart

### Acceptance Criteria

- Benchmark runs without requiring a GPU.
- GPU result appears only when available.
- Charts are readable.

## Phase 9: Dummy Signal Integration

Purpose: show sentiment impact without adding real strategy logic.

### Rules

Keep the logic simple:

- positive sentiment slightly nudges `BUY`
- negative sentiment slightly nudges `SELL`
- neutral sentiment mostly leaves signal unchanged

Do not add:

- pattern detectors
- risk formulas
- stop-loss algorithms
- capital allocation
- real entry logic
- production indicators

### Tasks

1. Use:

```text
engine/dummy_signal_generator.py
```

2. Confirm signal output includes:

```text
symbol
signal
entry
sl
target
confidence
sentiment
sentiment_score
sentiment_adjustment
```

3. Confirm README says this is not a real strategy.

### Acceptance Criteria

- Signal logic is transparent and non-realistic.
- Dashboard shows sentiment fields.
- IP safety scan passes.

## Phase 10: Fake Execution Simulation

Purpose: make the demo feel alive without real execution.

### Simulator Features

The simulator should include:

- fake trade IDs
- side: BUY or SELL
- slippage
- opened timestamp
- closed timestamp
- status: CLOSED
- P&L
- return percentage
- total P&L
- win rate
- average return

### Tasks

1. Use:

```text
simulator/execution_simulator.py
```

2. Confirm dashboard shows:

- closed trades
- win rate
- average return
- cumulative P&L
- trade lifecycle table

### Acceptance Criteria

- No real external execution.
- No order placement code.
- No broker terminology in executable code.
- Simulation is clearly fake.

## Phase 11: Dashboard Polish

Purpose: make the dashboard judge-ready.

### First Viewport Must Show

- project title
- device
- latency
- throughput
- GPU speedup
- demo P&L
- live signal feed
- sentiment section start

### Main Panels

1. Live Signal Feed
2. Headline Sentiment
3. Sentiment Distribution
4. Sentiment Score Over Time
5. Market Data
6. Trade Simulation
7. Performance Metrics
8. Trade Lifecycle
9. Cumulative P&L

### UI Rules

- Use color-coded signal labels.
- Keep text concise.
- Avoid jargon in headings.
- Charts should explain themselves.
- Do not show raw debug JSON unless useful.

### Acceptance Criteria

- Dashboard is understandable without reading code.
- Screenshots look clean.
- No embarrassing blank sections.

## Phase 12: Screenshot Capture

Purpose: make the README visually compelling.

### Required Screenshots

Save screenshots to:

```text
assets/screenshots/
```

Required files:

```text
dashboard-overview.png
gpu-benchmark.png
sentiment-panel.png
```

### Capture Procedure

1. Start dashboard:

```bash
streamlit run app.py
```

2. Capture screenshots manually or with:

```bash
node scripts/capture_screenshots.js
```

3. Confirm files exist:

```bash
ls assets/screenshots
```

4. Confirm README displays:

```markdown
![Dashboard overview](assets/screenshots/dashboard-overview.png)
![GPU benchmark](assets/screenshots/gpu-benchmark.png)
![Sentiment panel](assets/screenshots/sentiment-panel.png)
```

### Acceptance Criteria

- Screenshots are visible in GitHub README.
- No blank white screenshots.
- Images show real dashboard content.

## Phase 13: README Finalization

Purpose: make the GitHub repo strong enough to judge from the README alone.

### README Must Include

- project title
- one-sentence overview
- screenshots
- feature list
- architecture diagram
- AMD optimization explanation
- benchmark explanation
- sentiment pipeline explanation
- setup instructions
- demo flow
- safety boundary

### Strong Opening

The README should immediately communicate:

```text
AMD GPU-accelerated AI signal pipeline with open-weight sentiment,
batch inference benchmarks, and simulated trading dashboard.
```

### Acceptance Criteria

- README opens strong.
- Screenshots are near the top.
- Setup commands are copy-paste friendly.
- Safety boundary is explicit.

## Phase 14: GitHub Repo Metadata

Purpose: make the repo discoverable and professional.

### Description

Use this exact description:

```text
AMD GPU-accelerated AI signal pipeline with open-weight sentiment, batch inference benchmarks, and simulated trading dashboard.
```

### Topics

Use these topics:

```text
amd
rocm
pytorch
streamlit
sentiment-analysis
distilbert
gpu-acceleration
hackathon
ai-pipeline
simulation
```

### Acceptance Criteria

- GitHub description is set.
- Topics are set.
- README images render.

## Phase 15: IP Safety Scan

Purpose: prevent accidental leakage before final sharing.

### Scan For

- `.env`
- API keys
- tokens
- passwords
- OpenEnv
- RL
- GRPO
- broker code
- order placement
- real strategy logic
- private prompts
- capital allocation
- risk formulas

### Suggested Commands

```bash
find . -name ".env" -o -name ".env.*"
```

```bash
rg -n "api_key|secret|password|token|OpenEnv|GRPO|broker|place_order|SYSTEM_PROMPT|risk_reward|trade_policy"
```

### Expected Result

- No sensitive executable code hits.
- Documentation may mention prohibited items only as exclusions.

### Acceptance Criteria

- No `.env`.
- No API keys.
- No OpenEnv/RL core.
- No real strategy logic.
- No broker code.
- README safety statement is clear.

## Phase 16: Final Test Run

Purpose: verify the project works right before submission.

### Commands

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Verify In Browser

- dashboard loads
- signals table appears
- sentiment panel appears
- benchmark charts appear
- trade lifecycle appears
- screenshots match README

### Acceptance Criteria

- App runs locally.
- No startup exceptions.
- Dashboard is not blank.
- README screenshots are current.

## Phase 17: Commit And Push

Purpose: publish the final submission state.

### Commands

```bash
git status
git add .
git commit -m "Finalize hackathon task plan and submission workflow"
git push
```

### Acceptance Criteria

- Git status is clean after push.
- GitHub repo shows the latest commit.
- README renders correctly.

## Phase 18: Demo Script

Purpose: present confidently in a short judge walkthrough.

### 60-Second Demo Flow

1. "This is an AMD GPU-accelerated AI signal pipeline, not a trading bot."
2. Show top metrics: device, latency, throughput, speedup.
3. Show live dummy signals.
4. Show headline sentiment and distribution chart.
5. Show performance chart for CPU vs GPU batches.
6. Show fake trade lifecycle and P&L.
7. End with safety: no broker, no real strategy, no private data.

### 90-Second Demo Flow

Use the 60-second flow, plus:

- mention DistilBERT open-weight fine-tuning
- mention synthetic balanced dataset
- mention ROCm/PyTorch compatibility
- mention screenshot-backed README

### Acceptance Criteria

- Judges understand the project quickly.
- GPU acceleration is visible.
- Safety boundary is credible.

## Phase 19: Optional Upgrades

Only do these if the core submission is already clean.

### Optional Improvements

- Add `scripts/validate_dataset.py`.
- Add a small benchmark CSV export.
- Add a demo GIF.
- Add hosted Streamlit deployment.
- Add real AMD GPU benchmark numbers to README.
- Add a small architecture image.

### Avoid

- complex trading logic
- real execution
- adding many dependencies
- training a large model
- overbuilding the UI

## Final Submission Checklist

Before sharing the repo, confirm:

- [ ] Repo is separate from the actual trading system.
- [ ] `pip install -r requirements.txt` works.
- [ ] `streamlit run app.py` works.
- [ ] README opens strong.
- [ ] README screenshots render.
- [ ] Dashboard overview screenshot exists.
- [ ] GPU benchmark screenshot exists.
- [ ] Sentiment panel screenshot exists.
- [ ] No `.env` files.
- [ ] No API keys.
- [ ] No private prompts.
- [ ] No OpenEnv/RL code.
- [ ] No broker code.
- [ ] No real strategy logic.
- [ ] No real capital allocation or risk formula.
- [ ] Dummy signal logic is clearly non-realistic.
- [ ] Sentiment model uses open-weight base model.
- [ ] GPU benchmark is visible.
- [ ] GitHub description is set.
- [ ] GitHub topics are set.
- [ ] Final commit is pushed.

## Owner Checklist

Use this section during the hackathon.

| Area | Owner | Status | Notes |
|---|---|---|---|
| Repo separation | TBD | TODO | Move or clone outside production workspace |
| Data generation | TBD | TODO | Synthetic only |
| Label review | TBD | TODO | Confirm balanced labels |
| Fine-tuning | TBD | TODO | DistilBERT, 3 epochs |
| Dashboard QA | TBD | TODO | Run locally |
| AMD benchmark | TBD | TODO | Capture real ROCm numbers if possible |
| Screenshots | TBD | TODO | Place in `assets/screenshots/` |
| README final | TBD | TODO | Confirm images render |
| Safety scan | TBD | TODO | No sensitive logic |
| Submission | TBD | TODO | Push final commit |

## Recommended Timeline

### Day 1

- Separate repo.
- Run app locally.
- Validate structure.
- Generate synthetic data.
- Review labels.

### Day 2

- Fine-tune sentiment model.
- Verify inference.
- Improve dashboard.
- Run CPU benchmark.

### Day 3

- Run AMD GPU benchmark.
- Capture screenshots.
- Update README.
- Record short demo video.

### Final Hours

- Run safety scan.
- Commit and push.
- Confirm GitHub metadata.
- Submit repo link.

