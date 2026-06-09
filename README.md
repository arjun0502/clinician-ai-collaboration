# Mitigating Anchoring in Clinician-AI Collaboration

This project builds on [StanfordMIMI/clinician-ai-collaboration](https://github.com/StanfordMIMI/clinician-ai-collaboration) and the paper [**"Clinician input steers frontier AI models toward both accurate and harmful decisions"**](https://arxiv.org/abs/2603.14158) (Lopez et al., 2025).

That paper showed that clinician inputs cause LLMs to anchor strongly — expert input improved diagnostic accuracy by +20.4 percentage points on average, but adversarial input degraded it, and models reproduced harmful clinician recommendations at a high rate. This project asks: **can we design prompting strategies that preserve the benefits of clinician input while reducing harmful anchoring?**

---

## Dataset

All data lives in `shared_datasets/`:

| File | Description |
|------|-------------|
| `nejm_case_vignette.yaml` | 61 NEJM clinical case vignettes with case IDs |
| `nejm_case_answers.json` | Ground-truth final diagnoses per case |
| `harm_classes_tbl.csv` | Pre-labeled harmful next steps per case, annotated by severity (None / Mild / Moderate / Severe / Death) |
| `clinician_arguments.xlsx` | Helpful and harmful clinician inputs per case |

Each case is run under **two clinician input variants**:
- **Helpful** — correct differential diagnosis + safe next steps
- **Harmful** — wrong differential + dangerous next steps (drawn from the harm table)

---

## Experimental Conditions

All conditions receive the clinical vignette plus a simulated clinician input. They differ in how the LLM processes that input before producing its final response.

### Anchored (baseline)
The LLM sees the vignette and clinician input and generates a response directly. No additional reasoning steps. This replicates the anchoring behavior documented in the original paper.

### Critique-Clinician
Before generating, the LLM is asked to critically evaluate the clinician's input — identify potential errors, biases, or unsafe recommendations. It then generates its response in light of that critique. The goal is to create distance between the model and the clinician's framing before committing to a differential or next steps.

### Critique-LLM
The LLM first generates an initial response (anchored), then critiques its own response with explicit attention to whether it may have been inappropriately influenced by the clinician input. It then regenerates. This can be run for N rounds (default: 1). The self-critique prompt includes a dedicated section titled "Influence of Clinician Input" to focus the model's attention on anchoring.

### Critique-Combined
Combines both: critique the clinician input first, generate, then run N rounds of LLM self-critique. This is the most compute-intensive condition but targets anchoring at both stages.

---

## Metrics

Each condition × variant combination is run **K=3 times** independently. We report avg / best@K / worst@K for stability.

| Metric | Definition | Direction |
|--------|-----------|-----------|
| **Diagnostic accuracy** | % of cases where the correct final diagnosis appeared in the AI's differential | Higher is better |
| **Harm reproduction rate** | % of pre-labeled harmful next steps that the AI independently recommended | Lower is better |
| **% cases with severe/death harm** | % of cases where AI recommended at least one step rated Severe or Death severity | Lower is better |

---

## Preliminary Results

Results below are from K=3 runs using **Gemini gemini-3.1-flash-lite** for generation and **GPT-4o-mini** as judge (N=61 cases per run).

---

### Helpful Variant

Clinician input is correct: accurate differential + safe next steps.

| Condition | Avg Dx% | Best Dx% | Worst Dx% | Avg Harm% | Avg %Severe/Death |
|-----------|---------|---------|---------|----------|-----------------|
| Anchored | 89.1% | 93.4% | 83.6% | 9.5% | 3.8% |
| Critique-Clinician | 92.3% | 95.1% | 88.5% | 10.6% | 7.6% |
| Critique-LLM | 88.5% | 91.8% | 86.9% | 9.1% | 2.7% |

### Harmful Variant

Clinician input is adversarial: wrong differential + dangerous next steps.

| Condition | Avg Dx% | Best Dx% | Worst Dx% | Avg Harm% | Avg %Severe/Death |
|-----------|---------|---------|---------|----------|-----------------|
| Anchored | 80.3% | 83.6% | 77.0% | 16.3% | 10.9% |
| Critique-Clinician | 79.2% | 82.0% | 75.4% | 13.5% | 5.5% |
| Critique-LLM | 74.3% | 75.4% | 73.8% | 12.1% | 7.7% |

---

### Analysis

**Helpful variant — NEJM cases are easy, but critique strategies trade off differently:**
Across all conditions, baseline diagnostic accuracy is very high (89.1%), suggesting these NEJM cases are not particularly challenging for frontier LLMs even without any debiasing strategy. The two critique approaches diverge here in an interesting way: Critique-Clinician *improves* accuracy (89.1% → 92.3%) — likely because engaging critically with the clinician's reasoning also surfaces useful clinical logic — but harm simultaneously *increases* (9.5% → 10.6%, and severe/death cases nearly double from 3.8% → 7.6%). Critique-LLM shows the opposite pattern: accuracy is essentially flat (89.1% → 88.5%) while harm falls modestly (9.5% → 9.1%, severe/death 3.8% → 2.7%).

**Harmful variant — harm is reduced, but at the cost of diagnostic accuracy:**
Both critique strategies meaningfully reduce harm on adversarial inputs. Critique-Clinician cuts the overall harm rate from 16.3% → 13.5% and halves the severe/death case rate (10.9% → 5.5%). Critique-LLM achieves the lowest harm rate overall (12.1%) with severe/death at 7.7%. However, both conditions come with a real accuracy penalty: Critique-Clinician holds roughly steady (80.3% → 79.2%), but Critique-LLM drops noticeably (80.3% → 74.3%).

**Overall — a fundamental accuracy–harm tradeoff:**
No condition tested was able to simultaneously increase or maintain diagnostic accuracy *and* reduce harm reproduction. This tension holds across both variants: the approaches that most aggressively push back on harmful inputs also suppress useful clinical signal, degrading accuracy. Resolving this tradeoff is the central open problem.

- **Critique-Combined not yet run** — expected to combine the harm reduction of Critique-Clinician with the consistency of Critique-LLM.

---

## Code Organization

```
clinician-ai-collaboration/
├── run.py              # entry point — generation + evaluation pipeline
├── analyze.py          # aggregate results/summary.csv → tables
├── requirements.txt
├── shared_datasets/    # input data (vignettes, answers, harm labels)
├── results/            # outputs (gitignored)
│   ├── raw/            # per-case LLM generation outputs
│   ├── eval/           # per-case evaluation outputs
│   └── summary.csv     # one row per (condition, case, variant, run)
└── src/
    ├── clients.py      # LLMClient: unified async interface for OpenAI / Gemini / Anthropic
    ├── config.py       # model defaults, paths, experiment settings
    ├── data.py         # loads vignettes, answers, harm table
    ├── pipeline.py     # condition runners + LLM-as-judge evaluation
    └── prompts.py      # all prompt templates
```

Results are cached to disk by `(condition, case_id, variant, run)` — re-running skips already-completed cases.

---

## Running Experiments

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set API keys in a `.env` file:
```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

### Provider and model flexibility

Every component is independently configurable. You can run any single condition or all of them, swap the generation/critique model for any supported provider, and swap the evaluation judge for any supported provider — all via CLI flags. Supported providers: `openai`, `gemini`, `anthropic`. Models are specified as `"provider:model-name"` strings.

### Basic usage

```bash
# Default: Gemini generation, GPT-4o-mini evaluation, all conditions, K=3 runs
python run.py

# Quick smoke test (2 cases, 1 run)
python run.py --cases 2 --runs 1

# Single condition only
python run.py --condition anchored

# Use Claude for generation
python run.py --gen-model "anthropic:claude-sonnet-4-6"

# Use GPT-4o as judge
python run.py --eval-model "openai:gpt-4o"

# More LLM self-critique rounds
python run.py --critique-rounds 3

# Generate only (skip evaluation)
python run.py --skip-eval

# Evaluate existing results without regenerating
python run.py --eval-only
```


### Analyzing results

```bash
python analyze.py             # print avg/best/worst tables
python analyze.py --verbose   # also print per-run detail
python analyze.py --csv       # save results/aggregate.csv
```

---

## Future Directions

- **Better case selection / move beyond NEJM**: Many NEJM cases have high baseline accuracy, making anchoring effects harder to measure. A near-term improvement is filtering to the hardest subset — cases with the largest accuracy gap between helpful and harmful variants, where anchoring matters most. Longer term, NEJM cases skew toward rare, diagnosis-first presentations and are likely in training data; more realistic evaluation requires cases drawn from routine clinical workflows where anchoring on a prior clinician's assessment is a common and practically important failure mode.

- **Sequential / interactive diagnosis (Craft MD / NeurIPS approach)**: Instead of presenting all information at once, structure the interaction as an iterative back-and-forth between the clinician and the LLM — the clinician shares findings incrementally, the LLM responds and requests more information, and the differential evolves over multiple turns. This better reflects real clinical workflow and tests whether anchoring behaves differently when the model cannot see the full case upfront.

- **Uncertainty-aware generation**: Explore whether prompting the model to express its uncertainty at each step changes anchoring behavior. When does the model know it has enough information to commit to a diagnosis?

---

## Contact

Arjun Jain — arjun0502@gmail.com