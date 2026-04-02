# NFL Play Predictor

A command-line tool that predicts whether an NFL offense will run or pass on a given play, then generates a grounded natural-language explanation of why.

The core design challenge: small local LLMs hallucinate badly when asked to reason about domain-specific situations. This project solves that by keeping the LLM out of the reasoning entirely — a deterministic rules engine identifies the real factors, and the LLM's only job is to turn those facts into readable commentary.

---

## Architecture

```
User input (game situation)
        │
        ▼
┌───────────────────┐
│  Random Forest    │  ← trained on real NFL play-by-play data
│  Classifier       │    predicts run or pass + probability
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Rules Engine     │  ← deterministic Python logic
│  (_build_reasoning│    converts situation into factual bullet points
│   in explainer.py)│    (down/distance, field position, score/time)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Local LLM        │  ← llama3.2:3b via Ollama
│  (paraphrase only)│    rewrites bullet points as analyst commentary
└────────┬──────────┘
         │
         ▼
  Natural-language explanation
```

**Why this design?** When given the full game situation and asked to reason freely, a small LLM will invent facts — wrong quarter, wrong field position, false urgency. By pre-computing all factual context in Python and only asking the LLM to rephrase, the explanations stay accurate regardless of model size.

---

## Features

| Feature | Description |
|---|---|
| Run/pass prediction | Random Forest trained on 6 situational features |
| Probability output | Returns calibrated run % and pass % |
| Grounded explanation | Rules-based reasoning prevents LLM hallucination |
| Local LLM | Runs entirely offline via Ollama — no API key needed |
| Input validation | Rejects out-of-range game situations before inference |
| Saved metrics | Training writes `models/metrics.json` for reproducibility |

---

## Model

**Algorithm:** Random Forest Classifier (scikit-learn)

**Features:**

| Feature | Description |
|---|---|
| `down` | Current down (1–4) |
| `ydstogo` | Yards needed for a first down |
| `yardline_100` | Distance from the opponent's end zone (1–99) |
| `qtr` | Quarter (1–4) |
| `score_differential` | Offense score minus defense score |
| `game_seconds_remaining` | Seconds left in the game |

**Training data:** 2025 NFL play-by-play data from [nflverse](https://github.com/nflverse/nflverse-data), filtered to run and pass plays only.

**Performance:** After training, metrics are saved to `models/metrics.json`. Typical results on a held-out 20% test split:

```
              precision    recall  f1-score
         run       0.xx      0.xx      0.xx
        pass       0.xx      0.xx      0.xx
    accuracy                           0.xx
```

> Run `python src/train.py` to generate real numbers for your dataset.

---

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Pull the LLM

```bash
ollama pull llama3.2:3b
```

### 2. Clone and install

```bash
git clone https://github.com/your-username/nfl-play-prediction-ai.git
cd nfl-play-prediction-ai
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download the data

```bash
python data/get_data.py          # downloads 2025 season by default
python data/get_data.py 2024     # or specify a season
```

### 4. Train the model

```bash
python src/train.py
```

Prints a classification report, feature importances, and saves:
- `models/model.pkl` — trained classifier
- `models/metrics.json` — evaluation metrics

### 5. Run

```bash
python src/app.py
```

---

## Example

```
=== NFL Play Predictor ===

Quarter (1-4): 4
Down (1-4): 4
Yards to go: 3
Yard line (distance from opponent's end zone, 1-99): 3
Score differential (offense - defense): -2
Seconds remaining in game (max 3600): 24

Prediction: PASS
  Run probability:  23.0%
  Pass probability: 77.0%

Explanation (generating...)
On 4th and 3 at the goal line, trailing by 2 with only 24 seconds
remaining, this is a game-winning touchdown attempt — the offense must
score here to win, making a pass the only viable call.
```

---

## Running tests

```bash
pytest
```

Tests cover the rules engine (`_build_reasoning`) across all situational categories: down and distance, field position, score/time logic, and edge cases. The LLM layer is intentionally not tested — it's non-deterministic and requires a running Ollama instance.

---

## Limitations

- **Binary prediction only** — run vs. pass. Does not predict play type within those categories (screen, RPO, draw, etc.)
- **No personnel or formation data** — features are situational only; the model has no visibility into who is on the field
- **Play-call ≠ outcome** — the model predicts what is likely called, not whether it succeeds
- **LLM quality is hardware-dependent** — explanation quality varies with the Ollama model used and whether GPU acceleration is available
- **Single season training** — more seasons of data would improve generalization

---

## Future improvements

- Add more seasons of training data
- Evaluate calibration of probability outputs
- Explore gradient boosting (XGBoost/LightGBM) as an alternative classifier
- Add support for personnel groupings and formation as features
- Expose as a simple web API

---

## Repo structure

```
nfl-play-prediction-ai/
├── src/
│   ├── app.py          # CLI entry point and input validation
│   ├── predict.py      # Model loading and inference
│   ├── explainer.py    # Rules engine + LLM explanation layer
│   ├── train.py        # Training script
│   └── preprocess.py   # Data cleaning and feature preparation
├── data/
│   └── get_data.py     # Downloads NFL play-by-play data from nflverse
├── models/
│   ├── model.pkl       # Trained model (generated, not tracked)
│   └── metrics.json    # Evaluation metrics (generated, not tracked)
├── notebooks/
│   └── eda.ipynb       # Exploratory data analysis
├── tests/
│   └── test_explainer.py  # Unit tests for the rules engine
├── pyproject.toml      # Pytest config
└── requirements.txt
```
