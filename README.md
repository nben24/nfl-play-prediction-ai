# NFL Play Predictor

A command-line tool that predicts whether an NFL offense will run or pass, then uses a local LLM to explain the prediction in plain English.

Built to showcase a full ML + LLM pipeline on real football data.

## How it works

1. A **Random Forest classifier** trained on NFL play-by-play data predicts run vs. pass given the game situation
2. A **rule-based reasoning engine** translates the situation into factual bullet points (down & distance, field position, score context)
3. A **local LLM** (via [Ollama](https://ollama.com)) paraphrases those bullet points into natural analyst commentary

This design keeps the LLM focused on language — not reasoning — so explanations stay accurate and grounded.

## Tech stack

- **ML model**: `scikit-learn` RandomForestClassifier
- **Data**: NFL play-by-play via [nflverse](https://github.com/nflverse/nflverse-data)
- **LLM**: `llama3.2:3b` running locally via Ollama
- **LLM API**: `openai` SDK pointed at `localhost:11434`

## Setup

### 1. Install Ollama and pull the model

Download Ollama from [ollama.com](https://ollama.com), then:

```bash
ollama pull llama3.2:3b
```

### 2. Clone and install dependencies

```bash
git clone https://github.com/your-username/nfl-play-prediction-ai.git
cd nfl-play-prediction-ai
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Download the data

```bash
cd data
python get_data.py
cd ..
```

This fetches the 2025 NFL play-by-play CSV from nflverse.

### 4. Train the model

```bash
python src/train_model.py
```

This trains the classifier and saves it to `models/model.pkl`.

### 5. Run the app

```bash
python src/app.py
```

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
On 4th and 3 at the goal line with only 24 seconds left and trailing by 2,
the offense has no choice but to attack through the air. A touchdown here wins
the game, making this a clear passing situation despite the short yardage.
```

## Project structure

```
src/
  app.py            - CLI entry point and input validation
  predict.py        - Loads model, runs inference
  llm_explainer.py  - Builds reasoning bullets + calls LLM
  train_model.py    - Trains and saves the RandomForest model
  preprocess.py     - Cleans and prepares training data
  load_data.py      - Loads the raw CSV
data/
  get_data.py       - Downloads NFL play-by-play data from nflverse
notebooks/
  eda.ipynb         - Exploratory data analysis
tests/
  test_llm_explainer.py - Unit tests for the reasoning engine
```

## Running tests

```bash
pytest tests/
```
