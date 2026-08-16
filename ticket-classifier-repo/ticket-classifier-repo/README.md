# Support Ticket Auto-Categorization

A small ML pipeline that reads support ticket text (subject + body) and predicts its
category — **Billing**, **Technical**, **HR**, or **General** — using TF-IDF features
and a classic text classifier (Naive Bayes / Logistic Regression).

Built as a technical assessment for the Fobes Skill Itech AI/ML Internship Program.

## What it does

1. **Preprocessing** — lowercases, strips URLs/punctuation/numbers from ticket text.
2. **Feature representation** — TF-IDF vectorization (unigrams, `sublinear_tf`).
3. **Model training** — trains both Multinomial Naive Bayes and Logistic Regression,
   compares held-out accuracy, and selects the better model.
4. **Evaluation** — accuracy, per-class precision/recall/F1, and a confusion matrix.
5. **Real-time prediction** — `predict_ticket(subject, body)` classifies a single new
   ticket on demand.
6. **Confidence score** — returns a probability (%) alongside the predicted category.
7. **Human-review fallback** — predictions below 60% confidence are routed to
   `"Needs Human Review"` instead of being auto-assigned.
8. **Priority tagging** — a keyword rule (`urgent`, `down`, `crash`, `broken`, etc.)
   flags tickets as `Urgent` / `Normal`, independent of the category model.
9. **Mini CLI demo** — type a ticket in and get an instant classification.

## Results

- Dataset: 80 hand-labeled tickets (20 per category).
- Best model: **Logistic Regression** over TF-IDF, ~69% test accuracy
  (vs. ~56% for Naive Bayes on the same split).
- Tested on 5 unseen, hand-written sample tickets — all correctly classified with
  69–83% confidence.

See `ticket_classifier.ipynb` for the full pipeline, evaluation output, confusion
matrix, and a reflection note on what more data/time would improve.

## Approach summary

Used TF-IDF (unigrams, sublinear scaling) with Logistic Regression, chosen over Naive
Bayes after comparing test accuracy — LR's discriminative boundary and better-calibrated
probabilities suited the confidence-score requirement. Edge cases are handled via a 60%
confidence threshold that routes uncertain or empty tickets to "Needs Human Review"
rather than forcing a category, plus an independent keyword-based urgency tag layered
on top of the classifier.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook ticket_classifier.ipynb
```

Run all cells top to bottom. To try the live CLI demo, uncomment `run_cli_demo()` in
the last code cell and run it interactively.

## Project structure

```
.
├── ticket_classifier.ipynb   # full pipeline: data, training, eval, demo
├── requirements.txt
└── README.md
```
