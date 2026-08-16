# Support Ticket Auto-Categorization

A small ML pipeline that reads support ticket text (subject + body) and predicts its
category — **Billing**, **Technical**, **HR**, or **General** — using TF-IDF features
and a classic text classifier (Naive Bayes / Logistic Regression).

Built as a technical assessment for the Fobes Skill Itech AI/ML Internship Program.

## How it works (plain-English explanation)

If you're new to machine learning, here's the idea without the jargon:

**The problem.** A support team gets emails/tickets all day — "my invoice is wrong",
"the app crashed", "I need sick leave", etc. Someone has to read each one and decide
which team it goes to: Billing, Technical, HR, or General. Doing that by hand for every
ticket is slow. This project automates that first sorting step.

**The idea, step by step:**

1. **Teach the computer with examples.** You can't just tell a computer "sort these
   emails" — it has to learn from examples first, the same way you'd learn to sort mail
   after seeing a few examples of each type. So the first step is a small dataset: 80
   example tickets, 20 for each category, already correctly labeled by a human (that's
   you, in this case).

2. **Turn words into numbers.** Computers don't understand English, only numbers. So
   every ticket's text gets converted into a list of numbers that represent which
   words appear and how important each word is. This technique is called **TF-IDF**.
   In simple terms: common, boring words like "please" or "thanks" that show up in
   *every* category get a low score, while distinctive words like "invoice", "crashed",
   or "salary" that strongly hint at one category get a high score. This turns raw text
   into something a model can actually learn patterns from.

3. **Train a "brain" (the model).** With the tickets now turned into numbers, you feed
   them to a classifier — think of it as a pattern-matching brain. Two were tried here:
   Naive Bayes and Logistic Regression. Both look at the numbers and learn, "tickets
   with these word patterns tend to be Billing, these tend to be Technical," and so on.
   Logistic Regression turned out to guess correctly more often, so that's the one kept.

4. **Test it on tickets it's never seen.** To check if the model actually learned
   something useful (rather than just memorizing), some tickets are held back during
   training and only shown afterward as a test. Getting ~69% of those right tells you
   roughly how well it'll do on brand-new tickets in real life.

5. **Use it on a live, brand-new ticket.** The whole point is a function you can call
   with any new ticket text, and it instantly returns a predicted category — this is
   the part that would actually sit in front of a real ticket queue.

6. **Don't let it guess blindly.** Every prediction comes with a **confidence score**
   — basically "how sure am I?" as a percentage. If the model is less than 60% sure,
   instead of forcing a possibly-wrong guess, it says **"Needs Human Review"** so a
   real person checks it. This mirrors how real triage systems work — automate the
   easy, obvious cases and hand off the unclear ones to a human.

7. **Bonus: flag urgent tickets.** Separately from the category guessing, a simple
   keyword check (looking for words like "urgent", "down", "broken", "crash") tags a
   ticket as Urgent or Normal — so time-sensitive issues stand out regardless of which
   category they land in.

In short: **feed the computer labeled examples → turn text into numbers → train a
model to recognize patterns → test it → use it on new tickets → fall back to a human
when unsure.** That's the whole pipeline.

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
