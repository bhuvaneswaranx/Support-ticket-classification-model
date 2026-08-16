# Support-ticket-classification-model
used Logistic Regression to classify tickets because it gave better accuracy than Naive Bayes when I tested both. If the model isn't at least 60% confident about a category, it flags the ticket as "Needs Human Review" instead of guessing wrong. I also added a simple keyword check to mark tickets as urgent.
