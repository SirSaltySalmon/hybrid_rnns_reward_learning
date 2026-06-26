# Experiment 1 — How far back does memory reach? (Binh Khang)
<li>Take the existing Memory-ANN training code and add a single setting N that limits how far back the model is allowed to look — anything beyond N trials ago gets ignored.
<li>Train a separate version of Memory-ANN for each value of N ∈ {3, 5, 8, 12, 20, 35, 50, 75, 100, 150}, keeping everything else identical to the original paper. Train each version 5 times with different random starting points.
<li>For each trained version, measure how accurately it predicts held-out human choices.
<li>Plot prediction accuracy on the y-axis and memory window size N on the x-axis. Find the elbow — the point where making the window shorter starts clearly hurting accuracy. That elbow is our estimate of how far back humans actually remember.
