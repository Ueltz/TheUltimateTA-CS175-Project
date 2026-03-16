import pandas as pd

# Load the ASAP2_train_sourcetexts.csv file
df = pd.read_csv('data/ASAP2_train_sourcetexts.csv')

# Get the 'score' column
scores = df['score']

# Display the score range
min_score = scores.min()
max_score = scores.max()
print(f"Score range for ASAP 2.0: {min_score} to {max_score}")

# Print how many of each score
print("\nCount of each score:")
score_counts = scores.value_counts().sort_index()
for score, count in score_counts.items():
    print(f"Score {score}: {count}")