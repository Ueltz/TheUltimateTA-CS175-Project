import pandas as pd
"""
Checking the distrubution of scores in the ASAP 2.0 dataset, which has a different score range than ASAP 1.0. This is important for understanding how to evaluate the model's performance on this dataset and to ensure that the model is trained to predict scores within the correct range.    
"""
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