
import pandas as pd
import numpy as np

# Load data
X = pd.read_csv("data/processed/X_dataset.csv")
Y = pd.read_csv("data/processed/Y_labels.csv")

# Create combination IDs
Y_copy = Y.copy()
Y_copy['combination_id'] = Y_copy.groupby(['NO_conc', 'NO2_conc', 'SO2_conc']).ngroup()

# Split combinations (no overlap)
unique_combinations = Y_copy['combination_id'].unique()
np.random.seed(42)
shuffled = np.random.permutation(unique_combinations)

n_test = int(len(unique_combinations) * 0.2)
n_val = int(len(unique_combinations) * 0.1)

test_combinations = shuffled[:n_test]
val_combinations = shuffled[n_test:n_test+n_val]
train_combinations = shuffled[n_test+n_val:]

# Create splits
train_mask = Y_copy['combination_id'].isin(train_combinations)
val_mask = Y_copy['combination_id'].isin(val_combinations)
test_mask = Y_copy['combination_id'].isin(test_combinations)

# Save splits
X[train_mask].reset_index(drop=True).to_csv("data/processed/X_train.csv", index=False)
Y[train_mask].reset_index(drop=True).to_csv("data/processed/Y_train.csv", index=False)
X[val_mask].reset_index(drop=True).to_csv("data/processed/X_val.csv", index=False)
Y[val_mask].reset_index(drop=True).to_csv("data/processed/Y_val.csv", index=False)
X[test_mask].reset_index(drop=True).to_csv("data/processed/X_test.csv", index=False)
Y[test_mask].reset_index(drop=True).to_csv("data/processed/Y_test.csv", index=False)

print(f"Split completed: Train={sum(train_mask)}, Val={sum(val_mask)}, Test={sum(test_mask)}")
