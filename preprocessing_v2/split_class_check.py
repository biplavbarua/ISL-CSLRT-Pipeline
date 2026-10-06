import os, csv
from collections import defaultdict

splits_dir = '../preprocessed_v4/splits'
manifest_path = '../preprocessed_v4/logs/segment_manifest.csv'

# Get all 97 valid sentence classes
with open(manifest_path, 'r', encoding='utf-8') as f:
    manifest = list(csv.DictReader(f))
    
all_sentences = set(r['sentence_slug'] for r in manifest)
# wait, 'he_is_on_the_way' lost signer 3. But it still has other signers? Yes, only 3 was excluded entirely.
# Let's see if any sentence class has exactly 0 clips in the entire filtered manifest.
# the manifest contains all 663 clips. Excluded he_is_on_the_way/3 makes it 662.
# So all 97 sentences still have examples.

print(f"Total valid sentence classes in corpus: {len(all_sentences)}\n")

files = [f for f in os.listdir(splits_dir) if f.endswith('.csv')]
files.sort()

# Data structure to hold sizes
sizes = {}
missing = defaultdict(lambda: {'val': [], 'test': []})

for split_file in files:
    with open(os.path.join(splits_dir, split_file), 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    
    sizes[split_file] = len(rows)
    
    # We only care about val and test sets for missing classes
    if '_train' in split_file: continue
    
    # What fold is this?
    if split_file.startswith('standard_'):
        fold = 'standard'
        split_type = split_file.replace('standard_', '').replace('.csv', '')
    else:
        # loso_fold_1_test.csv
        parts = split_file.replace('.csv', '').split('_')
        fold = f"fold_{parts[2]}"
        split_type = parts[3]
        
    sentences_present = set(r['sentence_slug'] for r in rows)
    missing_classes = all_sentences - sentences_present
    
    missing[fold][split_type] = list(missing_classes)


print("--- SPLIT SIZES ---")
# Standard
print(f"Standard Split: Train={sizes.get('standard_train.csv', 0)}, Val={sizes.get('standard_val.csv', 0)}, Test={sizes.get('standard_test.csv', 0)}")
# LOSO
for i in range(1, 8):
    t_key = f"loso_fold_{i}_train.csv"
    v_key = f"loso_fold_{i}_val.csv"
    te_key = f"loso_fold_{i}_test.csv"
    print(f"LOSO Fold {i}: Train={sizes.get(t_key, 0)}, Val={sizes.get(v_key, 0)}, Test={sizes.get(te_key, 0)}")

print("\n--- ZERO-COVERAGE CLASSES IN VAL/TEST ---")
def print_missing(fold_name, data):
    for stype in ['val', 'test']:
        m = data.get(stype, [])
        print(f"  {stype.upper()} ({len(m)} missing classes):")
        for cls in sorted(m):
            print(f"    - {cls}")

print("Standard Split:")
print_missing('standard', missing['standard'])

for i in range(1, 8):
    fold = f"fold_{i}"
    print(f"LOSO Fold {i}:")
    print_missing(fold, missing[fold])

