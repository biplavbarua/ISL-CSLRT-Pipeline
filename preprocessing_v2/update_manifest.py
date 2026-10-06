import csv
import glob, numpy as np, itertools

files = glob.glob('../preprocessed_v4/keypoints_raw/*/*/confidence.npy')
long_gap_set = set()

for f in files:
    conf = np.load(f)
    if conf.shape[0] == 0: continue
    det = conf[:, 33:75].max(axis=1) > 0.5
    for is_det, run in itertools.groupby(det):
        if not is_det and sum(1 for _ in run) > 8:
            parts = f.split('/')
            sentence = parts[-3]
            signer = parts[-2]
            long_gap_set.add(f"{sentence}/{signer}")
            break

index_path = '../preprocessed_v4/logs/segment_manifest.csv'
with open(index_path, 'r', encoding='utf-8') as f:
    reader = list(csv.DictReader(f))

fieldnames = list(reader[0].keys())
if 'has_long_gap' not in fieldnames:
    fieldnames.append('has_long_gap')

for row in reader:
    clip_id = f"{row['sentence_slug']}/{row['signer_id']}"
    row['has_long_gap'] = 'True' if clip_id in long_gap_set else 'False'

with open(index_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(reader)
    
print("Updated segment_manifest.csv with has_long_gap column.")
