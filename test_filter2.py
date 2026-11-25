import sys
sys.path.insert(0, '.')

import importlib
import ml.pipeline
importlib.reload(ml.pipeline)

from ml.pipeline import extract_text, chunk_text, is_financial_chunk, filter_financial_chunks

text = extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = chunk_text(text)

for i in range(8):
    result = is_financial_chunk(chunks[i])
    print('Chunk {}: is_financial={}'.format(i, result))

print()
filtered = filter_financial_chunks(chunks)
print('Filtered count: {}'.format(len(filtered)))
for i, c in enumerate(filtered):
    print('  Filtered chunk {}: {} chars'.format(i, len(c)))