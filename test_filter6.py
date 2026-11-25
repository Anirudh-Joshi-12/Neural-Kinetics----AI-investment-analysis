import sys
sys.path.insert(0, '.')

import importlib
import ml.pipeline
importlib.reload(ml.pipeline)

from ml.pipeline import is_financial_chunk, extract_text, chunk_text

text = ml.pipeline.extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = ml.pipeline.chunk_text(text)

print("Total chunks:", len(chunks))

# Test with explicit loop and print
print("\nExplicit loop:")
filtered = []
for i, c in enumerate(chunks):
    result = is_financial_chunk(c)
    if result:
        filtered.append(c)
        print('  Added chunk {}: {} chars'.format(i, len(c)))
    else:
        print('  Rejected chunk {}: {} chars'.format(i, len(c)))

print('\nFiltered count:', len(filtered))
for i, c in enumerate(filtered):
    print('  Filtered chunk {}: {} chars'.format(i, len(c)))

# Also test the list comprehension separately
print("\nList comprehension:")
filtered2 = [c for c in chunks if is_financial_chunk(c)]
print('Filtered count:', len(filtered2))
for i, c in enumerate(filtered2):
    print('  Filtered chunk {}: {} chars'.format(i, len(c)))