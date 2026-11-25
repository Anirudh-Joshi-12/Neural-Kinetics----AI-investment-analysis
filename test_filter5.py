import sys
sys.path.insert(0, '.')

import importlib
import ml.pipeline
importlib.reload(ml.pipeline)

from ml.pipeline import is_financial_chunk, extract_text, chunk_text

text = ml.pipeline.extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = ml.pipeline.chunk_text(text)

print("Total chunks:", len(chunks))

# Test each chunk with explicit index
for i in range(10):
    chunk = chunks[i]
    result = is_financial_chunk(chunk)
    print('Chunk {}: is_fin={}, id={}'.format(i, is_financial_chunk(chunk), id(chunk)))

print()
print("Now testing list comprehension:")
filtered = [c for c in chunks if is_financial_chunk(c)]
print('Filtered count:', len(filtered))
for i, c in enumerate(filtered):
    print('  Filtered chunk {}: {} chars, id={}'.format(i, len(c), id(c)))

print()
print("Check if chunks list is modified:")
print('Original chunks id:', id(chunks))
print('Filtered chunks id:', id(filtered))