import sys
sys.path.insert(0, '.')

import importlib
import ml.pipeline
importlib.reload(ml.pipeline)

# Check the actual function objects
from ml.pipeline import is_financial_chunk, filter_financial_chunks

print("is_financial_chunk:", is_financial_chunk)
print("filter_financial_chunks:", filter_financial_chunks)
print("filter_financial_chunks code:", filter_financial_chunks.__code__)

# Check closure
print("filter_financial_chunks closure:", filter_financial_chunks.__closure__)

# Test manually
text = ml.pipeline.extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = ml.pipeline.chunk_text(text)

print("Direct call to is_financial_chunk:")
for i in range(8):
    result = is_financial_chunk(chunks[i])
    print('Chunk {}: is_fin={}'.format(i, result))

print()
print("List comprehension with is_financial_chunk:")
filtered = [c for c in chunks if is_financial_chunk(c)]
print('Filtered count: {}'.format(len(filtered)))

print()
print("Using filter_financial_chunks function:")
filtered2 = filter_financial_chunks(chunks)
print('Filtered count: {}'.format(len(filtered2)))