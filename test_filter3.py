import sys
sys.path.insert(0, '.')

import importlib
import ml.pipeline
importlib.reload(ml.pipeline)

from ml.pipeline import extract_text, chunk_text, is_financial_chunk, filter_financial_chunks, FIN_STMT_TABLE_KEYWORDS, FIN_STMT_NEGATIVE_KEYWORDS

text = extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = chunk_text(text)

print("Testing is_financial_chunk directly:")
for i in range(8):
    result = is_financial_chunk(chunks[i])
    chunk_lower = chunks[i].lower()
    has_jan26 = 'jan 26, 2025' in chunk_lower
    has_jan28 = 'jan 28, 2024' in chunk_lower
    neg_matches = [kw for kw in FIN_STMT_NEGATIVE_KEYWORDS if kw.lower() in chunk_lower]
    table_matches = [kw for kw in FIN_STMT_TABLE_KEYWORDS if kw.lower() in chunk_lower]
    print('Chunk {}: is_fin={}, jan26={}, jan28={}, neg={}, table={}'.format(
        i, result, has_jan26, has_jan28, len(neg_matches), len(table_matches)))

print()
print("Now calling filter_financial_chunks:")
filtered = filter_financial_chunks(chunks)
print('Filtered count: {}'.format(len(filtered)))
for i, c in enumerate(filtered):
    print('  Filtered chunk {}: {} chars'.format(i, len(c)))