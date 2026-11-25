import sys
sys.path.insert(0, '.')

from ml.pipeline import extract_text, chunk_text, is_financial_chunk, FIN_STMT_TABLE_KEYWORDS, FIN_STMT_NEGATIVE_KEYWORDS

text = extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = chunk_text(text)

for i in range(4):
    chunk = chunks[i]
    chunk_lower = chunk.lower()
    
    neg_matches = [kw for kw in FIN_STMT_NEGATIVE_KEYWORDS if kw.lower() in chunk_lower]
    table_matches = [kw for kw in FIN_STMT_TABLE_KEYWORDS if kw.lower() in chunk_lower]
    
    has_jan26 = 'jan 26, 2025' in chunk_lower
    has_jan28 = 'jan 28, 2024' in chunk_lower
    
    result = is_financial_chunk(chunks[i])
    
    print('Chunk {}: result={}, jan26={}, jan28={}, neg_matches={}, table_matches={}'.format(
        i, result, has_jan26, has_jan28, len(neg_matches), len(table_matches)))
    if neg_matches:
        print('  Negative: {}'.format(neg_matches[:3]))
    if table_matches:
        print('  Table: {}'.format(table_matches[:5]))