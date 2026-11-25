import sys
sys.path.insert(0, '.')

from ml.pipeline import extract_text, chunk_text

text = extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = chunk_text(text)

for i in range(4):
    chunk = chunks[i].lower()
    has_jan26 = 'jan 26, 2025' in chunk
    has_jan28 = 'jan 28, 2024' in chunk
    print('Chunk {}: jan26={}, jan28={}'.format(i, has_jan26, has_jan28))
    if has_jan26 or has_jan28:
        for kw in ['jan 26, 2025', 'jan 28, 2024']:
            pos = chunk.find(kw)
            if pos >= 0:
                start = max(0, pos - 100)
                end = min(len(chunk), pos + 100)
                print('  Found "{}" at pos {}: ...{}...'.format(kw, pos, chunk[start:end]))
        print()