import sys
sys.path.insert(0, '.')

import importlib
import ml.pipeline
importlib.reload(ml.pipeline)

from ml.pipeline import extract_text, chunk_text, is_financial_chunk, extract_kpis, merge_kpi_results

text = ml.pipeline.extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = ml.pipeline.chunk_text(text)

# The chunks that were identified as financial (from the test)
good_indices = [153, 156, 159, 160, 170, 173, 179]

print('Testing specific financial chunks:')
kpi_list = []
for idx in good_indices:
    if idx < len(chunks):
        kpis = extract_kpis(chunks[idx])
        print('Chunk {}: {}'.format(idx, kpis))
        kpi_list.append(kpis)

merged = merge_kpi_results(kpi_list)
print('\nMerged KPIs:', merged)