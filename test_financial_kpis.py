import sys
sys.path.insert(0, '.')

import importlib
import ml.pipeline
importlib.reload(ml.pipeline)

from ml.pipeline import extract_text, chunk_text, filter_financial_chunks, extract_kpis, merge_kpi_results

text = ml.pipeline.extract_text(r'backend\data\757b344ab62ba282.pdf')
chunks = ml.pipeline.chunk_text(text)
fin_chunks = filter_financial_chunks(chunks)

print('Financial chunks:', len(fin_chunks))

kpi_list = []
for i, c in enumerate(fin_chunks):
    kpis = extract_kpis(c)
    print('Financial chunk {}: {}'.format(i, kpis))
    kpi_list.append(kpis)

merged = merge_kpi_results(kpi_list)
print('\nMerged KPIs:', merged)