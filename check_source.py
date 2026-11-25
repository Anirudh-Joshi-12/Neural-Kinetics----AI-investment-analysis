import sys
sys.path.insert(0, '.')

import importlib
import ml.pipeline
importlib.reload(ml.pipeline)

import inspect
print(inspect.getsource(ml.pipeline.filter_financial_chunks))