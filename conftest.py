"""
conftest.py · 测试根路径注入，保证 `import core / data / ...` 可用。
作者：晨星
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
