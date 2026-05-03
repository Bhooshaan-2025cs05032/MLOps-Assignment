import sys
import os

# Ensure the project root is on sys.path so that `import api.xxx` and
# `import src.xxx` resolve correctly when pytest is invoked from any directory.
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
