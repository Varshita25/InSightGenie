
import sys
import os

try:
    import pandas as pd
    import streamlit as st
    from core.loader import load_table
    from core.profiler import basic_profile
    from core.hypothesis import generate_hypotheses
    from core.insights import generate_insights
    print("SUCCESS: Imports are working")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
