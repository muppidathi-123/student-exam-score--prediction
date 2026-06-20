"""
src package
============
This package contains the core, reusable logic for the
Student Exam Score Prediction project:
 
- data_preprocessing.py : cleaning, encoding, splitting, scaling
- train_model.py        : model training entry point
- evaluate_model.py     : metric computation and reporting
- predict.py            : reusable prediction pipeline (used by app.py)
 
Keeping this logic in src/ (instead of only in notebooks) means it can be
imported and reused by the Streamlit app, future scripts, or tests —
without copy-pasting code or re-running a notebook.
"""
 
__version__ = "1.0.0"
 