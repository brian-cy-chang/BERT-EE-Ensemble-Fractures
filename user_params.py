"""
This module sets user-defined parameters
for the ensemble methods of the BERT-EE model predictions.
"""
import os
from pathlib import Path

""" 
BERT-EE data directories 
Relative paths, do not modify
"""
ROOT_DIR = Path(__file__).parent
BERT_EE_DIR = os.path.join(ROOT_DIR, "BERT-EE")
# BERT_ENSEMBLE_DIR = os.path.join(ROOT_DIR, "BERT_Ensemble")
BERT_MODEL_DIR = os.path.join(BERT_EE_DIR, "bert_models")
RESULTS_DIR = os.path.join(BERT_EE_DIR, "results")

# path of clinical notes to run prediction
NOTES_DIR = ""

# BERT-EE results directory with NER/RE predictions and hidden state files
BERT_BASE_DIR = os.path.join(RESULTS_DIR, "bert-base-cased")
CLINICALBERT_DIR = os.path.join(RESULTS_DIR, "ClinicalBERT")
BIOBERT_DIR = os.path.join(RESULTS_DIR, "BioBERT")

LOGITS = [BERT_BASE_DIR, CLINICALBERT_DIR, BIOBERT_DIR]

# file path of patient/subject ID to note ID, must be a csv file
# example = "<DATA_DIR>/subject_note_id.csv"
PATIENT_ID = ""

"""bert model output directories"""
SAVE_DIR = "" #desired directory to save final patient-level events
SAVE_NAME = "" #desired filename of the final patient-level events

"""
ensemble method
['discrete, 'hiddenState']
"""
# ensemble majority voting with discrete events or ensemble averaging with last hidden state embeddings
ENSEMBLE_METHOD = 'discrete'
# if ENSEMBLE_METHOD = 'discrete' set span tolerance for grouping events for ensemble majority voting
SPAN_TOLERANCE = 5
# ENSEMBLE_METHOD = 'hiddenState'

"""
if ENSEMBLE_METHOD = 'hiddenState'
['cls', 'average']
"""
HIDDENSTATE_MODE = 'cls' # only leverage CLS embeddings
# HIDDENSTATE_MODE = 'average' # leverage entire last hidden state output
