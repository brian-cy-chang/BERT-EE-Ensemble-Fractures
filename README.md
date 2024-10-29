# BERT-EE-Ensemble-Fractures
This framework includes ensemble models leveraging [BERT-EE](https://github.com/wilsonlau-uw/BERT-EE) to consolidate fracture events from clinical notes. There are two ensemble models: 
1) ensemble majority voting on discrete fracture events 
2) ensemble averaging on the last hidden state output

## Installation
The BERT-EE models require `python>=3.6.13`, `torch>=1.6.0` and `transformers>=4.4.0`. Please follow the instructions below to install the environment.

```bash
git clone https://github.com/brian-cy-chang/BERT-EE_Ensemble-Fractures.git

cd BERT-EE_Ensemble-Fractures & pip install -r requirements.txt
```

## Getting Started
### Datasets
1. Each clinical note must be an individual .txt file.
2. For training, each .txt file must have a corresponding annotation (.ann) file according to the [brat rapid annotation tool](https://brat.nlplab.org/).
3. In `user_params.py`, the parameter **PATIENT_ID** must be a path to a .csv file with the data format below.

| subject_id                    | NoteID         | 
|-------------------------------|----------------|
| *string*                      | *string*       |

Notes:

1. Please refer to the [BRAT rapid annotation tool](https://brat.nlplab.org/) documentation as needed for annotating notes. 

### BERT-EE Models & Ensemble Models
Three fine-tuned BERT models will independently be run on the notes: *bert-base-cased*, *ClinicalBERT*, and *BioBERT* found in `BERT-EE/bert_models`.
Once inferencing is completed, one of two ensemble models will be run:

1. Ensemble majority voting on discrete events
2. Ensemble averaging of the last hidden state outputs

Please specify the ensemble model to run in `user_params.py`

```python
python main.py
```

## Acknowledgments
The annotation of the training corpus used a [Docker container of BRAT](https://github.com/ddevaraj/docker-brat).