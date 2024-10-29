# BERT-EE-Ensemble-Fractures
This framework includes ensemble models leveraging [BERT-EE](https://github.com/wilsonlau-uw/BERT-EE) to consolidate fracture events from clinical notes. There are two ensemble models: 1) esnemble majority voting on discrete fracture events and 2) ensemble averaging on [CLS] embeddings.

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
3. In `BERT_Ensemble/user_params.py`, the parameter **PATIENT_ID** must be a path to a .csv file with the data format below.

| subject_id                    | NoteID         | 
|-------------------------------|----------------|
| *string*                      | *string*       |

Notes:

1. Please refer to the [brat rapid annotation tool](https://brat.nlplab.org/) documentation as needed for annotating notes. 

### BERT-EE Models & Ensemble Models
Please specify the ensemble model to run in `BERT_Ensemble/user_params.py`

```python
python BERT_Ensemble/main.py
```

## Acknowledgments
The annotation of the training corpus used a [Docker container of BRAT](https://github.com/ddevaraj/docker-brat).