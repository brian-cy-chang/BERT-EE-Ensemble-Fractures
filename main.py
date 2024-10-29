import os
import subprocess

import user_params as uparams
from utils import *

from bert_discrete_ensemble import BERT_Discrete
from bert_hidden_state_ensemble import BERT_HiddenState

def main():
    # run each BERT-based model in parallel
    print(f"*** Running BERT-EE models on notes from {uparams.NOTES_DIR} ***")
    # List of commands to run
    bert_models = [
        ['python', os.path.join(uparams.BERT_MODELS_DIR, "main_base.py"), f"--ner_prediction_folder={uparams.NOTES_DIR}", f"--general_results_folder={uparams.BERT_BASE_DIR}", f"--general_fine-tuned-path={uparams.BERT_BASE_FINE_TUNED_DIR}"],
        ['python', os.path.join(uparams.BERT_MODELS_DIR, "main_clinical.py"), f"--ner_prediction_folder={uparams.NOTES_DIR}", f"--general_results_folder={uparams.CLINICALBERT_DIR}", f"--general_fine-tuned-path={uparams.CLINICALBERT_FINE_TUNED_DIR}"],
        ['python', os.path.join(uparams.BERT_MODELS_DIR, "main_bio.py"), f"--ner_prediction_folder={uparams.NOTES_DIR}", f"--general_results_folder={uparams.BIOBERT_DIR}", f"--general_fine-tuned-path={uparams.BIOBERT_FINE_TUNED_DIR}"]
    ]

    # Start all processes
    processes = [subprocess.Popen(cmd) for cmd in bert_models]

    # Wait for all processes to complete
    for proc in processes:
        proc.wait()
    print(f"******** BERT-EE models completed *******")

    # run post processing with user-defined parameters after BERT-based models complete
    if uparams.ENSEMBLE_METHOD == "discrete":
        print(f"******** Running BERT-EE ensemble majority voting ********")
        brat = BRAT_annotations(pred_dir=uparams.LOGITS)
        re_df = brat.get_events_ner()
        ner_df = brat.get_events_re()

        bert_discrete_obj = BERT_Discrete(ner_df=ner_df, re_df=re_df, pred_dir=uparams.LOGITS, patientID=uparams.PATIENT_ID, save_dir=uparams.SAVE_DIR, save_name=uparams.SAVE_NAME, span_tolerance=uparams.SPAN_TOLERANCE)
        bert_discrete_obj.save_pt_events()

    elif (
        uparams.ENSEMBLE_METHOD == "hiddenState"
        and uparams.HIDDENSTATE_MODE == "cls"
    ):  # TODO
        print(f"******** Running BERT-EE ensemble averaging on [CLS] embeddings ********")
        hidden_state_obj = BERT_HiddenState(
            logits=uparams.LOGITS, patientID=uparams.PATIENT_ID, mode=uparams.HIDDENSTATE_MODE, save_dir=uparams.SAVE_DIR, save_name=uparams.SAVE_NAME
        )
        hidden_state_obj.get_patient_event()
        print(f"******** BERT-EE ensemble averaging [CLS] features saved to {uparams.SAVE_DIR}/{uparams.SAVE_NAME}.pkl ********")
        # hiddenState = BERT_HiddenState()
    elif (
        uparams.ENSEMBLE_METHOD == "hiddenState"
        and uparams.HIDDENSTATE_MODE == "average"
    ):
        print(f"******** Running BERT-EE ensemble averaging on last hidden states ********")
        hidden_state_obj = BERT_HiddenState(
            logits=uparams.LOGITS, patientID=uparams.PATIENT_ID, mode=uparams.HIDDENSTATE_MODE, save_dir=uparams.SAVE_DIR, save_name=uparams.SAVE_NAME
        )
        hidden_state_obj.getHiddenState()
        print(f"******** BERT-EE ensemble averaging hidden state features saved to {uparams.SAVE_DIR}/{uparams.SAVE_NAME}.pkl ********")
    else:
        pass

if __name__ == "__main__":
    main()