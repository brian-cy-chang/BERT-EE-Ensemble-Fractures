import os
from os import listdir
from os.path import isfile, join, splitext
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
import io

import torch
import torch.nn as nn
import torch.nn.functional as F

from brat_util import *

os.path.dirname(os.path.abspath(__file__))

class BERT_HiddenState:
    """
    Ensemble averaging on BERT [CLS] embeddings and average pooled last hidden state outputs

    Parameters
    ----------
    logits (str): directory to pickle files of BERT hidden states
    patientID (str): file with patient IDs matching to note IDs
    save_dir (str): directory to save output
    save_name (str): filename for the ensemble averaging output
    """

    def __init__(self, logits, patientID, mode, save_dir, save_name):
        super().__init__()
        self.logits = logits
        self.patientID = patientID
        self.mode = mode
        self.save_dir = save_dir
        self.save_name = save_name

        self.__to_device()

    def __to_device(self):
        """
        Load CUDA (GPU) device if available.
        Otherwise continue on CPU.
        """
        if torch.cuda.is_available():
            self.__device = torch.device("cuda")
        else:
            self.__device = torch.device("cpu")

    def getFileNames(self):
        """
        Get corresponding pickle files with BERT last hidden state outputs for each clinical note.

        Returns
        -------
        filenames (dict): pickle files for each BERT model for a given note
        """
        files = []
        NoteID = []
        for dir in self.logits:
            pklFiles = [
                f for f in listdir(dir) if isfile(join(dir, f)) and f.endswith(".pkl")
            ]
            tensor_dir = [f for f in pklFiles if "txt" not in f]

            for note in tensor_dir:
                basename = os.path.basename(note).split(".")[0][:-12]
                NoteID.append(note.split("_")[0])
                files.append(os.path.join(dir, note))

        NoteID = list(set(NoteID))
        filenames = {x: [f for f in files if x in f] for x in NoteID}

        return filenames

    def get_patient_notes(self):
        """
        Get corresponding note IDs for each patient ID

        Returns
        -------
        pt_notes (dict): {PatientID: [NoteIDs]}
        """
        self.__pt_notes = {}

        patient_df = pd.read_csv(self.patientID)

        for pt in patient_df.subject_id.unique().tolist():
            slc = patient_df.loc[patient_df.subject_id == pt]

            ptID = slc.subject_id.unique().item()
            notes = slc.NoteID.unique().tolist()

            self.__pt_notes[ptID] = notes

        return self.__pt_notes

    def getCLS(self):
        """
        Get CLS token embeddings from last hidden state outputs for each input sequence per note.
        Apply ensemble averaging for each input sequence.

        Returns
        -------
        Ensembles (dict): stacked ensemble CLS token embeddings per note
        """
        filenames = self.getFileNames()

        self.__note_ensembles = {}
        for note in filenames:
            files = filenames[note]

            self.__cls_embeddings = []
            for file in files:
                basename = (
                    file.split("/")[2]
                    + "_"
                    + os.path.basename(file).split(".")[0][:-12]
                )
                with open(file, "rb") as f:
                    outputs = pickle.load(f)

                    self.__cls = []
                    for idx in range(len(outputs)):
                        self.__tmp = dict()
                        slc = outputs[idx]
                        CLSToken = slc["hidden_state"][0, 0, :]

                        self.__tmp["filename"] = basename + "_" + str(idx)
                        self.__tmp["CLS"] = CLSToken.to(self.__device)

                        self.__cls.append(self.__cls)

                        self.__cls_embeddings.append(self.__tmp)
            AverageCLS = self.EnsembleCLS(self.__cls_embeddings)

            self.__note_ensembles[str(note)] = AverageCLS

        return self.__note_ensembles

    def EnsembleCLS(self, cls_embeddings):
        """
        Ensemble model of CLS token embeddings. For a note, each BERT model's
            embedding is passed through class Perceptron.

        Parameters
        ----------
        cls_embeddings (list): dictionaries of CLS token embeddings
            for each input sequence

        Returns
        -------
        Concatenated ensemble average CLS embedding features
        """
        self.cls_embeddings = cls_embeddings
        self.__ensembles = []
        for j in range(len(self.cls_embeddings[0])):
            cls = []
            for i in range(len(self.cls_embeddings)):
                cls.append(self.cls_embeddings[i][j]["CLS"])

            cls_stack = torch.stack(cls, dim=0)
            cls_mean = torch.mean(cls_stack, dim=0, keepdim=True)

            self.__ensembles.append(cls_mean.detach().cpu())

        return torch.stack(self.__ensembles)

    def getHiddenState(self):
        """
        Get last hidden state outputs from each BERT model per input sequence.
        Apply adaptive average pooling to each last hidden state output.
        Apply ensemble method by passing the pooled features to class Perceptron.

        Returns
        -------
        ensemble_hidden_states (dict): concatenated averaged last hidden state outputs per note
        """
        filenames = self.getFileNames()

        self.__ensemble_hidden_states = {}
        for note in filenames:
            files = filenames[note]

            notes = dict()

            self.__hidden_state_embeddings = []

            for file in files:
                basename = (
                    file.split("/")[2]
                    + "_"
                    + os.path.basename(file).split(".")[0][:-12]
                )
                with open(file, "rb") as f:
                    outputs = pickle.load(f)

                    self.__avg_hidden_state = []
                    self.__hidden_state = None

                    for idx in range(len(outputs)):
                        tmp = dict()
                        slc = outputs[idx]

                        # average pooling across the sequence
                        self.__hidden_state = torch.mean(slc["hidden_state"].unsqueeze(0), dim=1).to(self.__device)

                        tmp["filename"] = basename + "_" + str(idx)
                        tmp["avg_hidden_state"] = self.__hidden_state

                        self.__avg_hidden_state.append(tmp)

                    self.__hidden_state_embeddings.append(self.__hidden_state)
                AverageHiddenState = self.EnsembleHiddenState(self.__hidden_state_embeddings)

            self.__ensemble_hidden_states[note] = AverageHiddenState

        return self.__ensemble_hidden_states

    def EnsembleHiddenState(self, hidden_state_embeddings):
        """
        Returns
        -------
        NoteHiddenState (tensor): ensemble model output of the
            averaged last hidden state outputs for a note
        """
        self.hidden_state_embeddings = hidden_state_embeddings
        self.__EnsembleHiddenState = []

        for j in range(len(self.hidden_state_embeddings[0])):
            basename = self.hidden_state_embeddings[0][0]["filename"].split("_")[2]
            self.__hiddenState = []
            for i in range(len(self.hidden_state_embeddings)):
                self.__hiddenState.append(
                    self.hidden_state_embeddings[i][j]["avg_hidden_state"]
                )

            hidden_state_stack = torch.stack(self.__hiddenState, dim=0).to(self.__device)
            hidden_state_mean = torch.mean(hidden_state_stack, dim=0, keepdim=True)

            self.__EnsembleHiddenState.append(hidden_state_mean.detach().cpu())

        return torch.stack(self.__EnsembleHiddenState)

    def __AdaptiveAveragePool(self, input, output_size=(512, 256)):
        """
        Apply AdaptiveAvgPool2d to fixed output size (512,256)

        Parameters
        ----------
        input (PyTorch tensor)
        """
        self.input = input
        self.output_size = output_size

        m = nn.AdaptiveAvgPool2d(self.output_size)
        AvgPoolTensor = m(self.input)

        return AvgPoolTensor
    
    def get_patient_event(self):
        if self.mode == "cls":
            ensemble_notes_stack = self.getCLS()
        elif self.mode == "average":
            ensemble_notes_stack = self.getHiddenState()
        pt_notes = self.get_patient_notes()

        self.__pt_event = {}
        for pt in pt_notes:
            for note in pt_notes[pt]:
                self.__cls_features = []
                if str(note) in list(ensemble_notes_stack.keys()):
                    note_cls = ensemble_notes_stack[str(note)]
                    self.__cls_features.append(note_cls)

                    self.__pt_cls[pt] = self.__cls_features[0]

        # save to pickle file
        self.save_file(self.__pt_cls)         

    def save_file(self, ensembles):
        """
        Save getCLS() ensemble average [CLS] embeddings to a pickle file

        Parameters
        ----------
        ensembles (dict): ensemble averaged [CLS] features 
            per subject
        """
        with open(os.path.join(self.save_dir, self.save_name+".pkl"), "wb") as f:
            pickle.dump(ensembles, f)
