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

def create_dir(save_dir):
    """
    Creates new directory if it does not exists

    Parameters
    ----------
    save_dir (str): desired directory name
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, mode=0o777)


def majorityVote(x):
    """
    Majority voting algorithm that counts on Pandas series
    Will remove "ambiguous" events if there is no clear majority

    Parameters
    ----------
    x (Series): Pandas Series to run majority voting
    """
    m = pd.Series.mode(x)
    if len(m) == 1:
        return m

class BRAT_annotations:
    """
    Post process predicted BRAT annotation files to discrete events

    Parameters
    ----------
    pred_dir (list): directories of BERT-EE predictions (MUST contain predicted BRAT .ann files)
    task (str): outputs to use ["ner", "re"]

    Returns
    -------
    note_events (DataFrame): DataFrame of predicted fracture events, arguments, and relations
    """

    def __init__(self, pred_dir):
        super().__init__()
        self.pred_dir = pred_dir

    def get_events_re(self):
        """
        Parse BRAT .ann files to DataFrame
        """
        note_events = []
        for d in self.pred_dir:
            re_dir = os.path.join(d, "re")
            ann_files = [f for f in listdir(re_dir) if isfile(join(re_dir, f)) and f.endswith('.ann')]
            sortedFiles = [join(re_dir, splitext(a)[0]) for a in ann_files]
            sortedFiles.sort()

            events = []
            relations = []
            textbounds = []
            attributes = []

            empty_notes = []
            for f,fname in enumerate(tqdm(sortedFiles)):
                with open(fname+'.ann', 'r') as ann_file:
                    ann_lines = ann_file.read()
                    if len(ann_lines) > 0:
                        _events, _relations, _textbounds, _attributes=get_annotations(ann_lines)
                        events.append(_events)
                        relations.append(_relations)
                        textbounds.append(_textbounds)
                        attributes.append(_attributes)

                        df = []
                        for k, v in _events.items():
                            for _key, _value in v.arguments.items():
                                tmp = dict()

                                tmp['Filename'] = os.path.basename(fname).split('/')[0]
                                tmp['FxEvent'] = v.id
                                tmp['FxRelation'] = _key
                                tmp['FxText'] = _textbounds[_value].text
                                tmp['FxAttribute'] = _textbounds[_value].type_
                                tmp['start_idx'] = _textbounds[_value].start
                                tmp['end_idx'] = _textbounds[_value].end
                                tmp['FxAttributeID'] = _value
                                if _key != 'Fracture':
                                    tmp['FxAttributeValue'] = _attributes[_value].value
                                    df.append(tmp)

                    else:
                        tmp = dict()

                        tmp['Filename'] = os.path.basename(fname).split('/')[0]
                        tmp['FxEvent'] = 'None'
                        tmp['FxRelation'] = 'None'
                        tmp['FxText'] = 'None'
                        tmp['FxAttribute'] = 'None'
                        tmp['start_idx'] = 'None'
                        tmp['end_idx'] = 'None'
                        tmp['FxAttributeID'] = 'None'
                        tmp['FxAttributeValue'] = 'None'

                        empty_notes.append(tmp)

                    results = pd.DataFrame((df + empty_notes))
                    note_events.append(results)

        return pd.concat(note_events)
    
    def get_events_ner(self):
        ner_events = []
        for d in self.pred_dir:
            ner_dir = os.path.join(d, 'ner')
            ann_files = [f for f in listdir(ner_dir) if isfile(join(ner_dir, f)) and f.endswith('.ann')]
            sortedFiles = [join(ner_dir, splitext(a)[0]) for a in ann_files]
            sortedFiles.sort()

            events = []
            relations = []
            textbounds = []
            attributes = []

            empty_notes = []
            df = []
            
            for f,fname in enumerate(tqdm(sortedFiles)):
                # if os.path.getsize(fname+'.ann') > 0:
                # tmp = dict()
                with open(fname+'.ann', 'r') as ann_file:
                    ann_lines = ann_file.read()
                    if len(ann_lines) > 0:                
                        _events, _relations, _textbounds, _attributes=get_annotations(ann_lines)
                        events.append(_events)
                        relations.append(_relations)
                        textbounds.append(_textbounds)
                        attributes.append(_attributes)

                        for k, v in _events.items():
                            for _key, _value in v.arguments.items():
                                tmp = dict()

                                # tmp['Filename'] = fname.split('/')[2]+'_'+fname.split('/')[4]
                                tmp['Filename'] = os.path.basename(fname).split('/')[0]
                                tmp['FxEvent'] = v.id
                                tmp['FxRelation'] = _key
                                tmp['FxText'] = _textbounds[_value].text
                                tmp['FxAttribute'] = _textbounds[_value].type_
                                tmp['start_idx'] = _textbounds[_value].start
                                tmp['end_idx'] = _textbounds[_value].end
                                tmp['FxAttributeID'] = _value
                                df.append(tmp)
                                if _key != 'Fracture':
                                    tmp['FxAttributeValue'] = _attributes[_value].value
                                    df.append(tmp)

                    else:
                        tmp2 = dict()
                        # tmp2['Filename'] = fname.split('/')[2]+'_'+fname.split('/')[4]
                        tmp2['Filename'] = os.path.basename(fname).split('/')[0]
                        tmp2['FxEvent'] = 'None'
                        tmp2['FxRelation'] = 'None'
                        tmp2['FxText'] = 'None'
                        tmp2['FxAttribute'] = 'None'
                        tmp2['start_idx'] = 'None'
                        tmp2['end_idx'] = 'None'
                        tmp2['FxAttributeID'] = 'None'
                        tmp2['FxAttributeValue'] = 'None'

                        empty_notes.append(tmp2)

                    results = pd.DataFrame((df + empty_notes))
                    ner_events.append(results)

        return pd.concat(ner_events)


class CPU_Unpickler(pickle.Unpickler):
    """
    For handling loading tensors from a CUDA device to CPU
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


# class Perceptron(nn.Module):
#     """
#     Neural network for the BERT hidden state ensemble method
#     """

#     def __init__(self, bert_hidden_state):
#         super(Perceptron, self).__init__()
#         self.bert_hidden_state = bert_hidden_state
#         # self.num_neurons = num_neurons

#         self.fc1 = nn.Linear(self.bert_hidden_state, 128)
#         self.hidden1 = nn.Linear(128, 64)
#         self.hidden2 = nn.Linear(64, 32)
#         self.hidden3 = nn.Linear(32, 16)
#         self.fc2 = nn.Linear(16, 8)
#         # self.fc2 = nn.Linear(8, 4)

#     def forward(self, x):
#         out = F.leaky_relu(self.fc1(x))
#         out = F.leaky_relu(self.hidden1(out))
#         out = F.leaky_relu(self.hidden2(out))
#         out = F.leaky_relu(self.hidden3(out))
#         # out = F.leaky_relu(self.hidden4(out))
#         out = F.leaky_relu(self.fc2(out))

#         return out
    
# class Perceptron(nn.Module):
#     """
#     Neural network for the BERT hidden state ensemble method
#     """

#     def __init__(self, bert_hidden_state):
#         super(Perceptron, self).__init__()
#         self.bert_hidden_state = bert_hidden_state
#         # self.num_neurons = num_neurons

#         self.fc1 = nn.Linear(self.bert_hidden_state, 256)
#         # self.hidden1 = nn.Linear(128, 64)
#         # self.hidden2 = nn.Linear(64, 32)
#         # self.hidden3 = nn.Linear(32, 16)
#         # self.fc2 = nn.Linear(16, 8)
#         # self.fc2 = nn.Linear(8, 4)

#     def forward(self, x):
#         # out = F.leaky_relu(self.fc1(x))
#         out = self.fc1(x)
#         # out = F.leaky_relu(self.hidden1(out))
#         # out = F.leaky_relu(self.hidden2(out))
#         # out = F.leaky_relu(self.hidden3(out))
#         # # out = F.leaky_relu(self.hidden4(out))
#         # out = F.leaky_relu(self.fc2(out))

#         return out
