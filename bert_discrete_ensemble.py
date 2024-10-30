import os
from os import listdir
from os.path import isfile, join
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from brat_util import *
from utils import create_dir

os.path.dirname(os.path.abspath(__file__))

class BERT_Discrete:
    """
    Consolidate predicted fracture events to note- and patient-level from bert-base-cased, ClinicalBERT, & BioBERT
    Counts unique fracture events via rule-based approach
    - If fracture event has the same FxAttributeID and FxAttributeValue, count as the same event.
    - Applies ensemble majority voting grouped by 'FxEvent' and 'FxRelation' on 'FxAttributeValue'
    - If fracture event has FxAttributeValue = 'Absent' or 'Possible' for FractureAssertion,
        it is not includeed in the unique fracture count.

    Parameters
    ----------
    ner_df (DataFrame): output from BRAT_annotations.get_events_ner
    re_df (DataFrame): output from BRAT_annotations.get_events_re
    pred_dir (list): directories of predicted BRAT ann files
    save_dir (str): directory to save consolidated events
    patientID (str): file with patient IDs matching to note IDs
    span_tolerance (int): number of characters for majority voting tolerance across events

    Returns
    -------
    consolidated_events (list): list of dictionaries with filename and
        consolidated fracture events
    """

    def __init__(self, ner_df, re_df, pred_dir, patientID, save_dir, save_name, span_tolerance=10):
        super().__init__()
        self.ner_df = ner_df
        self.re_df = re_df
        self.pred_dir = pred_dir
        self.patientID = patientID
        self.save_dir = save_dir
        self.save_name = save_name
        self.span_tolerance = span_tolerance
        create_dir(self.save_dir)

    def getFileNames_ner(self):
        """
        Get matching files for each clinical note ID.

        Returns
        -------
        filenames (dict): {NoteID: [note files]}
        NoteID (list): list of unique Note IDs
        """
        files = []
        NoteID = []
        for d in self.pred_dir:
            d = os.path.join(d, "ner")
            annFiles = [f for f in listdir(d) if isfile(join(d, f)) and f.endswith('.ann')]
            # ann_dir = [f for f in annFiles if 'txt' not in f]

            for note in annFiles:
                basename = os.path.basename(note).split('.')[0][:-12]
                NoteID.append(note.split('_')[0])
                files.append(os.path.join(d, note))

        NoteID = list(set(NoteID))
        filenames = {x:[f for f in files if x in f] for x in NoteID}

        return filenames, NoteID
    
    def getFileNames_re(self):
        """
        Get matching files for each clinical note ID.

        Returns
        -------
        filenames (dict): {NoteID: [note files]}
        NoteID (list): list of unique Note IDs
        """
        files = []
        NoteID = []
        for d in self.pred_dir:
            d = os.path.join(d, "re")
            annFiles = [f for f in listdir(d) if isfile(join(d, f)) and f.endswith('.ann')]
            # ann_dir = [f for f in annFiles if 'txt' not in f]

            for note in annFiles:
                basename = os.path.basename(note).split('.')[0][:-12]
                NoteID.append(note.split('_')[0])
                files.append(os.path.join(d, note))

        NoteID = list(set(NoteID))
        filenames = {x:[f for f in files if x in f] for x in NoteID}

        return filenames, NoteID

    def get_patientIDs(self):
        """
        Get matching note IDs for each patient.

        Returns
        -------
        pt_notes (dict): {PatientID: [NoteIDs]}
        """
        self.__pt_notes = {}
        patient_df = pd.read_csv(self.patientID)
        for pt in patient_df.subject_id.unique().tolist():
            slc = patient_df.loc[patient_df.subject_id == pt]

            subject_id = slc.subject_id.unique().item()
            notes = slc.NoteID.unique().tolist()

            self.__pt_notes[subject_id] = notes

        return self.__pt_notes

    def consolidate_events(self):
        """
        Returns
        -------
        consolidated_events (dict): consolidated ensemble majority voting
            fracture events at note level
        """
        self.__consolidated_events = dict()
        filenames, NoteID = self.getFileNames()
        for note in NoteID:
            slc = self.df.loc[self.df["Filename"].str.contains(note)].reset_index()

            # drop duplicate textbound IDs by common FxAttributeValue
            slc2 = (
                slc.drop_duplicates(
                    subset=[
                        "Filename",
                        "FxAttributeID",
                        "FxAttributeValue",
                        "FxRelation",
                    ],
                    keep="first",
                )
                .sort_values(by="FxEvent")
                .reset_index()
            )

            slc_grouped = (
                slc.groupby(["FxEvent", "FxRelation", "FxAttributeValue"])
                .apply(pd.DataFrame.mode)
                .reset_index(drop=True)
                .dropna()
            )

            # conslidate each unique fracture event ID to columns for FxAttribute and return values
            unique_events = []
            # for file in consolidated_events['Filename'].unique().tolist():
            unique_fracture = 0
            # tmp = dict()
            if slc_grouped.shape[1] != 4:
                tmp = dict()
                tmp["FractureEvent"] = "no_mention"
                tmp["FractureAnatomy"] = "no_mention"
                tmp["FractureAssertion"] = "no_mention"
                unique_events.append(tmp)
            else:
                for event in slc_grouped.FxEvent.unique().tolist():
                    slc_event = slc_grouped.loc[slc_grouped.FxEvent == event]
                    for i, row in slc_event.iterrows():
                        tmp = dict()
                        tmp["FractureEvent"] = event
                        # if row.FxAttribute == 'FractureAnatomy':
                        if "Anatomy" in row.FxRelation:
                            tmp["FractureAnatomy"] = row.FxAttributeValue
                        # elif row.FxAttribute == 'FractureCause':
                        elif "Cause" in row.FxRelation:
                            tmp["FractureCause"] = row.FxAttributeValue
                        # elif row.FxAttribute == 'FractureAssertion':
                        elif "Assertion" in row.FxRelation:
                            tmp["FractureAssertion"] = row.FxAttributeValue

                        unique_events.append(tmp)

            unique_events_df = pd.DataFrame(unique_events)

            if unique_events_df.shape[0] != 0:
                unique_events_grouped = (
                    unique_events_df.groupby("FractureEvent")
                    .apply(lambda x: x.drop_duplicates())
                    .reset_index(drop=True)
                )

                if "FractureAssertion" in unique_events_grouped.columns:
                    unique_fracture_count = unique_events_grouped.loc[
                        ~unique_events_grouped.FractureAssertion.isin(
                            ["Absent", "Possible", "no_mention"]
                        )
                    ].shape[0]
                    unique_fracture += int(unique_fracture_count)
                else:
                    unique_fracture += int(unique_events_grouped.shape[0])
            else:
                pass

            uniqueFxEvents = []
            for event in unique_events_grouped.FractureEvent.unique().tolist():
                slc = unique_events_grouped.loc[
                    unique_events_grouped.FractureEvent == event
                ]
                if slc.shape[0] > 1:
                    slc = slc.ffill().drop_duplicates().dropna()
                    uniqueFxEvents.append(slc)
                else:
                    uniqueFxEvents.append(slc)

            finalFxEvents = pd.concat(uniqueFxEvents)

            self.__consolidated_events[note] = {
                "FxEvents": finalFxEvents,
                "FxCount": unique_fracture,
            }
        return self.__consolidated_events
    
    def consolidate_events_ner(self):
        filenames, NoteID = self.getFileNames_ner()
        self.ner_df = self.ner_df.drop_duplicates(subset=['Filename','FxEvent', 'start_idx'], keep='first')

        self.__ner_events = {}
        for note in NoteID:
            unique_events = []
            slc = self.ner_df.loc[self.ner_df['Filename'].str.contains(note)].reset_index()
            
            if slc.shape[0] == 0:
                tmp = dict()
                tmp['FractureEvent'] = 'no_mention'
                tmp['FractureAnatomy'] = 'no_mention'
                tmp['FractureAssertion'] = 'no_mention'
                unique_events.append(tmp)
            elif slc.shape[0] == 1:
                tmp = dict()
                tmp['start_idx'] = slc.start_idx.iloc[0]
                if 'Anatomy' in slc.FxAttribute.iloc[0]:
                    tmp['FractureAnatomy'] = slc.FxAttributeValue.iloc[0]
                unique_events.append(tmp)
            elif slc['start_idx'].eq('None').all():
                tmp3 = dict()
                tmp3['FractureAnatomy'] = "no_mention"
                tmp3['FractureCause'] = "no_mention"
                tmp3['FractureAssertion'] = "no_mention"
                unique_events.append(tmp3)
            else:    
                slc_none = slc.loc[slc.FxEvent == "None"]
                slc_drop_none = slc.loc[slc.FxEvent != "None"]
                slc_grouped = slc_drop_none.groupby(['FxEvent', (slc_drop_none['start_idx'].astype(int) + self.span_tolerance)])\
                                .FxAttribute.apply(pd.Series.mode).reset_index()

                if 'start_idx' in slc_grouped.columns:
                    for idx in slc_grouped.start_idx.unique().tolist():
                        slc_event = slc_grouped.loc[slc_grouped.start_idx == idx]
                        for i,row in slc_event.iterrows():
                            tmp3 = dict()
                            tmp3['start_idx'] = idx
                            # tmp['FxEvent'] = row.FxEvent
                            # if row.FxAttribute == 'FractureAnatomy':
                            tmp3['FractureAnatomy'] = "unspecified"
                            tmp3['FractureCause'] = "unspecified"
                            tmp3['FractureAssertion'] = "unspecified"
                            unique_events.append(tmp3)
                elif slc_grouped.empty:
                    tmp3 = dict()
                    tmp3['FractureAnatomy'] = "no_mention"
                    tmp3['FractureCause'] = "no_mention"
                    tmp3['FractureAssertion'] = "no_mention"
                    unique_events.append(tmp3)


            unique_events_df = pd.DataFrame(unique_events)
            unique_fracture = 0
            if 'start_idx' in unique_events_df.columns:
                unique_events_grouped = unique_events_df.groupby(['start_idx'], group_keys=False).\
                                            apply(lambda x: x.drop_duplicates()).reset_index(drop=True)

                if 'FractureAssertion' in unique_events_grouped.columns:
                    unique_fracture_count = unique_events_grouped.loc[~unique_events_grouped.FractureAssertion.isin(['Absent', 'Possible', 'no_mention'])].shape[0]
                    unique_fracture += int(unique_fracture_count)
                else:
                    # unique_fracture += int(unique_events_grouped.shape[0])
                    pass
            else:
                # pass
                unique_events_grouped = pd.DataFrame(columns=["FractureAnatomy"])

            uniqueFxEvents = []
            if not unique_events_grouped.empty:
                for idx in unique_events_grouped.start_idx.unique().tolist():
                    unique_slc = unique_events_grouped.loc[unique_events_grouped.start_idx == idx]
                    if unique_slc.shape[0] > 1:
                        unique_slc = unique_slc.ffill().drop_duplicates().dropna()
                        uniqueFxEvents.append(unique_slc)
                    else:
                        uniqueFxEvents.append(unique_slc)
                FxEvents = pd.concat(uniqueFxEvents)
            else:
                FxEvents = pd.DataFrame(data={"FractureAnatomy": "no_mention", 
                                            "FractureCause": "no_mention", 
                                            "FractureAssertion": "no_mention"}, index=[0])

            if "start_idx" in FxEvents.columns:
                FxEvents = FxEvents.drop(columns=["start_idx"])

            unique_fracture = 0
            if 'FractureAnatomy' in FxEvents.columns:
                finalFxEvents = FxEvents
                unique_fracture += int(finalFxEvents.shape[0])
            elif FxEvents.empty:
                finalFxEvents = FxEvents
                finalFxEvents['FractureAnatomy'] = "no_mention"
                finalFxEvents['FractureCause'] = "no_mention"
                finalFxEvents['FractureAssertion'] = "no_mention"
                unique_fracture += 0
            
            self.__ner_events[note] = {'FxEvents': finalFxEvents, 'FxCount': unique_fracture}

        return self.__ner_events

    def consolidate_events_re(self):
        """
        Consolidate events via ensemble majority voting
            by grouping based on span start idx

        Paramters
        ---------
        span_tolerance (int): number of characters for span start_idx tolerance

        Returns
        -------
        consolidated_events (dict): consolidated ensemble majority voting
            fracture events at note level
        """
        filenames, NoteID = self.getFileNames_re()

        self.__re_events = dict()
        for note in NoteID:
            unique_events = []
            unique_fracture = 0

            slc = self.re_df.loc[self.re_df['Filename'].str.contains(note)].reset_index()
            if slc.shape[0] == 0:
                tmp = dict()
                tmp['start_idx'] = 'no_mention'
                tmp['FractureAnatomy'] = 'no_mention'
                tmp['FractureAssertion'] = 'no_mention'
                unique_events.append(tmp)
            elif slc.shape[0] == 1:
                tmp = dict()
                tmp['start_idx'] = slc.start_idx.iloc[0]
                if 'Anatomy' in slc.FxAttribute.iloc[0]:
                    tmp['FractureAnatomy'] = slc.FxAttributeValue.iloc[0]

                unique_events.append(tmp)
            else:
                slc['FxRelation'] = slc['FxRelation'].str.replace('\d+', '')

                # drop duplicate textbound IDs by common FxAttributeValue
                slc2 = slc.drop_duplicates(subset=['Filename', 'FxAttributeID', 'FxAttributeValue', 'FxRelation'],
                                             keep='first').sort_values(by='FxEvent').reset_index()

                slc_grouped = slc.groupby(['FxAttribute', (slc['start_idx'].astype(int) + self.span_tolerance)]).FxAttributeValue\
                                    .apply(pd.Series.mode).reset_index()

                if 'start_idx' in slc_grouped.columns:
                    for idx in slc_grouped.start_idx.unique().tolist():
                        slc_event = slc_grouped.loc[slc_grouped.start_idx == idx]
                        for i,row in slc_event.iterrows():
                            tmp3 = dict()
                            tmp3['start_idx'] = idx
                            # tmp['FxEvent'] = row.FxEvent
                            # if row.FxAttribute == 'FractureAnatomy':
                            if 'Anatomy' in row.FxAttribute:
                                tmp3['FractureAnatomy'] = row.FxAttributeValue
                            # elif row.FxAttribute == 'FractureCause':
                            elif 'Cause' in row.FxAttribute:
                                tmp3['FractureCause'] = row.FxAttributeValue
                            # elif row.FxAttribute == 'FractureAssertion':
                            elif 'Assertion' in row.FxAttribute:
                                tmp3['FractureAssertion'] = row.FxAttributeValue

                            unique_events.append(tmp3)
                elif not slc_grouped.empty:
                    tmp3 = dict()
                    tmp3['FractureAnatomy'] = row.FxAttributeValue

                    unique_events.append(tmp3)

            unique_events_df = pd.DataFrame(unique_events)

            if 'start_idx' in unique_events_df.columns:
                unique_events_grouped = unique_events_df.groupby(['start_idx'], group_keys=False).\
                                            apply(lambda x: x.drop_duplicates()).reset_index(drop=True)

                if 'FractureAssertion' in unique_events_grouped.columns:
                    unique_fracture_count = unique_events_grouped.loc[~unique_events_grouped.FractureAssertion.isin(['Absent', 'Possible', 'no_mention'])].shape[0]
                    unique_fracture += int(unique_fracture_count)
                else:
                    pass
            else:
                pass

            uniqueFxEvents = []
            for idx in unique_events_grouped.start_idx.unique().tolist():
                unique_slc = unique_events_grouped.loc[unique_events_grouped.start_idx == idx]
                if unique_slc.shape[0] > 1:
                    unique_slc = unique_slc.ffill().drop_duplicates().dropna()
                    uniqueFxEvents.append(unique_slc)
                else:
                    uniqueFxEvents.append(unique_slc)

            FxEvents = pd.concat(uniqueFxEvents)
            unique_fracture = 0
            if 'FractureAnatomy' in FxEvents.columns:
                finalFxEvents = FxEvents.drop_duplicates(subset='FractureAnatomy', keep='first')
                positiveFx = finalFxEvents.loc[~finalFxEvents.FractureAnatomy.isin(['no_mention'])]
                unique_fracture += int(positiveFx.FractureAnatomy.dropna().nunique())
            else:
                finalFxEvents = FxEvents
                finalFxEvents['FractureAnatomy'] = "no_mention"
                unique_fracture += 0

            self.__re_events[note] = {'FxEvents': finalFxEvents, 'FxCount': unique_fracture}

        return self.__re_events
        
    def update_events(self):
        """
        Update NER events with RE events if keys match
        """
        self.__ner_events = None
        self.__re_events = None
        self.__ner_events = self.consolidate_events_ner()
        try:
            self.__re_events = self.consolidate_events_re()
        except Exception as e:
            print(f"Exception: {e}")

        if self.__re_events:
            self.__ner_events.update(self.__re_events)

        return self.__ner_events

    def consolidate_patient_level(self):
        """
        Consolidates patient-level fracture events from
            note-level fracture events

        Returns
        -------
        pt_events (dict): patient-level fracture events
        """
        self.__final_events = self.update_events()
        self.__pt_notes = self.get_patientIDs()
        self.__pt_events = {}
        self.__dfs = []

        for pt in self.__pt_notes:
            for note in self.__pt_notes[pt]:
                self.__events = []
                count = 0
                if str(note) in self.__final_events:
                    df = self.__final_events[str(note)]['FxEvents']
                    
                    self.__events.append(df)
                if len(self.__events) > 0:
                    # concatenate all the note-level events for the same patient
                    concat_df = pd.concat(self.__events)
                    concat_df = concat_df.ffill().drop_duplicates()
                    
                    if 'FractureCause' not in concat_df.columns:
                        concat_df['FractureCause'] = np.NaN
                    if 'FractureAssertion' not in concat_df.columns:
                        concat_df['FractureAssertion'] = np.NaN
                    
                    # count unique fracture events only if explicitly "positive"
                    fx_count = concat_df.loc[~concat_df.FractureAssertion.isin(["Absent", "Possible", "no_mention"])].shape[0]
                    count += fx_count
                    self.__dfs.append(concat_df)
                    
                    self.__pt_events[pt] = {'FxEvents':concat_df.to_records(index=False), 'FxCount':count}
                else:
                    data = {'FractureAnatomy': np.NaN, 'FractureCause': np.NaN, 'FractureAssertion': np.NaN}
                    df = pd.DataFrame(data, index=[0])
                    self.__dfs.append(df)
                    
                    self.__pt_events[pt] = {'FxEvents': df.to_records(index=False), 'FxCount':count}

        return self.__pt_events
    
    def save_pt_events(self):
        """
        Save pt_events to pickle file
        """
        self.__pt_events = self.consolidate_patient_level()

        with open(os.path.join(self.save_dir, self.save_name+".pkl"), "wb") as f:
            pickle.dump(self.__pt_events, f)

    def note_fracture_labels(self):
        """
        Returns note-level labels via a rule-based method

        Returns
        -------
        note_labels (DataFrame): contains note-level fracture labels per note
        """
        filenames, NoteID = self.getFileNames_re()
        re_events = self.consolidate_events_re()

        note_labels = []
        for note in NoteID:
            slc = re_events[note]["FxEvents"]

            if "no_mention" in slc.FractureEvent.values:
                tmp = dict()
                tmp["radReport"] = note
                tmp["fracture_label"] = 0
                note_labels.append(tmp)
            elif ("FractureAssertion" in slc.columns) and (
                "absent" in slc.FractureAssertion.values
            ):
                tmp2 = dict()
                tmp2["radReport"] = note
                tmp2["fracture_label"] = -1
                note_labels.append(tmp2)
            else:
                tmp3 = dict()
                tmp3["radReport"] = note
                tmp3["fracture_label"] = 1
                note_labels.append(tmp3)

        return pd.DataFrame(note_labels)