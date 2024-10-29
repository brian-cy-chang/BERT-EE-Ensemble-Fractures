"""
This module contains classes for parsing through
BRAT annotation files and converting them to a Python
dictioary or Pandas DataFrame.

Adopted from https://github.com/Lybarger/brat_scoring
"""
import os
from os import listdir
from os.path import isfile, join, splitext
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
from string import punctuation
import string
from collections import Counter
from collections import OrderedDict
import logging

import itertools


class Attribute(object):
    """
    Container for attribute

    annotation file examples:
        A1      Value T2 current
        A2      Value T8 current
        A3      Value T9 none
        A4      Value T13 current
        A5      Value T17 current
    """

    def __init__(self, id, type_, textbound, value):
        self.id = id
        self.type_ = type_
        self.textbound = textbound
        self.value = value

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return (
            (self.type_ == other.type_)
            and (self.textbound == other.textbound)
            and (self.value == other.value)
        )

    def brat_str(self):
        return attr_str(
            attr_id=self.id, arg_type=self.type_, tb_id=self.textbound, value=self.value
        )

    def id_numerical(self):
        assert self.id[0] == "A"
        id = int(self.id[1:])
        return id


class Textbound(object):
    """
    Container for textbound

    Annotation file examples:
        T2	Tobacco 30 36	smoker
        T4	Status 38 46	Does not
        T5	Alcohol 47 62	consume alcohol
        T6	Status 64 74	No history
    """

    def __init__(self, id, type_, start, end, text, tokens=None):
        self.id = id
        self.type_ = type_
        self.start = start
        self.end = end
        self.text = text
        self.tokens = tokens

    def __str__(self):
        return str(self.__dict__)

    def token_indices(self, char_indices):
        i_sent, (out_start, out_stop) = find_span(char_indices, self.start, self.end)
        return (i_sent, (out_start, out_stop))

    def brat_str(self):
        return textbound_str(
            id=self.id, type_=self.type_, start=self.start, end=self.end, text=self.text
        )

    def id_numerical(self):
        assert self.id[0] == "T"
        id = int(self.id[1:])
        return id


class Event(object):
    """
    Container for event

    Annotation file examples:
        E3      Family:T7 Amount:T8 Type:T9
        E4      Tobacco:T11 State:T10
        E2      Alcohol:T5 State:T4

        id     event:head (entities)
    """

    def __init__(self, id, type_, arguments):
        self.id = id
        self.type_ = type_
        self.arguments = arguments

    def get_trigger(self):
        for argument, tb in self.arguments.items():
            return (argument, tb)

    def __str__(self):
        return str(self.__dict__)

    def brat_str(self, tb_ids_keep=None):
        return event_str(
            id=self.id,
            event_type=self.type_,
            textbounds=self.arguments,
            tb_ids_keep=tb_ids_keep,
        )


class Relation(object):
    """
    Container for event

    Annotation file examples:
    R1  attr Arg1:T2 Arg2:T1
    R2  attr Arg1:T5 Arg2:T6
    R3  attr Arg1:T7 Arg2:T1

    """

    def __init__(self, id, role, arg1, arg2):
        self.id = id
        self.role = role
        self.arg1 = arg1
        self.arg2 = arg2

    def __str__(self):
        return str(self.__dict__)

    def brat_str(self):
        return relation_str(id=self.id, role=self.role, arg1=self.arg1, arg2=self.arg2)


COMMENT_RE = re.compile(r"^#")
TEXTBOUND_RE = re.compile(r"^T\d+")
EVENT_RE = re.compile(r"^E\d+\t")
ATTRIBUTE_RE = re.compile(r"^A\d+\t")
RELATION_RE = re.compile(r"^R\d+\t")

TEXTBOUND_LB_SEP = ";"


def get_annotations(ann):
    """
    Load annotations, including taxbounds, attributes, and events
    ann is a string
    Returns dictionaries for events, relations, textbounds, and attributes
    """
    # Parse string into nonblank lines
    lines = [l for l in ann.split("\n") if len(l) > 0]

    # Confirm all lines consumed
    remaining = [
        l
        for l in lines
        if not (
            COMMENT_RE.search(l)
            or TEXTBOUND_RE.search(l)
            or EVENT_RE.search(l)
            or RELATION_RE.search(l)
            or ATTRIBUTE_RE.search(l)
        )
    ]

    msg = f"""Could not match all annotation lines. This maybe due to a textbound with line break character in the text span."""
    for i, r in enumerate(remaining):
        msg += f"\n{i} - {repr(r)}"
        msg += "\n"

    assert len(remaining) == 0, msg

    # Get events
    events = parse_events(lines)
    # Get relations
    relations = parse_relations(lines)
    # Get text bounds
    textbounds = parse_textbounds(lines)
    # Get attributes
    attributes = parse_attributes(lines)

    return (events, relations, textbounds, attributes)


def parse_textbounds(lines):
    """
    Parse textbound annotations in input, returning a list of
    Textbound.

    ex.
        T1	Status 21 29	does not
        T1	Status 27 30	non
        T8	Drug 91 99	drug use
    """
    textbounds = {}
    for l in lines:
        if TEXTBOUND_RE.search(l):
            # Split line
            # id, type_start_end, text = l.split('\t')
            # need to update to accommodate text with tab chars
            id, type_start_end, text = l.split("\t", maxsplit=2)

            # Check to see if text bound spans multiple sentences
            mult_sent = len(type_start_end.split(";")) > 1

            # Multiple sentence span, only use portion from first sentence
            if mult_sent:
                # type_start_end = 'Drug 99 111;112 123'
                # type_start_end = ['Drug', '99', '111;112', '123']
                type_start_end = type_start_end.split()

                # type = 'Drug'
                # start_end = ['99', '111;112', '123']
                type_ = type_start_end[0]
                start_end = type_start_end[1:]

                # start_end = '99 111;112 123'
                start_end = " ".join(start_end)

                # start_ends = ['99 111', '112 123']
                start_ends = start_end.split(";")

                # start_ends = [('99', '111'), ('112', '123')]
                start_ends = [tuple(start_end.split()) for start_end in start_ends]

                # start_ends = [(99, 111), (112, 123)]
                start_ends = [(int(start), int(end)) for (start, end) in start_ends]
                start = start_ends[0][0]

                # ends = [111, 123]
                ends = [end for (start, end) in start_ends]

                text = list(text)
                for end in ends[:-1]:
                    n = end - start
                    # assert text[n].isspace()
                    text[n] = "\n"
                text = "".join(text)

                start = start_ends[0][0]
                end = start_ends[-1][-1]

            else:
                # Split type and offsets
                type_, start, end = type_start_end.split()

            # Convert start and stop indices to integer
            start, end = int(start), int(end)

            # Build text bound object
            assert id not in textbounds
            textbounds[id] = Textbound(
                id=id,
                type_=type_,
                start=start,
                end=end,
                text=text,
            )

    return textbounds


def parse_attributes(lines):
    """
    Parse attributes, returning a list of Textbound.
        Assume all attributes are 'Value'

        ex.

        A2      Value T4 current
        A3      Value T11 none

    """
    attributes = {}
    for l in lines:
        if ATTRIBUTE_RE.search(l):
            # Split on tabs
            attr_id, attr_textbound_value = l.split("\t")

            type, tb_id, value = attr_textbound_value.split()

            attr_ob = Attribute(id=attr_id, type_=type, textbound=tb_id, value=value)

            if tb_id in attributes:
                # attribute defined for textbound already, but value and type match
                # ok
                if (attributes[tb_id].type_ == attr_ob.type_) and (
                    attributes[tb_id].value == attr_ob.value
                ):
                    pass

                # attribute defined for text found already an there's a conflict between the new and old value
                # raise error
                else:
                    logging.warn("Attribute already exists for textbound")
                    logging.warn(f"Existing attribute: {attributes[tb_id]}")
                    logging.warn(f"New attribute:      {attr_ob}")
                    raise ValueError(f"Attribute already defined for textbound")

            # Add attribute to dictionary
            attributes[tb_id] = attr_ob

    return attributes


def parse_events(lines):
    """
    Parse events, returning a list of Textbound.

    ex.
        E2      Tobacco:T7 State:T6 Amount:T8 Type:T9 ExposureHistory:T18 QuitHistory:T10
        E4      Occupation:T9 State:T12 Location:T10 Type:T11

        id     event:tb_id ROLE:TYPE ROLE:TYPE ROLE:TYPE ROLE:TYPE
    """
    events = {}
    for l in lines:
        if EVENT_RE.search(l):
            # Split based on white space
            entries = [tuple(x.split(":")) for x in l.split()]

            # Get ID
            id = entries.pop(0)[0]

            # Entity type
            event_type, _ = tuple(entries[0])

            # Role-type
            arguments = OrderedDict()
            for i, (argument, tb) in enumerate(entries):
                argument = get_unique_arg(argument, arguments)
                assert argument not in arguments
                arguments[argument] = tb

            # Only include desired arguments
            events[id] = Event(id=id, type_=event_type, arguments=arguments)

    return events


def parse_relations(lines):
    """
    Parse events, returning a list of Textbound.

    ex.
    R1  attr Arg1:T2 Arg2:T1
    R2  attr Arg1:T5 Arg2:T6
    R3  attr Arg1:T7 Arg2:T1

    """
    relations = {}
    for line in lines:
        if RELATION_RE.search(line):
            # road move trailing white space
            line = line.rstrip()

            x = line.split()
            id = x.pop(0)
            role = x.pop(0)
            arg1 = x.pop(0).split(":")[1]
            arg2 = x.pop(0).split(":")[1]

            # Only include desired arguments
            assert id not in relations
            relations[id] = Relation(id=id, role=role, arg1=arg1, arg2=arg2)

    return relations


def get_unique_arg(argument, arguments):
    if argument in arguments:
        argument_strip = argument.rstrip(string.digits)
        for i in range(1, 20):
            argument_new = f"{argument_strip}{i}"
            if argument_new not in arguments:
                break
    else:
        argument_new = argument

    assert argument_new not in arguments, "Could not modify argument for uniqueness"

    if argument_new != argument:
        # logging.warn(f"Event decoding: '{argument}' --> '{argument_new}'")
        pass

    return argument_new


def get_spans(fname, _textbounds):
    idx_spans = []

    filename = fname.split("/")[2] + "_" + fname.split("/")[4]

    idx_spans.append(
        {
            "filename": filename,
            "id": [v.id for k, v in _textbounds.items()],
            "type": [v.type_ for k, v in _textbounds.items()],
            "start_idx": [v.start for k, v in _textbounds.items()],
            "end_idx": [v.end for k, v in _textbounds.items()],
        }
    )

    return idx_spans


def create_dir(save_dir):
    if os.path.isdir(save_dir):
        print("Directory to save results exists")
    else:
        print("Directory to save results doesn't exists")
        os.mkdir(save_dir)


def ann_to_df(_events):
    for k, v in _events.items():
        for _key, _value in v.arguments.items():
            tmp = {}

            # tmp['Filename'] = fname.split('/')[2]+'_'+fname.split('/')[4]
            tmp["Filename"] = fname.split("/")[4]
            tmp["FxEvent"] = v.id
            tmp["FxRelation"] = _key
            tmp["FxText"] = _textbounds[_value].text
            tmp["FxAttribute"] = _textbounds[_value].type_
            tmp["start_idx"] = _textbounds[_value].start
            tmp["end_idx"] = _textbounds[_value].end
            tmp["FxAttributeID"] = _value
            if _key != "Fracture":
                tmp["FxAttributeValue"] = _attributes[_value].value
                df.append(tmp)

    return df
