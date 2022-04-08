import os
import re
import json
import glob
import random
from lxml import etree
from bs4 import BeautifulSoup
from dataclasses import dataclass
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from nltk import sent_tokenize, wordpunct_tokenize
from transformers import (BertTokenizer, RobertaTokenizer,
                          XLMRobertaTokenizer, PreTrainedTokenizer,
                          AutoTokenizer)
from argparse import ArgumentParser, Namespace


ERE_V1 = 'LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V1'
ERE_V2 = 'LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2'
ERE_R2_V2 = 'LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2'
ERE_PARL_V2 = 'LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2'
SPANISH = 'LDC2015E107_DEFT_Rich_ERE_Spanish_Annotation_V2'

WRAPPED_DOCS = {'XIN_ENG_20101125.0137'}
DOCS_TO_REVISE_SENT = {'NYT_ENG_20131225.0200',
                       'NYT_ENG_20130716.0217',
                       '4fbb1eec7dfd5c2fefb94a2d873ddfa5',
                       '1f60eb9697e240af089b134b69c2042d',
                       'XIN_SPA_20050402.0105',
                       'APW_SPA_19980914.0097'}

EVENT_TYPE_WEIGHT = {
    'conflict': 5,
    'justice': 4,
    'life': 3,
}

relation_type_mapping = {
    'orgaffiliation': 'ORG-AFF',
    'personalsocial': 'PER-SOC',
    'physical': 'PHYS',
    'generalaffiliation': 'GEN-AFF',
    'partwhole': 'PART-WHOLE',
}
relation_subtype_mapping = {
    'opra': 'OPRA',
    'subsidiary': 'Subsidiary',
    'business': 'Business',
    'membership': 'Membership',
    'founder': 'Founder',
    'studentalum': 'StudentAlum',
    'employmentmembership': 'EmploymentMembership',
    'family': 'Family',
    'unspecified': 'Unspecified',
    'ownership': 'Ownership',
    'investorshareholder': 'InvestorShareholder',
    'orgheadquarter': 'OrgHeadquarter',
    'locatednear': 'LocatedNear',
    'more': 'MORE',
    'resident': 'Resident',
    'orglocationorigin': 'OrgLocationOrigin',
    'leadership': 'Leadership'
}

event_type_mapping = {
    'business:declarebankruptcy': 'Business:Declare-Bankruptcy',
    'business:endorg': 'Business:End-Org',
    'business:mergeorg': 'Business:Merge-Org',
    'business:startorg': 'Business:Start-Org',
    'conflict:attack': 'Conflict:Attack',
    'conflict:demonstrate': 'Conflict:Demonstrate',
    'contact:broadcast': 'Contact:Broadcast',
    'contact:contact': 'Contact:Contact',
    'contact:correspondence': 'Contact:Correspondence',
    'contact:meet': 'Contact:Meet',
    'justice:acquit': 'Justice:Acquit',
    'justice:appeal': 'Justice:Appeal',
    'justice:arrestjail': 'Justice:Arrest-Jail',
    'justice:chargeindict': 'Justice:Charge-Indict',
    'justice:convict': 'Justice:Convict',
    'justice:execute': 'Justice:Execute',
    'justice:extradite': 'Justice:Extradite',
    'justice:fine': 'Justice:Fine',
    'justice:pardon': 'Justice:Pardon',
    'justice:releaseparole': 'Justice:Release-Parole',
    'justice:sentence': 'Justice:Sentence',
    'justice:sue': 'Justice:Sue',
    'justice:trialhearing': 'Justice:Trial-Hearing',
    'life:beborn': 'Life:Be-Born',
    'life:die': 'Life:Die',
    'life:divorce': 'Life:Divorce',
    'life:injure': 'Life:Injure',
    'life:marry': 'Life:Marry',
    'manufacture:artifact': 'Manufacture:Artifact',
    'movement:transportartifact': 'Movement:Transport-Artifact',
    'movement:transportperson': 'Movement:Transport-Person',
    'personnel:elect': 'Personnel:Elect',
    'personnel:endposition': 'Personnel:End-Position',
    'personnel:nominate': 'Personnel:Nominate',
    'personnel:startposition': 'Personnel:Start-Position',
    'transaction:transaction': 'Transaction:Transaction',
    'transaction:transfermoney': 'Transaction:Transfer-Money',
    'transaction:transferownership': 'Transaction:Transfer-Ownership',
}

role_type_mapping = {
    'victim': 'Victim',
    'attacker': 'Attacker',
    'person': 'Person',
    'plaintiff': 'Plaintiff',
    'audience': 'Audience',
    'destination': 'Destination',
    'prosecutor': 'Prosecutor',
    'target': 'Target',
    'origin': 'Origin',
    'recipient': 'Recipient',
    'beneficiary': 'Beneficiary',
    'adjudicator': 'Adjudicator',
    'thing': 'Thing',
    'giver': 'Giver',
    'defendant': 'Defendant',
    'entity': 'Entity',
    'org': 'Org',
    'agent': 'Agent',
    'place': 'Place',
    'artifact': 'Artifact',
    'instrument': 'Instrument'
}


def mask_escape(text: str) -> str:
    """"Replaces escaped characters with rare sequences.

    Args:
        text (str): text to mask.
    
    Returns:
        str: masked string.
    """
    return text.replace('&amp;', 'ҪҪҪҪҪ').replace('&lt;', 'ҚҚҚҚ').replace('&gt;', 'ҺҺҺҺ')


def unmask_escape(text: str) -> str:
    """Replaces masking sequences with the original escaped characters.

    Args:
        text (str): masked string.
    
    Returns:
        str: unmasked string.
    """
    return text.replace('ҪҪҪҪҪ', '&amp;').replace('ҚҚҚҚ', '&lt;').replace('ҺҺҺҺ', '&gt;')


def recover_escape(text: str) -> str:
    """Converts named character references in the given string to the corresponding
    Unicode characters. I didn't notice any numeric character references in this
    dataset.

    Args:
        text (str): text to unescape.
    
    Returns:
        str: unescaped string.
    """
    return text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')


def sentence_tokenize(sentence: Tuple[int, int, str],
                      language: str = 'english') -> List[Tuple[int, int, str]]:
    """Split a sentence and adds offsets.

    Args:
        sentence (Tuple[int, int, str]): a sentence tuple consisting of start_offset,
            end_offset, and sentence text.
        language (str, optional): sentence language.
    
    Returns:
        List[Tuple[int, int, str]]: a list of sentence tuples.
    """
    start, end, text = sentence
    sents = sent_tokenize(text, language='english')

    last = 0
    sents_ = []
    for sent in sents:
        index = text[last:].find(sent)
        if index == -1:
            print(text, sent)
        else:
            sents_.append((last + index + start, last + index + len(sent) + start, sent))
        last += index + len(sent)
    return(sents_)


@dataclass
class Span:
    start: int
    end: int
    text: str

    def __post_init__(self):
        self.start = int(self.start)
        self.end = int(self.end)
        self.text = self.text.replace('\n', ' ')

    def char_offsets_to_token_offsets(self, tokens: List[Tuple[int, int, str]]):
        """Converts self.start and self.end from character offsets to token
        offsets.

        Args:
            tokens (List[Tuple[int, int, str]]): a list of token tuples. Each
                item in the list is a triple (start_offset, end_offset, text).
        """
        start_ = end_ = -1
        for i, (s, e, _) in enumerate(tokens):
            if s == self.start:
                start_ = i
            if e == self.end:
                end_ = i + 1
        if start_ == -1 or end_ == -1 or start_ > end_:
            raise ValueError('Failed to update offsets for {}-{}:{} in {}'.format(
                self.start, self.end, self.text, tokens))
        self.start, self.end = start_, end_

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'text': recover_escape(self.text),
            'start': self.start,
            'end': self.end
        }

    def remove_space(self):
        """Removes heading and trailing spaces in the span text."""
        # heading spaces
        text = self.text.lstrip(' ')
        self.start += len(self.text) - len(text)
        # trailing spaces
        after_text = text.rstrip(' ')
        #self.end = self.start + len(text)
        #self.text = text
        self.end += len(after_text) - len(text)
        self.text = after_text

    def copy(self):
        """Makes a copy of itself.

        Returns:
            Span: a copy of itself."""
        return Span(self.start, self.end, self.text)

@dataclass
class Entity(Span):
    entity_id: str
    mention_id: str
    entity_type: str
    mention_type: str

    def copy(self):
        return Entity(self.start, self.end, self.text,
            self.entity_id, self.mention_id, self.entity_type,
            self.mention_type)

    def to_dict(self, sent_id: str = None) -> Dict[str, Any]:
        """Converts instance variables to a dict.

        Args:
            sent_id (str): Sentence ID.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        if sent_id:
            entity_id = '{}-{}-{}'.format(sent_id,
                                          self.entity_id.split('-')[-1],
                                          self.mention_id.split('-')[-1])
        else:
            entity_id = '{}-{}'.format(self.entity_id.split('-')[-1],
                                       self.mention_id.split('-')[-1])
        return {
            'entity_id': entity_id,
            'entity_type': self.entity_type,
            'mention_type': self.mention_type,
            'start': self.start, 
            'end': self.end,
            'text': recover_escape(self.text)
        }

@dataclass
class Entity_whole:
    entity_id: str
    entity_type: str
    entity_mentions: List[Entity]

    def __str__(self, sent_id: str = None):
        return self.to_dict(sent_id)

    def to_dict(self, sent_id: str = None) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict: a dict of instance variables.
        """
        if sent_id:
            entity_id = '{}-{}'.format(sent_id,
                                          self.entity_id.split('-')[-1])
        else:
            entity_id = '{}'.format(self.entity_id.split('-')[-1])
        entity_dict = {
            'entity_id': entity_id,
            'entity_type': self.entity_type,
            'entity_mentions': [e.to_dict(sent_id) for e in self.entity_mentions]
        }
        return entity_dict

    def copy(self):
        """Makes a copy of itself.
        """
        return Entity_whole(self.entity_id, self.entity_type, self.entity_mentions)

@dataclass
class RelationArgument:
    entity_id: str
    mention_id: str
    role: str
    text: str

    def to_dict(self, sent_id:str = None) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        if sent_id:
            entity_id = '{}-{}-{}'.format(sent_id,
                                          self.entity_id.split('-')[-1],
                                          self.mention_id.split('-')[-1])
        else:
            entity_id = '{}-{}'.format(self.entity_id.split('-')[-1],
                                       self.mention_id.split('-')[-1])

        return {
            'entity_id': entity_id,
            'role': self.role,
            'text': recover_escape(self.text)
        }


@dataclass
class Relation:
    relation_id: str
    mention_id: str
    relation_type: str
    relation_subtype: str
    arg1: RelationArgument
    arg2: RelationArgument

    def to_dict(self, sent_id: str = None) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        if sent_id:
            relation_id = '{}-{}-{}'.format(sent_id,
                                            self.relation_id.split('-')[-1],
                                            self.mention_id.split('-')[-1])
        else:
            relation_id = '{}-{}'.format(self.relation_id.split('-')[-1],
                                         self.mention_id.split('-')[-1])

        return {
            'relation_id': relation_id,
            'relation_type': self.relation_type,
            'relation_subtype': self.relation_subtype,
            'arg1': self.arg1.to_dict(sent_id),
            'arg2': self.arg2.to_dict(sent_id)
        }


@dataclass
class EventArgument:
    entity_id: str
    mention_id: str
    role: str
    text: str

    def to_dict(self, sent_id: str = None) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        if sent_id:
            entity_id = '{}-{}-{}'.format(sent_id,
                                          self.entity_id.split('-')[-1],
                                          self.mention_id.split('-')[-1])
        else:
            entity_id = '{}-{}'.format(self.entity_id.split('-')[-1],
                                       self.mention_id.split('-')[-1])

        return {
            'entity_id': entity_id,
            'role': self.role,
            'text': recover_escape(self.text),
        }


@dataclass
class Event:
    event_id: str
    mention_id: str
    event_type: str
    event_subtype: str
    trigger: Span
    arguments: List[EventArgument]

    def copy(self):
        return Event(self.event_id, self.mention_id, self.event_type,
                self.event_subtype, self.trigger.copy(), self.arguments)

    def to_dict(self, sent_id: str = None) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        if sent_id:
            event_id = '{}-{}-{}'.format(sent_id,
                                         self.event_id.split('-')[-1],
                                         self.mention_id.split('-')[-1])
        else:
            event_id = '{}-{}'.format(self.event_id.split('-')[-1],
                                      self.mention_id.split('-')[-1])
        return {
            'event_id': event_id,
            'event_type': self.event_type,
            'event_subtype': self.event_subtype,
            'trigger': self.trigger.to_dict(),
            'arguments': [arg.to_dict(sent_id) for arg in self.arguments]
        }

@dataclass
class EventArgument_whole:
    entity_id: str
    role: str

    def to_dict(self, sent_id: str = None) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        if sent_id:
            entity_id = '{}-{}'.format(sent_id, self.entity_id.split('-')[-1])
        else:
            entity_id = '{}'.format(self.entity_id.split('-')[-1])
        return {
            'entity_id': entity_id,
            'role': self.role,
        }

@dataclass
class Event_whole:
    event_id: str
    event_type: str
    event_subtype: str
    arguments: List[EventArgument_whole]
    event_mentions: List[Event]

    def to_dict(self, sent_id: str = None) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        if sent_id:
            event_id = '{}-{}'.format(sent_id,
                                         self.event_id.split('-')[-1])
        else:
            event_id = '{}'.format(self.event_id.split('-')[-1])
        return {
            'event_id': event_id,
            'event_type': self.event_type,
            'event_subtype': self.event_subtype,
            'arguments': [arg.to_dict(sent_id) for arg in self.arguments],
            'event_mentions': [e.to_dict(sent_id) for e in self.event_mentions]
        }       



@dataclass
class Sentence(Span):
    doc_id: str
    sent_id: str
    tokens: List[str]
    entities: List[Entity]
    relations: List[Relation]
    events: List[Event]
    sent_starts: List[int]
    entity_cluster: List[Entity_whole]
    event_cluster: List[Event_whole]

    def to_dict(self) -> Dict[str, Any]:
        """Converts instance variables to a dict.
        
        Returns:
            Dict[str, Any]: a dict of instance variables.
        """
        return {
            'doc_id': self.doc_id,
            'wnd_id': self.sent_id,
            'tokens': [recover_escape(t) for t in self.tokens],
            'entities': [entity.to_dict(self.sent_id) for entity in self.entities],
            'relations': [relation.to_dict(self.sent_id) for relation in self.relations],
            'events': [event.to_dict(self.sent_id) for event in self.events],
            'sentence_starts': self.sent_starts,
            'entity_cluster': [entity.to_dict(self.sent_id) for entity in self.entity_cluster],
            'event_cluster': [event.to_dict(self.sent_id) for event in self.event_cluster],
            'start': self.start,
            'end': self.end,
            'text': recover_escape(self.text).replace('\t', ' ')
        }


@dataclass
class Document:
    doc_id: str
    sentences: List[Sentence]


def process_wrapped_text(text: str) -> str:
    """Handles wrapped text in some documents by replacing linebreaks between a
    a pair of <p> and </p> tags with spaces.

    Args:
        text (str): text string to process.
    
    Returns:
        str: processed text string.
    """
    segments = text.split('\n')
    segments_new = []
    in_p = False
    for segment in segments:
        if in_p:
            if segment == '</P>':
                segments_new.append(segment)
                in_p = False
            else:
                if segments_new[-1]:
                    segments_new[-1] += ' '
                segments_new[-1] += segment
        else:
            segments_new.append(segment)
            if segment == '<P>':
                segments_new.append('')
                in_p = True
    return '\n'.join(segments_new)


def revise_sentences(sentences: List[Tuple[int, int, str]],
                     doc_id: str) -> List[Tuple[int, int, str]]:
    """Automatic sentence tokenization may have errors for a few documents. This
    function joins sentences that are mistakenly split into multiple segments.

    Args:
        sentences (List[Tuple[int, int, str]]): a list of sentence tuples.
        doc_id (str): document ID.

    Returns:
        List[Tuple[int, int, str]]: a list of revised sentence tuples.
    """
    sentences_ = []
    if doc_id == 'NYT_ENG_20131225.0200':
        for sentence_idx, (start, end, text) in enumerate(sentences):
            if start == 2089:
                next_start, next_end, next_text = sentences[sentence_idx + 1]
                space = ' ' * (next_start - end)
                sentences_.append((start, next_end, text + space + next_text))
            elif start == 2163:
                continue
            else:
                sentences_.append((start, end, text))
    elif doc_id == 'NYT_ENG_20130716.0217':
        for sentence_idx, (start, end, text) in enumerate(sentences):
            if start == 1781:
                next_start, next_end, next_text = sentences[sentence_idx + 1]
                space = ' ' * (next_start - end)
                sentences_.append((start, next_end, text + space + next_text))
            elif start == 1932:
                continue
            else:
                sentences_.append((start, end, text))
    elif doc_id == '4fbb1eec7dfd5c2fefb94a2d873ddfa5':
        for sentence_idx, (start, end, text) in enumerate(sentences):
            if start == 4540:
                next_start, next_end, next_text = sentences[sentence_idx + 1]
                space = ' ' * (next_start - end)
                sentences_.append((start, next_end, text + space + next_text))
            elif start == 4543:
                continue
            else:
                sentences_.append((start, end, text))
    elif doc_id == '1f60eb9697e240af089b134b69c2042d':
        for sentence_idx, (start, end, text) in enumerate(sentences):
            if start == 5373:
                next_start, next_end, next_text = sentences[sentence_idx + 1]
                space = ' ' * (next_start - end)
                sentences_.append((start, next_end, text + space + next_text))
            elif start == 5391:
                continue
            else:
                sentences_.append((start, end, text))
    elif doc_id == 'XIN_SPA_20050402.0105':
        for sentence_idx, (start, end, text) in enumerate(sentences):
            if start == 58:
                next_start, next_end, next_text = sentences[sentence_idx + 1]
                space = ' ' * (next_start - end)
                sentences_.append((start, next_end, text + space + next_text))
            elif start == 113:
                continue
            else:
                sentences_.append((start, end, text))
    elif doc_id == 'APW_SPA_19980914.0097':
        cur_start, cur_end, cur_text = 0, 0, ''
        for start, end, text in sentences:
            if '--' in text:
                sentences_.append((cur_start, cur_end, cur_text))
                sentences_.append((start, end, text))
                cur_start, cur_end, cur_text = 0, 0, ''
            else:
                if cur_end == 0:
                    cur_start, cur_end, cur_text = start, end, text
                else:
                    cur_text += ' ' * (start - cur_end) + text
                    cur_end = end
        if cur_text:
            sentences_.append((cur_start, cur_end, cur_text))
    
    return sentences_


def read_source_file(path: str,
                     language: str = 'english',
                     wrapped: bool = False,
                    ) -> List[Tuple[int, int, str]]:
    """Reads text file.

    Args:
        path (str): path to the text file.
        language (str, optional): document language. Defaults to 'english'.
        wrapped (bool, optional): if the document is wrapped. Defaults to False.

    Returns:
        List[Tuple[int, int, str]]: a list of sentence tuples.
    """
    data = open(path, 'r', encoding='utf-8').read()

    if wrapped:
        data = process_wrapped_text(data)
    # data = data.replace('\n<a', ' <a').replace('</a>\n', '</a> ')

    min_offset = max(0, data.find('<HEADLINE>'))

    intag = False
    in_a = False
    linebreak = True
    sentences = []
    start = end = 0
    sentence = ''
    for i, c in enumerate(data):
        if c == '<':
            intag = True
            if (len(data) > i + 2
                 and (data[i + 1: i + 3] == 'a '
                 or data[i + 1: i + 3] == 'a>'
                 or data[i + 1: i + 3] == '/a')
                 ):
                 in_a = True
            if in_a:
                sentence += ' '
                end = i + 1
            else:
                # linebreak = False
                if sentence and start >= min_offset:
                    sentences.append((start, end, sentence))
                    start = end = i + 1
                sentence = ''
        elif c == '>':
            intag = False
            if in_a:
                sentence += ' '
                end = i + 1
            else:
                start = end = i + 1
            in_a = False
        elif not intag and linebreak:
            if c == '\n':
                if sentence:
                    if start >= min_offset:
                        sentences.append((start, end, sentence))
                    sentence = ''
                start = end = i + 1
            else:
                sentence += c
                end = i + 1
        elif in_a:
            sentence += ' '
            end = i + 1
        if c == '\n':
            linebreak = True
            start = end = i + 1
    if sentence:
        if start >= min_offset:
            sentences.append((start, end, sentence))
    # for s, e, t in sentences:
    #     if t != data[s:e]:
    #         print(t, data[s:e])

    # Re-tokenize sentences
    sentences_ = []
    for sent in sentences:
        if not sent[-1].startswith('http') and '</a>' not in sent[-1]:
            sentences_.extend(sentence_tokenize(sent, language=language))
    return sentences_


def read_annotation(path: str
                   ) -> (str, str, List[Entity], List[Relation], List[Event]):
    """Reads annotation file.

    Args:
        path (str): path to the annotation file.

    Returns:
        doc_id (str): document ID.
        source_type (str): type of the source of the document.
        entity_list (List[Entity]): a list of Entity objects.
        relation_list (List[Relation]) a list of Relation objects.
        event_list (List[Event]): a list of Event objectss.
    """
    data = open(path, 'r', encoding='utf-8').read()

    soup = BeautifulSoup(data, 'lxml')

    # metadata
    root = soup.find('deft_ere')
    doc_id = root['doc_id']
    source_type = root['source_type']

    # entities
    entity_list = []
    entity_clusters = []
    entities_node = root.find('entities')
    if entities_node:
        for entity_node in entities_node.find_all('entity'):
            entity_id = entity_node['id']
            entity_type = entity_node['type']
            entity_cluster = []
            for entity_mention_node in entity_node.find_all('entity_mention'):
                mention_id = entity_mention_node['id']
                mention_type = entity_mention_node['noun_type']
                if mention_type == 'NOM':
                    mention_offset = int(entity_mention_node.find('nom_head')['offset'])
                    mention_length = int(entity_mention_node.find('nom_head')['length'])
                    mention_text = entity_mention_node.find('nom_head').text
                else:
                    mention_offset = int(entity_mention_node['offset'])
                    mention_length = int(entity_mention_node['length'])
                    mention_text = entity_mention_node.find('mention_text').text
                entity_list.append(Entity(
                    entity_id=entity_id, mention_id=mention_id,
                    entity_type=entity_type, mention_type=mention_type,
                    start=mention_offset, end=mention_offset + mention_length,
                    text=mention_text
                ))
                entity_cluster.append(Entity(
                    entity_id=entity_id, mention_id=mention_id,
                    entity_type=entity_type, mention_type=mention_type,
                    start=mention_offset, end=mention_offset + mention_length,
                    text=mention_text
                ))
            entity_clusters.append(Entity_whole(entity_id, entity_type, entity_cluster))
    fillers_node = root.find('fillers')
    if fillers_node:
        for filler_node in fillers_node.find_all('filler'):
            entity_id = filler_node['id']
            entity_type = filler_node['type']
            if entity_type == 'weapon':
                entity_type = 'WEA'
            elif entity_type == 'vehicle':
                entity_type = 'VEH'
            else:
                continue
            mention_offset = int(filler_node['offset'])
            mention_length = int(filler_node['length'])
            mention_text = filler_node.text
            entity_list.append(
                Entity(
                    entity_id=entity_id, mention_id=entity_id,
                    entity_type=entity_type, mention_type='NOM',
                    start=mention_offset, end=mention_offset + mention_length,
                    text=mention_text
                )
            )

    # relations
    relation_list = []
    relations_node = root.find('relations')
    if relations_node:
        for relation_node in relations_node.find_all('relation'):
            relation_id = relation_node['id']
            relation_type = relation_node['type']
            relation_subtype = relation_node['subtype']
            for relation_mention_node in relation_node.find_all('relation_mention'):
                mention_id = relation_mention_node['id']
                arg1 = relation_mention_node.find('rel_arg1')
                arg2 = relation_mention_node.find('rel_arg2')
                if arg1 and arg2:
                    if arg1.has_attr('entity_id'):
                        arg1_entity_id = arg1['entity_id']
                        arg1_mention_id = arg1['entity_mention_id']
                    else:
                        arg1_entity_id = arg1['filler_id']
                        arg1_mention_id = arg1['filler_id']
                    arg1_role = arg1['role']
                    arg1_text = arg1.text
                    if arg2.has_attr('entity_id'):
                        arg2_entity_id = arg2['entity_id']
                        arg2_mention_id = arg2['entity_mention_id']
                    else:
                        arg2_entity_id = arg2['filler_id']
                        arg2_mention_id = arg2['filler_id']
                    arg2_role = arg2['role']
                    arg2_text = arg2.text
                    relation_list.append(Relation(
                        relation_id=relation_id, mention_id=mention_id,
                        relation_type=relation_type,
                        relation_subtype=relation_subtype,
                        arg1=RelationArgument(entity_id=arg1_entity_id,
                                              mention_id=arg1_mention_id,
                                              role=arg1_role,
                                              text=arg1_text),
                        arg2=RelationArgument(entity_id=arg2_entity_id,
                                              mention_id=arg2_mention_id,
                                              role=arg2_role,
                                              text=arg2_text)))

    # events
    event_list = []
    event_clusters = []
    events_node = root.find('hoppers')
    if events_node:
        for event_node in events_node.find_all('hopper'):
            event_cluster = []
            event_arguments = []
            event_id = event_node['id']
            for event_mention_node in event_node.find_all('event_mention'):
                trigger = event_mention_node.find('trigger')
                trigger_offset = int(trigger['offset'])
                trigger_length = int(trigger['length'])
                arguments = []
                for arg in event_mention_node.find_all('em_arg'):
                    if arg['realis'] == 'false':
                        continue
                    if arg.has_attr('entity_id'):
                        arguments.append(EventArgument(
                            entity_id=arg['entity_id'],
                            mention_id=arg['entity_mention_id'],
                            role=arg['role'],
                            text=arg.text))
                        if (arg['entity_id'], arg['role']) not in event_arguments:
                            event_arguments.append((arg['entity_id'], arg['role']))
                    elif arg.has_attr('filler_id'):
                        arguments.append(EventArgument(
                            entity_id=arg['filler_id'],
                            mention_id=arg['filler_id'],
                            role=arg['role'],
                            text=arg.text
                        ))
                        if (arg['filler_id'], arg['role']) not in event_arguments:
                            event_arguments.append((arg['filler_id'], arg['role']))
                event_list.append(Event(
                    event_id=event_id,
                    mention_id=event_mention_node['id'],
                    event_type=event_mention_node['type'],
                    event_subtype=event_mention_node['subtype'],
                    trigger=Span(start=trigger_offset,
                                         end=trigger_offset + trigger_length,
                                         text=trigger.text),
                    arguments=arguments))
                event_cluster.append(Event(
                    event_id=event_id,
                    mention_id=event_mention_node['id'],
                    event_type=event_mention_node['type'],
                    event_subtype=event_mention_node['subtype'],
                    trigger=Span(start=trigger_offset,
                                         end=trigger_offset + trigger_length,
                                         text=trigger.text),
                    arguments=arguments))
            event_arguments = [EventArgument_whole(ent_id, ent_role) for ent_id, ent_role in event_arguments]
            event_clusters.append(Event_whole(event_id, event_cluster[0].event_type, event_cluster[0].event_subtype, event_arguments, event_cluster))

    return doc_id, source_type, entity_list, relation_list, event_list, entity_clusters, event_clusters


def clean_entities(entities: List[Entity],
                   sentences: List[List[Tuple[int, int, str]]],
                   ) -> List[List[Entity]]:
    """Cleans entities and assigns them to the corresponding sentences.

    Args:
        entities (List[Entity]): a list of Entity objects.
        sentences (List[List[Tuple[int, int, str]]]): a list of window of sentences.

    Returns:
        List[List[Entity]]: a list of lists of Entity objects. The i-th list
            represents Entity objects in the i-th window.
    """
    sentence_entities = [[] for _ in range(len(sentences))]
    for entity in entities:
        start, end = entity.start, entity.end
        for i, sentence in enumerate(sentences): # each window
            flag = False
            for s, e, text in sentence:
                if start >= s and end <= e:
                    entity.remove_space()
                    # put this entity into this window
                    flag = True
                    break
            if flag:
                sentence_entities[i].append(entity)


    # Remove overlapping entities
    # TODO: we keep it becauce of fair comparison to OneIE, but this should be prevented for golds.
    sentence_entities_ = [[] for _ in range(len(sentences))]
    for i, entities in enumerate(sentence_entities):
        if not entities:
            continue
        # Prefer longer entities
        entities.sort(key=lambda x: (x.end - x.start), reverse=True)
        chars = [0] * max([x.end for x in entities])
        for entity in entities:
            overlap = False
            for j in range(entity.start, entity.end):
                if chars[j] == 1:
                    overlap = True
                    break
            if overlap:
                continue
            else:
                chars[entity.start:entity.end] = [1] * (entity.end - entity.start)
                sentence_entities_[i].append(entity)
        sentence_entities_[i].sort(key=lambda x: x.start)

    return sentence_entities_


def detect_multievent_triggers(sentence_events: List[List[Event]]) -> List[int]:
    """Find sentences that contain multi-event triggers.

    Args:
        sentence_events (List[List[Event]]): a list of lists of Event objects.

    Returns:
        List[int]: a list of sentence indices.
    """
    multievent_trigger_sents = []
    for sent_idx, events in enumerate(sentence_events):
        trigger_event_types = defaultdict(set)
        for event in events:
            event_type = event.event_type
            key = (event.trigger.start, event.trigger.end)
            if event_type in trigger_event_types[key]:
                multievent_trigger_sents.append(sent_idx)
                break
            else:
                trigger_event_types[key].add(event_type)
    return multievent_trigger_sents


def clean_events(events: List[Event],
                 sentence_entities: List[List[Entity]],
                 sentences: List[List[Tuple[int, int, str]]]) -> List[List[Event]]:
    """Cleans events and assigns them to the corresponding sentences.

    Args:
        events (List[Event]): a list of Event objects.
        sentence_entities (List[List[Entity]]): a list of lists of Entity
            objects. The output of the `clean_entities`.
        sentences (List[List[Tuple[int, int, str]]]): a list of window of sentences.

    Returns:
        List[List[Event]]: a list of lists of Event objects. The i-th list
            represents Event objects in the i-th window.
    """
    sentence_events = [[] for _ in range(len(sentences))]
    for event in events:
        start, end = event.trigger.start, event.trigger.end
        for i, sentence in enumerate(sentences): # each window
            for s, e, _ in sentence:
                flag = False
                if start >= s and end <= e:
                    event.trigger.remove_space()
                    flag = True
                    break
            if flag:
                # Clean the argument list
                arguments = []
                entities = sentence_entities[i]
                for argument in event.arguments:
                    entity_id = argument.entity_id
                    mention_id = argument.mention_id
                    for entity in entities:
                        if (entity.entity_id == entity_id and
                                entity.mention_id == mention_id):
                            arguments.append(argument)
                event_ = Event(event_id=event.event_id,
                            mention_id=event.mention_id,
                            event_type=event.event_type,
                            event_subtype=event.event_subtype,
                            trigger=event.trigger.copy(),
                            arguments=arguments
                            )
                sentence_events[i].append(event_)
                

    multievent_trigger_sents = detect_multievent_triggers(sentence_events)
    # Remove overlapping events
    # TODO: we keep it becauce of fair comparison to OneIE, but this should be prevented for golds.
    sentence_events_ = [[] for _ in range(len(sentences))]
    for i, events in enumerate(sentence_events):
        if not events:
            continue
        events.sort(key=lambda x: (EVENT_TYPE_WEIGHT.get(x.event_type, 0),
                                   x.trigger.end - x.trigger.start),
                    reverse=True)
        chars = [0] * max([x.trigger.end for x in events])
        for event in events:
            overlap = False
            for j in range(event.trigger.start, event.trigger.end):
                if chars[j] == 1:
                    overlap = True
                    break
            if overlap:
                continue
            else:
                chars[event.trigger.start:event.trigger.end] = \
                    [1] * (event.trigger.end - event.trigger.start)
                sentence_events_[i].append(event)
        sentence_events_[i].sort(key=lambda x: x.trigger.start)

    return sentence_events_, multievent_trigger_sents


def clean_relations(relations: List[Relation],
                    sentence_entities: List[List[Entity]],
                    sentences: List[List[Tuple[int, int, str]]]
                    ) -> List[List[Relation]]:
    """Cleans relations and assigns them to the corresponding sentence.

    Args:
        relations (List[Relation]): a list of Relation objects.
        sentence_entities (List[List[Entity]]): a list of lists of Entity
            objects. The output of the `clean_entities` function.
        sentences (List[List[Tuple[int, int, str]]]): a list of sentence tuples.

    Returns:
        List[List[Relation]]: a list of lists of Relation objects. The i-th list
            represents Relation objects in the i-th window.
    """
    sentence_relations = [[] for _ in range(len(sentences))]
    for relation in relations:
        keep = False
        entity_id_1, mention_id_1 = relation.arg1.entity_id, relation.arg1.mention_id
        entity_id_2, mention_id_2 = relation.arg2.entity_id, relation.arg2.mention_id
        for i, entities in enumerate(sentence_entities):
            arg1_in_sent = arg2_in_sent = False
            for entity in entities:
                if entity.entity_id == entity_id_1 and entity.mention_id == mention_id_1:
                    arg1_in_sent = True
                if entity.entity_id == entity_id_2 and entity.mention_id == mention_id_2:
                    arg2_in_sent = True
            if arg1_in_sent and arg2_in_sent:
                sentence_relations[i].append(relation)
                keep = True
                break
            elif arg1_in_sent != arg2_in_sent:
                # Stop searching because we find only one entity in the current
                # sentence.
                break
    return sentence_relations

def process_entity_cluster(entity_clusters: List[Entity_whole],
                           sentence_entities: List[List[Entity]],
                           sentences: List[List[Tuple[int, int, str]]]
                           ) -> List[List[Entity_whole]]:
    """Cleans entity_cluster and splits them into lists

    Args:
        entity_clusters (List[Entity_whole]): a list of Entity_whole instances.
        entities (List[List[Entity]]): a list of sentence entity list.
        sentences (List[List[Tuple[int, int, str]]]): a list of window of sentences.

    Returns:
        List[List[Entity_whole]]: a list of sentence Entity_whole lists.
    """
    sentence_entity_cluster = [[] for _ in range(len(sentences))]

    for entity_cluster in entity_clusters:
        for i, sentence_entity in enumerate(sentence_entities):
            within_entities = [ent.copy() for ent in sentence_entity if ent.entity_id == entity_cluster.entity_id]
            if any(within_entities):
                sentence_entity_cluster[i].append(Entity_whole(entity_cluster.entity_id, entity_cluster.entity_type, within_entities))
    return sentence_entity_cluster

def process_event_cluster(event_clusters: List[Event_whole],
                        sentence_events: List[List[Event]],
                        sentences: List[List[Tuple[int, int, str]]]
                        ) -> List[List[Event_whole]]:
    """Cleans and assigns event cluster.

    Args:
        event_clusters (List[Event_whole]): A list of event cluster objects
        sentence_events (List[List[Event]]): A list of sentence event lists.
        sentences (List[List[Tuple[int, int, str]]]): a list of window of sentences.
    
    Returns:
        List[List[Event_whole]]: a list of sentence event cluster lists.
    """
    sentence_events_cluster = [[] for _ in range(len(sentences))]
    # Reconstruct event_whole based on the event mentions that within the sentence(window) span
    for event_cluster in event_clusters:
        for i, sentence_event in enumerate(sentence_events):
            within_events = [eve.copy() for eve in sentence_event if (eve.event_id == event_cluster.event_id)]
            if any(within_events):
                arguments = []
                for argu in event_cluster.arguments:
                    flag = False
                    for within_event in within_events:
                        for arg in within_event.arguments:
                            if arg.entity_id == argu.entity_id:
                                flag = True
                                break
                        if flag:
                            break
                    if flag:
                        arguments.append(argu)
                sentence_events_cluster[i].append(Event_whole(event_cluster.event_id, event_cluster.event_type,
                                                    event_cluster.event_subtype, arguments, within_events))

    return sentence_events_cluster

def tokenize(sentences: List[Tuple[int, int, str]],
             entities: List[Entity],
             events: List[Event],
             language: str = 'english'
             ) -> List[Tuple[int, int, str]]:
    """Tokenizes a sentence and makes sure entity and event spans are compatible
        with the tokenization result.

    Args:
        sentence (List[Tuple[int, int, str]]): a window of sentences.
        entities (List[Entity]): a list of Entity objects.
        events (List[Entity]): a list of Event objects.
        language (str, optional): sentence langauge.

    Returns:
        tokens (List[Tuple[int, int, str]]): a list of token tuples, where each 
            item consists of start_offset, end_offset, and text.
    """
    all_tokens = []
    token_starts = [0]
    for sentence in sentences:
        start, end, text = sentence
        text = mask_escape(text)
        # Split the sentence into chunks
        splits = {0, len(text)}
        for entity in entities:
            if entity.start >= start and entity.end <= end:
                splits.add(entity.start - start)
                splits.add(entity.end - start)
        for event in events:
            if event.trigger.start >= start and event.trigger.end <= end:
                splits.add(event.trigger.start - start)
                splits.add(event.trigger.end - start)
        splits = sorted(list(splits))
        chunks = [(splits[i], splits[i + 1], text[splits[i]:splits[i + 1]])
                for i in range(len(splits) - 1)]

        # Tokenize each chunk
        chunks = [(s, e, t, wordpunct_tokenize(t)) for s, e, t in chunks]

        # Merge chunks and add word offsets
        tokens = []
        for chunk_start, chunk_end, chunk_text, chunk_tokens in chunks:
            last = 0
            chunk_tokens_ = []
            for token in chunk_tokens:
                token_start = chunk_text[last:].find(token)
                if token_start == -1:
                    raise ValueError('Cannot find token {} in {}'.format(token, text))
                token_end = token_start + len(token)
                chunk_tokens_.append((token_start + start + last + chunk_start,
                                    token_end + start + last + chunk_start,
                                    unmask_escape(token)))
                last += token_end
            tokens.extend(chunk_tokens_)

        all_tokens.extend(tokens)
        token_starts.append(len(all_tokens))
    return all_tokens, token_starts

def sentence2window(sentences, window_size_):
    """Convert sentences list to windows

    Args:
        sentences (List[Tuple[int, int, str]])
        window_size : int

    Returns:
        List[List[Tuple[int, int, str]]]

    """
    if window_size_ > len(sentences):
        window_size = len(sentences)
    else:
        window_size = window_size_

    output = []
    for i in range(len(sentences)-window_size+1):
        output.append(sentences[i:i+window_size])
    return output

def extract(source_path: str,
            ere_path: str,
            doc_id: str,
            window_size: int = 1,
            language: str = 'english',
            discard_sentences_with_multievent_triggers: bool = True) -> Document:
    """Generates a Document object from a text file and an annotation file.

    Args:
        source_path (str): path to the source file.
        ere_path (str): path to the annotation file.
        doc_id (str): document ID.
        language (str, optional): document language. Defaults to 'english'.

    Returns:
        Document: a Document object that contains tokens, entities, relations,
            and events of a document.
    """
    wrapped = (doc_id in WRAPPED_DOCS
               or (language == 'spanish' and 'newswire' in source_path ))
    sentences = read_source_file(source_path,
                                          language=language,
                                          wrapped=wrapped)
    # Revise sentences
    if doc_id in DOCS_TO_REVISE_SENT:
        sentences = revise_sentences(sentences, doc_id)
    doc_id_, source_type, entities, relations, events, entity_cluster, event_cluster = read_annotation(ere_path)

    # concate windows
    sentences = sentence2window(sentences, window_size)

    # Remove entities and events out of extracted sentences
    sentence_entities = clean_entities(entities, sentences)
    sentence_events, multievent_trigger_sents = clean_events(
        events, sentence_entities, sentences)
    sentence_relations = clean_relations(relations, sentence_entities, sentences)

    # Process entity_cluster, event_cluster, relation(entity_base)
    sentence_entity_cluster = process_entity_cluster(entity_cluster, sentence_entities, sentences)
    sentence_event_cluster = process_event_cluster(event_cluster, sentence_events, sentences)  

    # Tokenization
    sentence_tokens = []
    sentence_starts = []
    for s, ent, evt in zip(sentences, sentence_entities, sentence_events):
        sent_tokens, sent_start = tokenize(s, ent, evt, language=language)
        sentence_tokens.append(sent_tokens)
        sentence_starts.append(sent_start)


    # Convert span character offsets to token index offsets
    sentence_objs = []
    for i, (tokens, entities, events, relations, sentence, entity_cluster, event_cluster, starts) in enumerate(zip(
            sentence_tokens, sentence_entities, sentence_events,
            sentence_relations, sentences, sentence_entity_cluster, sentence_event_cluster, sentence_starts)):
        wnd_id = '{}-{}'.format(doc_id, i)
        if (discard_sentences_with_multievent_triggers
            and i in multievent_trigger_sents):
            continue
        for entity in entities:
            entity.char_offsets_to_token_offsets(tokens)
        for entity in entity_cluster:
            for mention in entity.entity_mentions:
                mention.char_offsets_to_token_offsets(tokens)
        for event in events:
            event.trigger.char_offsets_to_token_offsets(tokens)
        for event in event_cluster:
            for mention in event.event_mentions:
                mention.trigger.char_offsets_to_token_offsets(tokens)
        sentence_objs.append(Sentence(
            doc_id=doc_id, sent_id=wnd_id,
            tokens=[t for _, _, t in tokens],
            entities=entities,
            relations=relations,
            events=events,
            sent_starts=starts,
            entity_cluster=entity_cluster,
            event_cluster=event_cluster,
            start=sentence[0][0],
            end=sentence[-1][1],
            text=' '.join([s[2] for s in sentence])
        ))
    return Document(doc_id=doc_id, sentences=sentence_objs)


def process_batch(input_dir, output_file, window_size=1, dataset='normal', language='english'):
    """Processes a batch of documents.

    Args:
        input_dir (str): path to the input directory.
        output_file (str): path to the output file.
        dataset (str, optional): dataset type. Defaults to 'normal'.
        language (str, optional): dataset language. Defaults to 'english'.
    """
    if dataset == 'normal':
        source_files = glob.glob(
            os.path.join(input_dir, 'source', 'cmptxt', '*.txt'))
    elif dataset == 'r2v2':
        source_files = glob.glob(os.path.join(input_dir, 'source', '*.txt'))
    elif dataset == 'parallel':
        source_files = glob.glob(
            os.path.join(input_dir, 'eng', 'translation', '*.txt'))
    elif dataset == 'spanish':
        source_files = glob.glob(
            os.path.join(input_dir, 'source', '**', '*.txt'))
    else:
        raise ValueError('Unknown dataset type: {}'.format(dataset))

    with open(output_file, 'w', encoding='utf-8') as w:
        for source_file in source_files:
            doc_id = os.path.basename(source_file).replace('.txt', '') \
                .replace('.cmp', '').replace('.mp', '')
            if dataset == 'normal':
                annotation_file = os.path.join(input_dir, 'ere', 'cmptxt',
                                               '{}.rich_ere.xml'.format(doc_id))
            elif dataset == 'r2v2':
                annotation_file = os.path.join(input_dir, 'ere',
                                               '{}.rich_ere.xml'.format(doc_id))
            elif dataset == 'parallel':
                annotation_file = os.path.join(input_dir, 'eng', 'ere',
                                               '{}.rich_ere.xml'.format(doc_id))
            elif dataset == 'spanish':
                annotation_file = (source_file.replace('/source/', '/ere/')
                                   .replace('.txt', '.rich_ere.xml'))
            doc = extract(source_file, annotation_file, doc_id, window_size, language)
            for sent in doc.sentences:
                w.write(json.dumps(sent.to_dict()) + '\n')


def ere_to_oneie(input_file: str,
                 output_file: str,
                 tokenizer: PreTrainedTokenizer):
    """Converts to OneIE format.

    Args:
        input_file (str): path to the input file.
        output_file (str): path to the output file.
        tokenizer (PreTrainedTokenizer): a tokenizer that converts tokens to
            word pieces.
    """
    skip_num = 0
    with open(input_file, 'r', encoding='utf-8') as r, \
        open(output_file, 'w', encoding='utf-8') as w:
        for line in r:
            inst = json.loads(line)
            # tokens
            tokens = inst['tokens']
            pieces = [tokenizer.tokenize(t) for t in tokens]
            token_lens = [len(x) for x in pieces]
            if 0 in token_lens:
                skip_num += 1
                continue
            pieces = [p for ps in pieces for p in ps]
            sentence = inst['text']
            # entities
            entity_mentions = []
            entity_text = {}
            for entity in inst['entities']:
                entity_mentions.append({
                    'id': entity['entity_id'],
                    'start': entity['start'],
                    'end': entity['end'],
                    'entity_type': entity['entity_type'],
                    'mention_type': entity['mention_type'],
                    'text': entity['text']
                })
                entity_text[entity['entity_id']] = entity['text']
            # relations
            relation_mentions = []
            for relation in inst['relations']:
                relation_type = relation_type_mapping[relation['relation_type']]
                relation_subtype = relation_subtype_mapping[
                    relation['relation_subtype']]
                relation_mentions.append({
                    'id': relation['relation_id'],
                    'relation_type': relation_type,
                    'relation_subtype': '{}:{}'.format(relation_type,
                                                       relation_subtype),
                    'arguments': [
                        {
                            'entity_id': relation['arg1']['entity_id'],
                            'role': 'Arg-1',
                            'text': entity_text[relation['arg1']['entity_id']]
                        },
                        {
                            'entity_id': relation['arg2']['entity_id'],
                            'role': 'Arg-2',
                            'text': entity_text[relation['arg2']['entity_id']]
                        },
                    ]
                })
            # events
            event_mentions = []
            for event in inst['events']:
                event_mentions.append({
                    'id': event['event_id'],
                    'event_type': event_type_mapping['{}:{}'.format(
                        event['event_type'], event['event_subtype'])],
                    'trigger': {
                        'start': event['trigger']['start'],
                        'end': event['trigger']['end'],
                        'text': event['trigger']['text']},
                    'arguments': [{
                        'entity_id': arg['entity_id'],
                        'text': entity_text[arg['entity_id']],
                        'role': role_type_mapping[arg['role']]
                    } for arg in event['arguments']]
                })

            # coreference
            corefs = []
            for entity in inst['entity_cluster']:
                if len(entity['entity_mentions']) > 1:
                    corefs.append({
                        'id': entity['entity_id'],
                        'entities': [{
                            'id': mention['entity_id'],
                            'text': mention['text'],
                            'entity_type': mention['entity_type'],
                            'mention_type': mention['mention_type'],
                            'start': mention['start'],
                            'end': mention['end']
                        } for mention in entity['entity_mentions']],
                        'entity_type': entity['entity_type']
                    })

            # event coreference
            event_corefs = []
            for event in inst['event_cluster']:
                if len(event['event_mentions']) > 1 :
                    event_corefs.append({
                        'id': event['event_id'],
                        'events': [{
                            'id': mention['event_id'],
                            'event_type': event_type_mapping['{}:{}'.format(
                                mention['event_type'], mention['event_subtype'])],
                            'trigger': {
                                'start': mention['trigger']['start'],
                                'end': mention['trigger']['end'],
                                'text': mention['trigger']['text']},
                            'arguments':[{
                                'entity_id': arg['entity_id'],
                                'text': entity_text[arg['entity_id']],
                                'role': role_type_mapping[arg['role']]
                            } for arg in mention['arguments']],
                        } for mention in event['event_mentions']],
                        'event_type': event_type_mapping['{}:{}'.format(
                            event['event_type'], event['event_subtype'])]
                    })

            w.write(json.dumps({
                'doc_id': inst['doc_id'],
                'wnd_id': inst['wnd_id'],
                'tokens': tokens,
                'pieces': pieces,
                'token_lens': token_lens,
                'sentence': inst['text'],
                'entity_mentions': entity_mentions,
                'relation_mentions': relation_mentions,
                'event_mentions': event_mentions,
                'entity_coreference': corefs,
                'event_coreference': event_corefs,
                'sentence_starts': inst['sentence_starts'][:-1]
            }) + '\n')
    print('#Skip: {}'.format(skip_num))


def split_data(input_file, output_dir, split_path):
    """Splits the input file into train/dev/test sets.

    Args:
        input_file (str): path to the input file.
        output_dir (str): path to the output directory.
        split_path (str): path to the split directory that contains three files,
            train.doc.txt, dev.doc.txt, and test.doc.txt . Each line in these
            files is a document ID.
    """
    print('Splitting the dataset into train/dev/test sets')
    train_docs, dev_docs, test_docs = set(), set(), set()
    # Load doc ids
    with open(os.path.join(split_path, 'train.doc.txt')) as r:
        train_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'dev.doc.txt')) as r:
        dev_docs.update(r.read().strip('\n').split('\n'))
    with open(os.path.join(split_path, 'test.doc.txt')) as r:
        test_docs.update(r.read().strip('\n').split('\n'))
    
    real_train, real_dev, real_test = set(), set(), set()
    # Split the dataset
    with open(input_file, 'r', encoding='utf-8') as r, \
        open(os.path.join(output_dir, 'train.oneie.json'), 'w') as w_train, \
        open(os.path.join(output_dir, 'dev.oneie.json'), 'w') as w_dev, \
        open(os.path.join(output_dir, 'test.oneie.json'), 'w') as w_test:
        for line in r:
            inst = json.loads(line)
            doc_id = inst['doc_id']
            if doc_id in train_docs:
                w_train.write(line)
                real_train.add(doc_id)
            elif doc_id in dev_docs:
                w_dev.write(line)
                real_dev.add(doc_id)
            elif doc_id in test_docs:
                w_test.write(line)
                real_test.add(doc_id)
            else:
                print('this doc is not been covered!')
    print(train_docs.difference(real_train))
    print(dev_docs.difference(real_dev))    
    print(test_docs.difference(real_test))    


def parse_arguments() -> Namespace:
    """Parses command line arguments.

    Returns:
        args (Namespace): a Namespace of parsed arguments.
    """
    parser = ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        required=True,
                        help='Path to the input folder')
    parser.add_argument('-o',
                        '--output',
                        required=True,
                        help='Path to the output folder')
    parser.add_argument('-b',
                        '--bert',
                        help='BERT model name',
                        default='bert-large-cased')
    parser.add_argument('-c',
                        '--bert_cache_dir',
                        help='Path to the BERT cahce directory')

    parser.add_argument('-s', '--split', default=None,
                        help='Path to the split folder')

    parser.add_argument('-w', '--window', default=1, help='Integer for window size', type=int)
    args = parser.parse_args()
   
    return args


def main():
    args = parse_arguments()

    model_name = args.bert
    cache_dir = args.bert_cache_dir

    # Create a tokenizer based on the model name
    if model_name.startswith('bert-'):
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                  cache_dir=cache_dir)
    elif model_name.startswith('roberta-'):
        tokenizer = RobertaTokenizer.from_pretrained(model_name,
                                                     cache_dir=cache_dir)
    elif model_name.startswith('xlm-roberta-'):
        tokenizer = XLMRobertaTokenizer.from_pretrained(model_name,
                                                        cache_dir=cache_dir)
    elif model_name.startswith('lanwuwei'):
        tokenizer = BertTokenizer.from_pretrained(model_name,
                                                cache_dir=cache_dir, 
                                                do_lower_case=True)        
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, do_lower_case=False, use_fast=False)

    if args.window > 10000:
        f_size = 'doc'
    else:
        f_size = 'w{}'.format(args.window)

    sub_set = [
        ("LDC2015E29_DEFT_Rich_ERE_English_Training_Annotation_V2/data", "normal"),
        ("LDC2015E68_DEFT_Rich_ERE_English_Training_Annotation_R2_V2/data", "r2v2"),
        ("LDC2015E78_DEFT_Rich_ERE_Chinese_and_English_Parallel_Annotation_V2/data", "parallel")
    ]

    for d_path, dataset in sub_set:
        # Convert to JSON format
        json_path = os.path.join(args.output, '{}.{}.{}.json'.format("english",
                                                                dataset,
                                                                f_size))
        process_batch(os.path.join(args.input, d_path),
                    json_path,
                    window_size=args.window,
                    dataset=dataset,
                    language="english"
                    )

        # Convert to OneIE foramt
        oneie_path = os.path.join(args.output, '{}.{}.{}.oneie.json'.format("english",
                                                                        dataset,
                                                                        f_size))
        ere_to_oneie(json_path, oneie_path, tokenizer=tokenizer)
    
    if args.split:
        all_data = []
        for ds in ["normal", "r2v2", "parallel"]:
            with open(os.path.join(args.output,'{}.{}.{}.oneie.json'.format("english",ds,f_size)), 'r', encoding='utf-8') as r:
                for line in r:
                    all_data.append(json.loads(line))
        train_docs, dev_docs, test_docs = set(), set(), set()
        # load doc ids
        with open(os.path.join(args.split, 'train.doc.txt')) as r:
            train_docs.update(r.read().strip('\n').split('\n'))
        with open(os.path.join(args.split, 'dev.doc.txt')) as r:
            dev_docs.update(r.read().strip('\n').split('\n'))
        with open(os.path.join(args.split, 'test.doc.txt')) as r:
            test_docs.update(r.read().strip('\n').split('\n'))
        
        with open(os.path.join(args.output, 'train.{}.oneie.json'.format(f_size)), 'w') as w_train, \
            open(os.path.join(args.output, 'dev.{}.oneie.json'.format(f_size)), 'w') as w_dev, \
            open(os.path.join(args.output, 'test.{}.oneie.json'.format(f_size)), 'w') as w_test:
            for inst in all_data:
                doc_id = inst['doc_id']
                if doc_id in train_docs:
                    w_train.write(json.dumps(inst) + '\n')
                elif doc_id in dev_docs:
                    w_dev.write(json.dumps(inst) + '\n')
                elif doc_id in test_docs:
                    w_test.write(json.dumps(inst) + '\n')
                else:
                    print('missing!! {}'.format(doc_id))

if __name__ == '__main__':
    main()