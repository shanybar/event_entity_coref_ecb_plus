import os
import sys
import spacy

sys.path.append("/src/shared/")

for pack in os.listdir("src"):
    sys.path.append(os.path.join("src", pack))

from classes import *

matched_args = 0
matched_args_same_ix = 0

matched_events = 0
matched_events_same_ix = 0

nlp = spacy.load('en')


def order_docs_by_topics(docs):
    '''
    Gets list of document objects and returns a Corpus object.
    The Corpus object contains Document objects, ordered by their gold topics
    :param docs: list of document objects
    :return: Corpus object
    '''
    corpus = Corpus()
    for doc_id, doc in docs.items():
        topic_id, doc_no = doc_id.split('_')
        if 'ecbplus' in doc_no:
            topic_id = topic_id + '_' +'ecbplus'
        else:
            topic_id = topic_id + '_' +'ecb'
        if topic_id not in corpus.topics:
            topic = Topic(topic_id)
            corpus.add_topic(topic_id, topic)
        topic = corpus.topics[topic_id]
        topic.add_doc(doc_id, doc)
    return corpus


def load_ECB_plus(processed_ecb_file):
    '''
    This function gets the intermediate data  (train/test/dev split after it was extracted
    from the XML files and stored as a text file) and load it into objects
    that represent a document structure
    :param processed_ecb_file: the filename of the intermediate representation of the split,
    which is stored as a text file
    :return: dictionary of document objects, represents the documents in the split
    '''
    doc_changed = True
    sent_changed = True
    docs = {}
    last_doc_name = None
    last_sent_id = None

    for line in open(processed_ecb_file, 'r'):
        stripped_line = line.strip()
        try:
            if stripped_line:
                doc_id,sent_id,token_num,word, coref_chain = stripped_line.split('\t')
                doc_id = doc_id.replace('.xml','')
        except:
            row = stripped_line.split('\t')
            clean_row = []
            for item in row:
                if item:
                    clean_row.append(item)
            doc_id, sent_id, token_num, word, coref_chain = clean_row
            doc_id = doc_id.replace('.xml', '')

        if stripped_line:
            sent_id = int(sent_id)

            if last_doc_name is None:
                last_doc_name = doc_id
            elif last_doc_name != doc_id:
                doc_changed = True
                sent_changed = True
            if doc_changed:
                new_doc = Document(doc_id)
                docs[doc_id] = new_doc
                doc_changed = False
                last_doc_name = doc_id

            if last_sent_id is None:
                last_sent_id = sent_id
            elif last_sent_id != sent_id:
                sent_changed = True
            if sent_changed:
                new_sent = Sentence(sent_id)
                sent_changed = False
                new_doc.add_sentence(sent_id,new_sent)
                last_sent_id = sent_id

            new_tok = Token(token_num,word,'-')
            new_sent.add_token(new_tok)

    return docs


def find_args_by_dependency_parsing(dataset, is_gold):
    '''
    Runs dependency parser on the split's sentences and augments the predicate-argument structures
    :param dataset: an object represents the split (Corpus object)
    :param is_gold: whether to match arguments and predicates with gold or predicted mentions
    '''
    global matched_args, matched_args_same_ix, matched_events,matched_events_same_ix
    matched_args = 0
    matched_args_same_ix = 0
    matched_events = 0
    matched_events_same_ix = 0
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                sent_str = sent.get_raw_sentence()
                parsed_sent = nlp(sent_str)
                findSVOs(parsed_sent=parsed_sent, sent=sent, is_gold=is_gold)

    print('matched events : {} '.format(matched_events))
    print('matched args : {} '.format(matched_args))


def find_left_and_right_mentions(dataset, is_gold):
    '''
    Finds for each event in the split's its closest left and right entity mentions
    and augments the predicate-argument structures
    :param dataset: an object represents the split (Corpus object)
    :param is_gold: whether to use gold or predicted mentions
    '''
    for topic_id, topic in dataset.topics.items():
        for doc_id, doc in topic.docs.items():
            for sent_id, sent in doc.get_sentences().items():
                add_left_and_right_mentions(sent, is_gold)


def match_subj_with_event(verb_text, verb_index, subj_text, subj_index, sent, is_gold):
    '''
    Given a verb and a subject extracted by the dependency parser , this function tries to match
    the verb with an event mention and the subject with an entity mention
    :param verb_text: the verb's text
    :param verb_index: the verb index in the parsed sentence
    :param subj_text: the subject's text
    :param subj_index: the subject index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    '''
    event = match_event(verb_text, verb_index, sent, is_gold)
    if event is not None and event.arg0 is None:
        entity = match_entity(subj_text, subj_index, sent, is_gold)
        if entity is not None:
            if event.arg1 is not None and event.arg1 == (entity.mention_str, entity.mention_id):
                return
            if event.amloc is not None and event.amloc == (entity.mention_str, entity.mention_id):
                return
            if event.amtmp is not None and event.amtmp == (entity.mention_str, entity.mention_id):
                return
            event.arg0 = (entity.mention_str, entity.mention_id)
            entity.add_predicate((event.mention_str, event.mention_id), 'A0')


def match_obj_with_event(verb_text, verb_index, obj_text, obj_index, sent, is_gold):
    '''
    Given a verb and an object extracted by the dependency parser , this function tries to match
    the verb with an event mention and the object with an entity mention
    :param verb_text: the verb's text
    :param verb_index: the verb index in the parsed sentence
    :param obj_text: the object's text
    :param obj_index: the object index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    '''
    event = match_event(verb_text, verb_index, sent, is_gold)
    if event is not None and event.arg1 is None:
        entity = match_entity(obj_text, obj_index, sent, is_gold)
        if entity is not None:
            if event.arg0 is not None and event.arg0 == (entity.mention_str, entity.mention_id):
                return
            if event.amloc is not None and event.amloc == (entity.mention_str, entity.mention_id):
                return
            if event.amtmp is not None and event.amtmp == (entity.mention_str, entity.mention_id):
                return
            event.arg1 = (entity.mention_str, entity.mention_id)
            entity.add_predicate((event.mention_str, event.mention_id), 'A1')


def match_event(verb_text, verb_index, sent, is_gold):
    '''
    Given a verb extracted by the dependency parser , this function tries to match
    the verb with an event mention.
    :param verb_text: the verb's text
    :param verb_index: the verb index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    :return: the matched event (and None if the verb doesn't match to any event mention)
    '''
    global matched_events, matched_events_same_ix
    sent_events = sent.gold_event_mentions if is_gold else sent.pred_event_mentions
    for event in sent_events:
        event_toks = event.tokens
        for tok in event_toks:
            if tok.get_token() == verb_text:
                if is_gold:
                    matched_events += 1
                elif event.gold_mention_id is not None:
                    matched_events += 1
                if verb_index == int(tok.token_id):
                    matched_events_same_ix += 1
                return event
    return None


def match_entity(entity_text, entity_index, sent, is_gold):
    '''
    Given an argument extracted by the dependency parser , this function tries to match
    the argument with an entity mention.
    :param entity_text: the argument's text
    :param entity_index: the argument index in the parsed sentence
    :param sent: an object represents the sentence (Sentence object)
    :param is_gold: whether to match with gold or predicted mentions
    :return: the matched entity (and None if the argument doesn't match to any event mention)
    '''
    global matched_args, matched_args_same_ix
    sent_entities = sent.gold_entity_mentions if is_gold else sent.pred_entity_mentions
    for entity in sent_entities:
        entity_toks = entity.tokens
        for tok in entity_toks:
            if tok.get_token() == entity_text:
                if is_gold:
                    matched_args += 1
                elif entity.gold_mention_id is not None:
                    matched_args += 1
                if entity_index == int(tok.token_id):
                    matched_args_same_ix += 1
                return entity
    return None

'''
Borrowed with modifications from https://github.com/NSchrading/intro-spacy-nlp/blob/master/subject_object_extraction.py
'''

SUBJECTS = ["nsubj"]
PASS_SUBJ = ["nsubjpass",  "csubjpass"]
OBJECTS = ["dobj", "iobj", "attr", "oprd"]


def getSubsFromConjunctions(subs):
    '''
    Finds subjects in conjunctions (and)
    :param subs: found subjects so far
    :return: additional subjects, if exist
    '''
    moreSubs = []
    for sub in subs:
        rights = list(sub.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreSubs.extend([tok for tok in rights if tok.dep_ in SUBJECTS or tok.pos_ == "NOUN"])
            if len(moreSubs) > 0:
                moreSubs.extend(getSubsFromConjunctions(moreSubs))
    return moreSubs


def getObjsFromConjunctions(objs):
    '''
    Finds objects in conjunctions (and)
    :param objs: found objects so far
    :return: additional objects, if exist
    '''
    moreObjs = []
    for obj in objs:
        # rights is a generator
        rights = list(obj.rights)
        rightDeps = {tok.lower_ for tok in rights}
        if "and" in rightDeps:
            moreObjs.extend([tok for tok in rights if tok.dep_ in OBJECTS or tok.pos_ == "NOUN"])
            if len(moreObjs) > 0:
                moreObjs.extend(getObjsFromConjunctions(moreObjs))
    return moreObjs


def getObjsFromPrepositions(deps):
    '''
    Finds objects in prepositions
    :param deps: dependencies extracted by spaCy parser
    :return: objects extracted from prepositions
    '''
    objs = []
    for dep in deps:
        if dep.pos_ == "ADP" and dep.dep_ == "prep":
            objs.extend([tok for tok in dep.rights if tok.dep_  in OBJECTS])
    return objs


def getObjFromXComp(deps):
    '''
     Finds objects in XComp phrases (X think that [...])
    :param deps: dependencies extracted by spaCy parser
    :return: objects extracted from XComp phrases
    '''
    for dep in deps:
        if dep.pos_ == "VERB" and dep.dep_ == "xcomp":
            v = dep
            rights = list(v.rights)
            objs = [tok for tok in rights if tok.dep_ in OBJECTS]
            objs.extend(getObjsFromPrepositions(rights))
            if len(objs) > 0:
                return v, objs
    return None, None


def getAllSubs(v):
    '''
    Finds all possible subjects of an extracted verb
    :param v: an extracted verb
    :return: all possible subjects of the verb
    '''
    subs = [tok for tok in v.lefts if tok.dep_ in SUBJECTS and tok.pos_ != "DET"]
    pass_subs = [tok for tok in v.lefts if tok.dep_ in PASS_SUBJ and tok.pos_ != "DET"]
    if len(subs) > 0:
        subs.extend(getSubsFromConjunctions(subs))
    return subs, pass_subs


def getAllObjs(v):
    '''
     Finds all the objects of an extracted verb
    :param v: an extracted verb
    :return: all possible objects of the verb
    '''
    rights = list(v.rights)
    objs = [tok for tok in rights if tok.dep_ in OBJECTS]
    objs.extend(getObjsFromPrepositions(rights))

    potentialNewVerb, potentialNewObjs = getObjFromXComp(rights)
    if potentialNewVerb is not None and potentialNewObjs is not None and len(potentialNewObjs) > 0:
        objs.extend(potentialNewObjs)
        v = potentialNewVerb
    if len(objs) > 0:
        objs.extend(getObjsFromConjunctions(objs))
    return v, objs


def findSVOs(parsed_sent, sent, is_gold):
    '''
    Given a parsed sentences, the function extracts its verbs, their subjects and objects and matches
    the verbs with event mentions, and matches the subjects and objects with entity mentions, and
    set them as Arg0 and Arg1 respectively.
    Finally, the function finds nominal event mentions with possesors, matches the possesor
    with entity mention and set it as Arg0.
    :param parsed_sent: a sentence, parsed by spaCy
    :param sent: the original Sentence object
    :param is_gold: whether to match with gold or predicted mentions
    '''
    global matched_events, matched_events_same_ix
    global matched_args, matched_args_same_ix
    verbs = [tok for tok in parsed_sent if tok.pos_ == "VERB" and tok.dep_ != "aux"]
    for v in verbs:
        subs, pass_subs = getAllSubs(v)
        v, objs = getAllObjs(v)
        if len(subs) > 0 or len(objs) > 0 or len(pass_subs) > 0:
            for sub in subs:
                match_subj_with_event(verb_text=v.orth_,
                                      verb_index=v.i, subj_text=sub.orth_,
                                      subj_index=sub.i, sent=sent, is_gold=is_gold)

            for obj in objs:
                match_obj_with_event(verb_text=v.orth_,
                                        verb_index=v.i, obj_text=obj.orth_,
                                        obj_index=obj.i, sent=sent, is_gold=is_gold)
            for obj in pass_subs:
                match_obj_with_event(verb_text=v.orth_,
                                        verb_index=v.i, obj_text=obj.orth_,
                                        obj_index=obj.i, sent=sent, is_gold=is_gold)

    find_nominalizations_args(parsed_sent, sent, is_gold) # Handling nominalizations


def find_nominalizations_args(parsed_sent, sent, is_gold):
    '''
    The function finds nominal event mentions with possesors, matches the possesor
    with entity mention and set it as Arg0.
    :param parsed_sent: a sentence, parsed by spaCy
    :param sent: the original Sentence object
    :param is_gold: whether to match with gold or predicted mentions
    '''
    possible_noms = [tok for tok in parsed_sent if tok.pos_ == "NOUN"]
    POSS = ['poss', 'possessive']
    for n in possible_noms:
        subs = [tok for tok in n.lefts if tok.dep_ in POSS and tok.pos_ != "DET"]
        if len(subs) > 0:
            for sub in subs:
                match_subj_with_event(verb_text=n.orth_,
                                      verb_index=n.i, subj_text=sub.orth_,
                                      subj_index=sub.i, sent=sent, is_gold=is_gold)


def add_left_and_right_mentions(sent, is_gold):
    '''
    The function finds the closest left and right entity mentions of each event mention
     and sets them as Arg0 and Arg1, respectively.
    :param sent: Sentence object
    :param is_gold: whether to match with gold or predicted mentions
    '''
    sent_events = sent.gold_event_mentions if is_gold else sent.pred_event_mentions
    for event in sent_events:
        if event.arg0 is None:
            left_ent = sent.find_nearest_entity_mention(event, is_left=True, is_gold=is_gold)
            if left_ent is not None:
                double_arg = False
                if event.arg1 is not None and event.arg1 == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True
                if event.amloc is not None and event.amloc == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True
                if event.amtmp is not None and event.amtmp == (left_ent.mention_str, left_ent.mention_id):
                    double_arg = True

                if not double_arg:
                    event.arg0 = (left_ent.mention_str, left_ent.mention_id)
                    left_ent.add_predicate((event.mention_str, event.mention_id), 'A0')

        if event.arg1 is None:
            right_ent = sent.find_nearest_entity_mention(event, is_left=False, is_gold=is_gold)
            if right_ent is not None:
                double_arg = False
                if event.arg0 is not None and event.arg0 == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if event.amloc is not None and event.amloc == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if event.amtmp is not None and event.amtmp == (right_ent.mention_str, right_ent.mention_id):
                    double_arg = True
                if not double_arg:
                    event.arg1 = (right_ent.mention_str, right_ent.mention_id)
                    right_ent.add_predicate((event.mention_str, event.mention_id), 'A1')