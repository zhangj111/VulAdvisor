from collections import Counter

import spacy

# Load the English NLP model
nlp = spacy.load("en_core_web_sm")


def find_verb_object_pairs(sentences):
    verb_objects = []
    doc = nlp(sentences)

    for token in doc:
        # Check if the token is a verb
        if token.pos_ == 'VERB':
            # Find all objects associated with the verb, including passive cases
            for child in token.children:
                # Look for direct objects, indirect objects, objects of prepositions, and passive subjects
                if child.dep_ in ['dobj', 'iobj', 'pobj', 'nsubjpass']:
                    # Check for compound objects and build a phrase if present
                    compound = ''
                    for grandchild in child.children:
                        if grandchild.dep_ == 'compound':
                            compound += grandchild.text + ' '
                    full_object = compound + child.text
                    verb_objects.append((token.lemma_, full_object))

    return verb_objects

def derive_verbs_objects(verb_object_pairs):
    verbs, objects = zip(*verb_object_pairs) if verb_object_pairs else ([], [])
    return list(verbs), list(objects)

def generalized_jaccard_similarity(list1, list2):
    c1 = Counter(list1)
    c2 = Counter(list2)
    intersection = c1 & c2
    union = c1 | c2
    intersection_sum = sum(intersection.values())
    union_sum = sum(union.values())
    return intersection_sum / union_sum if union_sum else 1.0

def behavioral_acc(ground_truth, prediction):
    gt_vo_pairs = find_verb_object_pairs(ground_truth)
    pred_vo_pairs = find_verb_object_pairs(prediction)
    # print(gt_vo_pairs)
    # print(pred_vo_pairs)

    vo_pair_jaccard = generalized_jaccard_similarity(gt_vo_pairs, pred_vo_pairs)
    return vo_pair_jaccard
