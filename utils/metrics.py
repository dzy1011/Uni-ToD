
import json
from nltk import word_tokenize, ngrams
from nltk.translate.bleu_score import sentence_bleu

def get_fourgrams(sequence, **kwargs):
    """
    Return the 4-grams generated from a sequence of items, as an iterator.

    :param sequence: the source data to be converted into 4-grams
    :type sequence: sequence or iter
    :rtype: iter(tuple)
    """

    for item in ngrams(sequence, 4, **kwargs):
        yield item


class Metric:
    def __init__(self):
        self.reset()

    def reset(self):
        pass

    def update(self, output):
        raise NotImplementedError()

    def compute(self):
        raise NotImplementedError()



class BLEU(Metric):
    def __init__(self, dataset):
        self.dataset = dataset
        self._bleu = None
        self._count = None
        self.domain = {
            "camrest":{

            },
            "incar": {
                "schedule": {
                    "scores": list(),
                    "count": 0
                },
                "navigate": {
                    "scores": list(),
                    "count": 0
                },
                "weather": {
                    "scores": list(),
                    "count": 0
                }
            },
            "woz2.1": {
                "attraction": {
                    "scores": list(),
                    "count": 0
                },
                "restaurant": {
                    "scores": list(),
                    "count": 0
                },
                "hotel": {
                    "scores": list(),
                    "count": 0
                }
            }
        }
        super(BLEU, self).__init__()

    def reset(self):
        self._bleu = 0
        self._count = 0
        super(BLEU, self).reset()

    def update(self, output):
        # hypothesis and reference are assumed to be actual sequences of tokens
        hypothesis, reference, task = output

        hyp_tokens = hypothesis.split()
        ref_tokens = reference.split()

        bleu = sentence_bleu([ref_tokens], hyp_tokens)
        if self.dataset == "incar" or self.dataset == "woz2.1":
            self.domain[self.dataset][task]["scores"].append(bleu)
            self.domain[self.dataset][task]["count"] += 1
        self._bleu += bleu
        self._count += 1

    def compute(self):
        if self._count == 0:
            raise ValueError("BLEU-1 must have at least one example before it can be computed!")

        if self.dataset=="incar" or self.dataset=="woz2.1":
            for k,v in self.domain[self.dataset].items():
                print(self.dataset, k, sum(v["scores"])/v["count"])
        return self._bleu / self._count

    def name(self):
        return "BLEU"


class EntityF1:
    def __init__(self, dataset):
        self.dataset = dataset
        self.score = 0.0
        self.count = 0
        self.domain = {
            "camrest":{

            },
            "incar": {
                "schedule": {
                    "scores": list(),
                    "count": 0
                },
                "navigate": {
                    "scores": list(),
                    "count": 0
                },
                "weather": {
                    "scores": list(),
                    "count": 0
                }
            },
            "woz2.1": {
                "attraction": {
                    "scores": list(),
                    "count": 0
                },
                "restaurant": {
                    "scores": list(),
                    "count": 0
                },
                "hotel": {
                    "scores": list(),
                    "count": 0
                }
            }
        }
        self.entities = self.get_global_entities(dataset=dataset)

    def get_global_entities(self, dataset="incar"):
        if dataset=="incar":
            with open('data/incar/kvret_entities.json') as f:
                global_entity = json.load(f)
                global_entity_list = []
                for key in global_entity.keys():
                    if key != 'poi':
                        global_entity_list += [item.lower().replace(' ', '_') for item in global_entity[key]]
                    else:
                        for item in global_entity['poi']:
                            global_entity_list += [item[k].lower().replace(' ', '_') for k in item.keys()]
                global_entity_list = list(set(global_entity_list))
            return global_entity_list
        elif dataset=="camrest" or dataset=="woz2.1":
            return json.load(open(f"data/{dataset}/entities.json"))

    def compute_prf(self, gold, pred, kb_plain):
        local_kb_word = [k[2] for k in kb_plain]
        local_kb_word = list(set(local_kb_word))
        TP, FP, FN = 0, 0, 0
        if len(gold) != 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in set(pred):
                if p in self.entities or p in local_kb_word:
                    if p not in gold:
                        FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
        else:
            precision, recall, F1, count = 0, 0, 0, 0
        return F1, count

    def update(self, output):
        pred, ref, kb, task = output
        kb_temp = kb
        f1, c = self.compute_prf(gold=ref, pred=pred, kb_plain=kb_temp)

        if self.dataset == "incar" or self.dataset == "woz2.1":
            self.domain[self.dataset][task]["scores"].append(f1)
            self.domain[self.dataset][task]["count"] += 1

        self.score+= f1
        self.count+= c

    def compute(self):
        if self.dataset=="incar" or self.dataset=="woz2.1":
            for k,v in self.domain[self.dataset].items():
                print(self.dataset, k, sum(v["scores"])/v["count"])
        return self.score/(self.count+1e-30)

    def name(self):
        return "Entity-F1"
