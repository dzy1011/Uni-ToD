import torch
from torch.utils.data import DataLoader, Dataset
from utils.dataset_utils import pad_ids, truncate_sequences
from itertools import chain
from tqdm import tqdm
import os
from os.path import join
import json
import numpy as np
import pickle
from .trie import Trie, get_trie, kgsort, get_kg_res


SPECIAL_TOKENS = {
    "bos_token": "[BOS]",
    "eos_token": "[EOS]",
    "pad_token": "[PAD]",
    "additional_special_tokens": ["[SYS]", "[USR]", "[EKG]","[SUB]", "[PRED]","[OBJ]","[TRIPLE]", "[SEP]", "[SKG]"],
}

SPECIAL_TOKENS_VALUES = ["[BOS]", "[EOS]", "[PAD]", "[USR]", "[SUB]", "[SYS]", "[USR]", "[SUB]", "[PRED]","[OBJ]","[TRIPLE]", "[SEP]"]

class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        self.args = args
        self.dataroot = args.dataroot
        self.tokenizer = tokenizer
        self.split_type = split_type

        self.SPECIAL_TOKENS = SPECIAL_TOKENS
        self.SPECIAL_TOKENS_VALUES = SPECIAL_TOKENS_VALUES
        self.bos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["bos_token"])
        self.eos = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["eos_token"])
        self.pad = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["pad_token"])
        self.sys_token, self.usr_token, self.kg, self.sub_token, self.pred_token, self.obj_token, self.triple_token, self.sep, self.skg = self.tokenizer.convert_tokens_to_ids(self.SPECIAL_TOKENS["additional_special_tokens"])
        self.dialogs = self._prepare_conversations(dataset=name, split_type=split_type)

        self._create_examples()

    def build_input_from_segments(self, knowledge, history, response, trie, example, with_eos=True):
        """ Build a sequence of input from 3 segments: knowledge, history and last reply """
        instance = {}
        sequence = [[self.bos] + knowledge]  + history+ [response + ([self.eos] if with_eos else [])]
        sequence_with_speaker = [ [self.usr_token if ((len(sequence)-i) % 2) == 0 else self.sys_token] + s
            for i, s in enumerate(sequence[1:])]  # From the history
        hist_token_type = [[self.usr_token if ((len(sequence)-i) % 2) == 0 else self.sys_token]*(len(s)+1)
                for i, s in enumerate(sequence[1:])
        ]

        sequence = [sequence[0]] + sequence_with_speaker
        # print('sequence',sequence)
        # exit()
        instance["input_ids"] = list(chain(*sequence))
        instance["token_type_ids"] = [self.kg]*len(sequence[0])+list(chain(*hist_token_type))
        instance["mc_token_ids"] = len(instance["input_ids"]) - 1
        instance["lm_labels"] = ([-100] * sum(len(s) for s in sequence[:-1])) + [-100] + sequence[-1][1:]
        instance["pos_ids"] = [j for j in range(len(instance["input_ids"]))]
        instance['trie'] =  trie


        return instance, sequence


    def _prepare_conversations(self, dataset="incar", split_type="train"):
        print("Loading dialogue data...")
        formatted_dialogs = pickle.load(open(join("data",dataset,split_type+".pkl"),"rb"))
        return formatted_dialogs

    def _knowledge_to_sequence(self, kg):
        st=""
        kg_dict = dict()
        for triple in kg:
            if triple[0] not in kg_dict:
                kg_dict[triple[0]] = [triple[1:]]
            else:
                kg_dict[triple[0]].append(triple[1:])
        return kg_dict.copy()

    def _create_examples(self):
        print("Creating examples")
        self.examples = []
        kk = 0
        n_history = 20
        for dialog in tqdm(self.dialogs):
            dialog_id = dialog["id"]
            ref_ents = dialog["ref_ents"]

            kgdict_m = self._knowledge_to_sequence(dialog["kg"])
            kgdict_r = self._knowledge_to_sequence(dialog["kg_tripe"])

            if kgdict_m !={} and kgdict_r !={}:
                all_kg = Trie()
                kgtrie = get_trie(kgdict_m, kgsort, all_kg, self.tokenizer)
                kgres = get_kg_res(kgdict_r, kgsort)
            else:
                kgtrie=[]
                kgres = []

            used_knowledge = self.tokenizer.convert_tokens_to_ids([])

            history = [self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(turn)) for turn in dialog["history"]]
            if kgtrie !=[]:
                gt_resp = ' '+' '.join(kgres)+' '+dialog["response"]
            else:
                gt_resp =' '+"[EKG]"+' '+ dialog["response"]

            tokenized_gt_resp = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(gt_resp))

            # apply history threshold at an utterance-level (a large value can be used to nullify its effect)
            truncated_history = history[-self.args.history_max_utterances:]


            # perform token-level truncation of history from the left
            truncated_history = truncate_sequences(truncated_history, self.args.history_max_tokens)
            # print('truncated_history', truncated_history)
            truncated_history[-1] = truncated_history[-1] + [self.tokenizer.convert_tokens_to_ids("[SKG]")]



            self.examples.append({
                "history": truncated_history,
                "task": dialog["task"],
                "knowledge": used_knowledge,
                "knowledge_text": dialog["kg"],
                "response": tokenized_gt_resp,
                "response_text": gt_resp,
                "dialog_id": dialog_id,
                "reference_entities": ref_ents,
                'trie':kgtrie
            })

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.examples)


class Dataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        super(Dataset, self).__init__(args, tokenizer, name, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        instance, _ = self.build_input_from_segments(example["knowledge"],example["history"],example["response"], example["trie"], example)
        return instance

    def collate_fn(self, batch):
        input_ids = [ins["input_ids"] for ins in batch]
        token_type_ids = [ins["token_type_ids"] for ins in batch]
        lm_labels = [ins["lm_labels"] for ins in batch]
        pos_ids = [ins["pos_ids"] for ins in batch]
        trie = [ins["trie"] for ins in batch]

        input_ids = torch.tensor(pad_ids(input_ids, self.pad))
        token_type_ids = torch.tensor(pad_ids(token_type_ids, self.pad))
        pos_ids = torch.tensor(pad_ids(pos_ids, self.pad))
        lm_labels = torch.tensor(pad_ids(lm_labels, -100))
        return input_ids, token_type_ids, pos_ids, lm_labels, trie


class EvalDataset(BaseDataset):
    def __init__(self, args, tokenizer, name, split_type, labels=True, labels_file=None):
        super(EvalDataset, self).__init__(args, tokenizer, name, split_type, labels, labels_file)

    def __getitem__(self, index):
        example = self.examples[index]
        return example

    def collate_fn(self, batch):
        return batch
