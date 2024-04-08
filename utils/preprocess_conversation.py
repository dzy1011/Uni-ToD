import json
import pickle
import ast
from os.path import join
from tqdm import tqdm


def knowledge_to_sequence(kg):
    kg_dict0 = dict()
    kg_dict1 = dict()
    for triple in kg:
        if triple[0] not in kg_dict0:
            kg_dict0[triple[0]] = [triple[0],triple[-1]]
        else:
            kg_dict0[triple[0]].append(triple[-1])

    for triple in kg:
        if triple[0] not in kg_dict1:
            kg_dict1[triple[0]] = [triple]
        else:
            kg_dict1[triple[0]].append(triple)
    return  kg_dict0.copy(), kg_dict1.copy()

def process_incar():
    splits = ["val","test","train"]
    dataroot = "../data/incar"
    for sp in splits:
        with open(join(dataroot,"kvr",sp+".txt")) as f:
            conv_id = 0
            data = dict()
            task = None
            for line in tqdm(f, desc=f"processing files (incar): {sp}:"):
                line = line.strip()
                if line:
                    if '#' in line:
                        line = line.replace("#", "")
                        task = line
                        conv_id += 1
                        data[conv_id] = {
                            "task": task,
                            "utterances": [],
                            "kg": []
                        }
                        continue

                    nid, line = line.split(' ', 1)

                    if '\t' in line:        # conversation
                        u, r, gold_ent = line.split('\t')
                        gold_ent = ast.literal_eval(gold_ent)
                        data[conv_id]["utterances"].append({
                            "user": u,
                            "response": r,
                            "reference_entities": gold_ent
                        })
                    else:                   # kg triples
                        triple = line.split()
                        if task=="weather":
                            if len(triple)==4:
                                data[conv_id]["kg"].append([triple[0],triple[1],triple[2]+" "+triple[3]])
                            elif len(triple)==2:
                                data[conv_id]["kg"].append([triple[0],triple[1],triple[0]])
                            else:
                                data[conv_id]["kg"].append(triple)
                        else:
                            if len(triple)==3:
                                data[conv_id]["kg"].append(triple)

            for k, v in data.items():
                all_enti = []
                if v["kg"]==[]:
                    for turn in v['utterances']:
                        turn['kg_tripe'] = []
                else:
                    if v["task"] == "weather" :
                    # if v["task"] == "schedule" or v["task"] == "weather" :
                            for turn in v['utterances']:
                                turn['kg_tripe'] = []
                    else:
                        kg_tripe = []
                        kgdict0, kgdict1 = knowledge_to_sequence(v['kg'])
                        for turn in v['utterances']:
                            all_enti += turn["reference_entities"]
                        for kb_it_k, kb_it_v in kgdict0.items():
                            if all(x in kb_it_v for x in list(set(all_enti))):
                                # print('kb_it_v',kb_it_v)
                                kg_tripe = kgdict1[kb_it_k]
                                # kg_tripe.append(kgdict1[kb_it_k])
                        # print(len(kg_tripe))
                        for turn in v['utterances']:
                            turn["kg_tripe"] = kg_tripe

            json.dump(data, open(join(dataroot,sp+".json"), "w"), indent=3)

def process_kg(dataset, kb):
    triples = list()
    for data in kb:
        ent = data["name"]
        for k,v in data.items():
            if k!="name":
                triples.append([ent,k,v])
    return triples

def process_camrest(dataset):
    # splits = ["test"]
    splits = ["val","test","train"]
    dataroot = "../data/"+dataset

    for sp in splits:
        data = json.load(open(join(dataroot,"raw",sp+".json")))
        save_data = dict()
        kg_tripe = []
        turns = [100]
        all_enti = []
        turn_ids = []
        for conv_id, item in tqdm(enumerate(data), desc=f"{sp}:"):

            if len(item['context']) <= turns[-1]:
                turns = []
                for id in turn_ids:
                    if kg_tripe!=[]:
                        # save_data[id]['kg_tripe'] = kg_tripe[-1]
                        save_data[id]['kg_tripe'] = process_kg(dataset=dataset, kb=[kg_tripe[-1]])
                    else:
                        save_data[id]['kg_tripe'] = []
                kg_tripe = []
                all_enti = []
                turn_ids = []
                ###如果turns为空，则新一轮的对话开始了
            if item['kb'] !=[] and item['gold_entities'] !=[]:

                all_enti += item["gold_entities"]
                for kb_it in item['kb']:
                    kb_it_k = list(kb_it.keys())
                    kb_it_v = list(kb_it.values())
                    if all(x in kb_it_v for x in list(set(all_enti))):
                        kg_tripe.append(kb_it)
            turn_ids.append(conv_id)
            save_data[conv_id] = {
                    "task": item["cusine"],
                    "history": item["context"][:-1],
                    "user": item["context"][-1],
                    "response": item["output"],
                    "reference_entities": item["gold_entities"],
                    "kg": process_kg(dataset=dataset, kb=item["kb"])
            }
            turns.append(len(item['context']))
        json.dump(save_data, open(join(dataroot,sp+".json"), "w"), indent=3)


def process_woz21(dataset):
    splits = ["val","test","train"]
    dataroot = "../data/"+dataset

    for sp in splits:
        data = json.load(open(join(dataroot,"raw",sp+".json")))
        save_data = dict()
        kg_tripe = []
        turns = [1]
        all_enti = []
        turn_ids = []
        for conv_id, item in tqdm(enumerate(data), desc=f"{sp}:"):
            if item["did"] not in turns:
                for id in turn_ids:
                    if kg_tripe!=[]:
                        save_data[id]['kg_tripe'] = process_kg(dataset=dataset, kb=[kg_tripe[-1]])
                    else:
                        save_data[id]['kg_tripe'] = []
                kg_tripe = []
                all_enti = []
                turn_ids = []
                ###如果turns为空，则新一轮的对话开始了
            if item['kb'] !=[] and item['gold_entities'] !=[]:
                all_enti += item["gold_entities"]
                for kb_it in item['kb']:
                    kb_it_k = list(kb_it.keys())
                    kb_it_v = list(kb_it.values())
                    if all(x in kb_it_v for x in list(set(all_enti))):
                        kg_tripe.append(kb_it)
            turn_ids.append(conv_id)
            save_data[conv_id] = {
                    "task": item["type"],
                    "history": item["context"][:-1],
                    "user": item["context"][-1],
                    "response": item["output"],
                    "reference_entities": item["gold_entities"],
                    "kg": process_kg(dataset=dataset, kb=item["kb"])
            }
            turns.append(item['did'])
        json.dump(save_data, open(join(dataroot,sp+".json"), "w"), indent=3)



def truncate_long_context(long_text):
    long_text = " ".join(long_text.split()[-400:])
    return long_text

def get_pkl(dataset):
    dataroot = "../data/"+dataset
    splits = ["val","test","train"]
    
    for datasplit in splits:
        data = json.load(open(join(dataroot, datasplit+".json")))
        formatted_dialogues = list()
        previous_id = ""
        for dial_id, dg in tqdm(data.items(), desc=f"get pkl: {dataset}:{datasplit}::: "):  # only show progress bar in one process
            current_id = dial_id
            if dataset=="incar":
                for t,atrun in enumerate(dg["utterances"]):
                    dialog = {}
                    dialog["id"] = dial_id
                    dialog["kg"] = dg["kg"]
                    dialog["task"] = dg["task"]
                    dialog["response"] = atrun["response"]

                    if current_id!=previous_id:
                        dialog["history"] = [atrun["user"]]
                    else:
                        dialog["history"] = formatted_dialogues[-1]["history"] + [dg["utterances"][t-1]["response"],atrun["user"]]

                    dialog["ref_ents"] = atrun["reference_entities"]
                    dialog["kg_tripe"] = atrun["kg_tripe"]


                    formatted_dialogues.append(dialog)
                    previous_id=current_id

            elif dataset=="camrest" or dataset=="woz2.1":
                dialog = {}
                # print(dg)
                dialog["id"] = dial_id
                dialog["kg"] = dg["kg"]
                dialog["task"] = dg["task"]
                dialog["response"] = dg["response"]
                dialog["history"] = dg["history"] + [dg["user"]]
                dialog["ref_ents"] = dg["reference_entities"]
                if 'kg_tripe' in dg.keys():
                    dialog['kg_tripe'] = dg['kg_tripe']
                else:
                    dialog['kg_tripe'] = []

                formatted_dialogues.append(dialog)

        pickle.dump(formatted_dialogues, open(join(dataroot, datasplit+".pkl"),"wb"))


def process_entities(dataset):
    if dataset=="camrest" or dataset=="woz2.1":
        ent_data = json.load(open(f"../data/{dataset}/raw/entities.json"))
        json.dump(ent_data["all_entities_list"], open(f"../data/{dataset}/entities.json","w"), indent=3)


def process_data(dataset="incar"):
    if dataset=="incar":
        process_incar()
        get_pkl(dataset)
    elif dataset=="camrest":
        process_camrest(dataset=dataset)
        process_entities(dataset=dataset)
        get_pkl(dataset=dataset)
    elif dataset=="woz2.1":
        process_woz21(dataset=dataset)
        process_entities(dataset=dataset)
        get_pkl(dataset=dataset)


if __name__=="__main__":
    process_data(dataset="incar")
    process_data(dataset="camrest")
    # process_data(dataset="woz2.1")
