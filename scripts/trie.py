import os, json
from itertools import chain

class Trie(object):
    """自定义Trie树对象，用来保存知识库
    """
    def __init__(self, value_key=-1):
        self.data = {}
        self.value_key = str(value_key)

    def __setitem__(self, key, value):
        """传入一对(key, value)到前缀树中
        """
        data = self.data
        for k in key:
            k = str(k)
            if k not in data:
                data[k] = {}
            data = data[k]
        if self.value_key in data:
            if data[self.value_key] != value:
                data[self.value_key] += ('\t' + value)
        else:
            data[self.value_key] = value

    def __getitem__(self, key):
        """获取key对应的value
        """
        data = self.data
        for k in key:
            k = str(k)
            data = data[k]
        return data[self.value_key]

    def next_ones(self, prefix):
        """获取prefix后一位的容许集
        """
        data = self.data
        for k in prefix:
            k = str(k)
            data = data[k]
        return [k for k in data if k != self.value_key]

    def keys(self, prefix=None, data=None):
        """获取以prefix开头的所有key
        """
        data = data or self.data
        prefix = prefix or []
        for k in prefix:
            k = str(k)
            if k not in data:
                return []
            data = data[k]
        results = []
        for k in data:
            if k == self.value_key:
                results.append([])
            else:
                results.extend([[k] + j for j in self.keys(None, data[k])])
        return [prefix + i for i in results]

    def save(self, filename):
        with open(filename, 'w') as f:
            json.dump(self.data, f, ensure_ascii=False)

    def load(self, filename):
        with open(filename) as f:
            self.data = json.load(f)
            
# kgdict ={'jinling_noodle_bar': [['address', '11_peas_hill_city_centre'], ['area', 'centre'], ['food', 'chinese'], ['phone', '01223_566188'], ['pricerange', 'moderate'], ['postcode', 'cb23pp']], 'lan_hong_house': [['address', '12_norfolk_street_city_centre'], ['area', 'centre'], ['food', 'chinese'], ['phone', '01223_350420'], ['pricerange', 'moderate'], ['postcode', 'cb12lf']], 'golden_wok': [['address', '191_histon_road_chesterton'], ['area', 'north'], ['food', 'chinese'], ['phone', '01223_350688'], ['pricerange', 'moderate'], ['postcode', 'cb43hl']], 'shanghai_family_restaurant': [['address', '39_burleigh_street_city_centre'], ['area', 'centre'], ['food', 'chinese'], ['phone', '01223_301761'], ['pricerange', 'moderate'], ['postcode', 'cb11dg']]}
kgsort = [['food','area','pricerange','name','address','phone','postcode'],['area','pricerange','food','name','address','phone','postcode'],
          ['area','food','pricerange','name','address','phone','postcode'],['food','pricerange','area','name','address','phone','postcode'],
          ['pricerange','food','area','name','address','phone','postcode'],['pricerange','area','food','name','address','phone','postcode']]
incarkgsort = [['poi_type','distance','traffic_info','name','address','time','date','agenda','room']]

wozkgsort = [['area','pricerange','type','food','stars','choice','name','address','phone','postcode','ref']]

def get_trie(kgdict, kgsort, KG, tokenizer):
    ids = []
    tpkg = {}
    for k, v in kgdict.items():
        v.append(['name', k])
        for idx in range(len(v)):
            tpkg[v[idx][0]] = v[idx][1]
        ids += [[tp, tpkg[tp]] for tp in kgsort[0] if tp in list(tpkg.keys())]
        # ids = list(chain(*ids))+["[EKG]"]
        ids = ["[USR]"]+list(chain(*ids))+["[EKG]"]
        # print('ids',ids)
        # exit()
        
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(' '.join(ids)))
        KG[ids] = 50258
        ids = []
        tpkg = {}
    return KG

def get_kg_res(kgdict, kgsort):
    ids = []
    tpkg = {}
    for k, v in kgdict.items():
        v.append(['name', k])
        for idx in range(len(v)):
            tpkg[v[idx][0]] = v[idx][1]
        ids += [[tp, tpkg[tp]] for tp in kgsort[0] if tp in list(tpkg.keys())]
        ids = list(chain(*ids))+["[EKG]"]
        # print(ids)
    return ids