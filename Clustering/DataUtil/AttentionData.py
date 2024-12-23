import numpy as np
import h5py
import matplotlib.pyplot as plt
from statistics import mode
from tqdm import tqdm
import torch
import regex as re
from DataUtil.Scalers import log_normalize_scaler

#Keep track of attention data in a dict of size, Query, Layer, Head

class Attention_DB():
    def __init__(self, db_name, langs, corrects, querys, layers, heads):
        types = ['raw', 'enc_cls', 'enc_mean', 'class_cls', 'class_mean']
        self.langs = langs
        self.corrects = corrects
        self.querys = querys
        self.layers = layers
        self.heads = heads
        self.db_name = db_name
        self.DB_folder = "./Database/" + db_name + '_%d' 
        self.id = -1
        self.free_id = -99

        with h5py.File(self.DB_folder, mode='a', driver = 'family') as db:
            for lang in langs:
                for c in corrects:
                    for q in querys:
                        for l in layers:
                            for h in heads:
                                grp = db.require_group("/{}/{}/{}/{}/{}".format(lang,c,q,l,h))
                                for t in types:
                                    if "raw" in t:
                                        try:
                                            grp.require_dataset(t, (500, 256, 256), 'f', chunks = (1,256,256), fillvalue = -99)
                                            continue
                                        except ValueError:
                                            continue
                                    if "enc" in t:
                                        try:
                                            grp.require_dataset(t, (500, 512), 'f', chunks= (100, 512), fillvalue = -99)
                                            continue
                                        except ValueError as e:
                                            print(e)
                                            continue
                                    if "class" in t:
                                        try:
                                            grp.require_dataset(t, (500, 1), 'f', chunks = True, fillvalue = -99)
                                            continue
                                        except ValueError:
                                            continue
    
    def clear_data(self, name):
        with h5py.File(self.DB_folder, mode='a', driver= 'family') as db:
            for lang in self.langs:
                for c in self.corrects:
                    for q in self.querys:
                        for l in self.layers:
                            for h in self.heads:
                                dset = db['/{}/{}/{}/{}/{}/{}'.format(lang, c, q, l, h, name)]
                                ovr = self.free_id * np.ones(dset.shape)
                                db['/{}/{}/{}/{}/{}/{}'.format(lang, c, q, l, h, name)][:] = ovr
                                    

    def clear_dset(self, name):
        (l, h, w) = f[name].shape
        del f[name]
        f.create_dataset(name, (l,h,w), chunks = (l//10,h,w))

    def bin_search(self, dset):
        lo = 0
        hi = dset.shape[0]
        cmp = self.free_id * np.ones(dset[0].shape)
        #cmp = np.zeros(dset[0].shape)
        if not np.array_equal(dset[-1], cmp):
            return hi
        while lo <= hi:
            mid = lo + (hi - lo)//2
            if mid >= dset.shape[0]-1:
                return mid
            if lo == dset.shape[0]-1:
                return lo
            if not np.array_equal(dset[mid], cmp) and np.array_equal(dset[mid+1], cmp):
                return mid + 1
            elif np.array_equal(dset[mid], cmp):
                hi = mid - 1
            else:
                lo = mid + 1
        return 0

    def get_free_slot(self, dset):
        return self.bin_search(dset)

    def get_dataset(self, db, lang, correct, query, layer, head, types):
        return db['/{}/{}/{}/{}/{}/{}'.format(lang, correct, query, layer, head, types)]

    def get_sample(self, db, lang, correct, query, layer, head, types):
        dset = self.get_dataset(db, lang, correct, query, layer, head, types)
        i = self.get_free_slot(dset)
        return dset[0:i]

    def get_sample_by_cluster(self, db, lang, correct, query, layer, head, test_types, return_types, cluster):
        idset = self.get_sample(db, lang, correct, query, layer, head, test_types)
        dset = self.get_dataset(db, lang, correct, query, layer, head, return_types)
        add = dset.shape[0] - idset.shape[0]
        add = np.expand_dims(np.array([False] * add), axis = 1)
        select = np.concatenate(((idset == cluster), add))
        if return_types =="raw":
            return dset[select.squeeze(),:,:]
        else:
            return dset[select.squeeze(),:]
        

    #Order must be exactly the same as when the samples are read using get_grouped_samples
    def write_grouped_samples(self, langs, corrects, querys, layers, heads, types, data):
        with h5py.File(self.DB_folder, mode='a', driver = 'family') as db:
            if langs == "*":
                langs = list(db.keys())
            for l in langs:
                grp_lang = db[l]
                if corrects == "*":
                    corrects = list(grp_lang.keys())
                for c in corrects:
                    grp_correct = grp_lang[c]
                    if querys == "*":
                        querys = list(grp_correct.keys())
                    for q in querys:
                        grp_query = grp_correct[q]
                        if layers == "*":
                            layers = list(grp_query.keys())
                        for ls in layers:
                            grp_layer = grp_query[ls]
                            if heads == "*":
                                heads = list(grp_layer.keys())
                            for h in heads:
                                length = self.get_free_slot(self.get_dataset(db, l,c,q,ls,h,"raw"))
                                ovr_data = data[0:length]
                                data = data[length:]
                                self.overwrite(db,l,c,q,ls,h,types, ovr_data)


    def get_grouped_predictions(self, langs, corrects, querys, types, grp_by_lang, grp_by_correct, grp_by_query):
        pred_name_list = []
        pred_data_list = []
        with h5py.File(self.DB_folder, mode='a', driver = 'family') as db:
            if langs == "*":
                langs = list(db.keys())
            for l in langs:
                grp_lang = db[l]
                if corrects == "*":
                    corrects = list(grp_lang.keys())
                for c in corrects:
                    grp_correct = grp_lang[c]
                    if querys == "*":
                        querys = list(grp_correct.keys())
                    for q in querys:
                        grp_query = grp_correct[q]

                        pred_max_id = self.get_free_slot(self.get_dataset(db, l,c,q,'0','0', types))
                        
                        for pred_id in tqdm(range(pred_max_id)):
                            pred = []
                            layers = list(grp_query.keys())
                            for ls in layers:
                                grp_layer = grp_query[ls]
                                heads = list(grp_layer.keys())
                                for h in heads:
                                    pred.append(self.get_dataset(db, l,c,q,ls,h,types)[pred_id])
                            pred_data_list.append(np.stack(pred))


                            pred_id_string = ''
                            if not grp_by_lang:
                                pred_id_string += l
                            if not grp_by_correct:
                                pred_id_string += c
                            if not grp_by_query:
                                pred_id_string += q
                            pred_name_list.append(np.array([pred_id_string]))

        return np.stack(pred_data_list), np.concatenate(pred_name_list)




    def get_grouped_samples(self, langs, corrects, querys, layers, heads, types):
        return_list = []
        with h5py.File(self.DB_folder, mode='a', driver = 'family') as db:
            if langs == "*":
                langs = list(db.keys())
            for l in langs:
                grp_lang = db[l]
                if corrects == "*":
                    corrects = list(grp_lang.keys())
                for c in corrects:
                    grp_correct = grp_lang[c]
                    if querys == "*":
                        querys = list(grp_correct.keys())
                    for q in querys:
                        grp_query = grp_correct[q]
                        if layers == "*":
                            layers = list(grp_query.keys())
                        for ls in layers:
                            grp_layer = grp_query[ls]
                            if heads == "*":
                                heads = list(grp_layer.keys())
                            for h in heads:
                                return_list.append(self.get_sample(db, l, c, q, ls, h, types))
        return np.concatenate(return_list)

    def get_by_cluster(self, langs, corrects, querys, layers, heads, test_types, cluster, return_types = 'raw'):
        return_list = []
        with h5py.File(self.DB_folder, mode='a', driver = 'family') as db:
            if langs == "*":
                langs = list(db.keys())
            for l in langs:
                grp_lang = db[l]
                if corrects == "*":
                    corrects = list(grp_lang.keys())
                for c in corrects:
                    grp_correct = grp_lang[c]
                    if querys == "*":
                        querys = list(grp_correct.keys())
                    for q in querys:
                        grp_query = grp_correct[q]
                        if layers == "*":
                            layers = list(grp_query.keys())
                        for ls in layers:
                            grp_layer = grp_query[ls]
                            if heads == "*":
                                heads = list(grp_layer.keys())
                            for h in heads:
                                return_list.append(self.get_sample_by_cluster(db, l, c, q, ls, h, test_types, return_types, cluster))
        return np.concatenate(return_list)


    
    def get_grouped_clusters(self, langs, corrects, querys, layers, heads, types, grp_by_lang, grp_by_correct, grp_by_query, grp_by_layer, grp_by_head):
        clusters_list = []
        with h5py.File(self.DB_folder, mode='a', driver='family') as db:
            if langs == "*":
                langs = list(db.keys())
            for l in langs:
                grp_lang = db[l]
                if corrects == "*":
                    corrects = list(grp_lang.keys())
                for c in corrects:
                    grp_correct = grp_lang[c]
                    if querys == "*":
                        querys = list(grp_correct.keys())
                    for q in querys:
                        grp_query = grp_correct[q]
                        if layers == "*":
                            layers = list(grp_query.keys())
                        for ls in layers:
                            grp_layer = grp_query[ls]
                            if heads == "*":
                                heads = list(grp_layer.keys())
                            for h in heads:
                                cluster_id_string = ''
                                if not grp_by_lang:
                                    cluster_id_string += l 
                                if not grp_by_correct:
                                    cluster_id_string += c 
                                if not grp_by_query:
                                    cluster_id_string += q 
                                if not grp_by_layer:
                                    cluster_id_string += 'l'+ls 
                                if not grp_by_head:
                                    cluster_id_string += 'h'+h 
                                clusters_list.append(np.array([cluster_id_string] * self.get_free_slot(self.get_dataset(db,l, c, q, ls, h, types))))
        return np.concatenate(clusters_list)




    def insert_into(self, db, lang, correct, query, layer, head, types, data):
        dset = self.get_dataset(db, lang, correct, query, layer, head, types)
        i = self.get_free_slot(dset)
        for j, d in enumerate(data):
            try:
                dset[i + j] = d
            except IndexError as e:
                raise e
                print("Insert failed for {}/{}/{}/{}/{}/{}, database full".format(lang, correct, query, layer, head, types))
                break
    
    def overwrite(self, db, lang, correct, query, layer, head, types, data):
        dset = self.get_dataset(db, lang, correct, query, layer, head, types)
        length = data.shape[0]
        dset[0:length] = data

    def print(self):
        return self.name


class AttentionData():
    def __init__(self, model_config, queries, languages, db_name):
        self.layers = model_config.num_hidden_layers
        self.heads = model_config.num_attention_heads

        tree_string = '({} ({} ({} ({} ({})))))'.format(' '.join(languages), ' '.join(['correct', 'incorrect']), ' '.join(queries), ' '.join([str(x) for x in range(0, self.layers)]), ' '.join([str(x) for x in range(0, self.heads)]))

        self.data = Attention_DB(db_name, langs = ['java'], corrects = ['correct', 'incorrect'], querys = queries, layers = list(range(self.layers)), heads = list(range(self.heads)))

    def getByQuery(self, lang, correct, query, layer, head, types):
        self.data.get(lang, correct, query, layer, head, types)
    
    def chunker(self, seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    def encode(self, encoding_model, attention_loader = None):
        with h5py.File(self.data.DB_folder, mode='a', driver ='family') as db:
            for lang in list(db.keys()):
                grp_lang = db[lang]
                for correct in list(grp_lang.keys()):
                    grp_correct = grp_lang[correct]
                    for query in list(grp_correct.keys()):
                        grp_query = grp_correct[query]
                        for layer in tqdm(list(grp_query.keys())):
                            grp_layer = grp_query[layer]
                            for head in list(grp_layer.keys()):
                                grp_head = grp_layer[head]
                                dset = grp_head['raw']
                                i = self.data.get_free_slot(dset)

                                for head_batch in self.chunker(dset[0:i], 250):
                                    scaled = log_normalize_scaler(torch.tensor(head_batch).unsqueeze(dim =1), None)
                                    encoded = encoding_model.encoder.encode(scaled).detach().cpu()
                                    idx = [0, 1, 9, 10, 17, 18, 19, 25, 26, 27, 28, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63, 64]
                                    encoded = encoded[:,idx]

                                    self.data.insert_into(db, lang, correct, query, layer, head, 'enc_mean', encoded.mean(dim= 1).numpy())
                                    self.data.insert_into(db, lang, correct, query, layer, head, 'enc_cls', encoded[:,0,:].numpy())

    def generate_patterns(self, attention_loader):
        #TODO: fix for multilang
        lang = 'java'
        with h5py.File(self.data.DB_folder, mode='a', driver = 'family') as db:
            for i in tqdm(attention_loader):
                attention, query, correct = i
                attention = attention.detach().cpu().numpy()

                for l, layer in enumerate(attention):
                    for h, head in enumerate(layer):
                        self.data.insert_into(db, lang, correct, query, l, h, "raw", [head])

    def train_classifier(self, n_clusters, langs = "*", corrects = "*", querys = "*", layers = "*", heads = "*"):

        self.classifier_cls = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
        self.classifier_mean = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)

        train_data = self.data.get_grouped_samples(langs, corrects, querys, layers, heads, 'enc_mean')
        self.classifier_mean.fit(train_data)

        train_data = self.data.get_grouped_samples(langs, corrects, querys, layers, heads, 'enc_cls')
        self.classifier_cls.fit(train_data)

    def generate_sample(self, attention_loader, encoding_model, cluster_id, closure = 'mean'):
        candidate_heads = []
        for i in tqdm(attention_loader):
            attention, query, correct = i

            for l, layer in enumerate(attention):
                raw_heads = layer.squeeze().unsqueeze(dim = 1)
                heads = encoding_model.encoder.encode(raw_heads).detach().cpu()
                if closure == 'mean':
                    encoded = heads.mean(dim = 1).numpy()
                    classified = self.classifier_mean.predict(encoded)
                elif closure == 'cls':
                    encoded = heads[:,0,:].numpy()
                    classified = self.classifier_cls.predict(encoded)
                else:
                    raise Exception("Not a valid closure")

                candidate_heads.extend(raw_heads.detach().cpu().numpy()[classified == cluster_id])
        return candidate_heads        

    def visualize(self, query):
        z = []
        q = query
        for l in self.labeled[q]:
            for h in self.labeled[q][l]:
                z.append(mode(self.labeled[q][l][h]))
        heads, layers = np.meshgrid(range(0,16), range(0,24))
        layers.sort()

        ax = plt.subplot(111, aspect='equal')
        ax.scatter(heads, layers, c=z, marker='s', s=90, lw=0, cmap='Reds')
        plt.title('Attention heads for ' + query)
        plt.xlabel('Head')
        plt.ylabel('Layer')
        plt.show()

