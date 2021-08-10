import os
import json
import copy
import spacy
import torch
import numpy as np
import networkx as nx
from collections import defaultdict
from operator import add

from . import LsrConfig
from .modules.bert import Bert
from .utils import h_t_idx_generator, get_default_device, join_document


class LsrPreprocessor:
    """Class for preprocessing a DocRED-like data batch to a tensor batch for LsrModel to predict on.
    """

    def __init__(
            self,
            rel2id_path: str,
            word2id_path: str,
            ner2id_path: str,
            output_file_prefix: str = 'dev',
            output_dir: str = None,
            config: LsrConfig = None,
            max_node_num: int = 200,
            max_node_per_sent: int = 40,
            max_sent_num: int = 30,
            max_sent_len: int = 200,
            max_entity_num: int = 100,
            h_t_limit: int = 1800,
            is_train: bool = False,
            device=None
    ):
        # Load mappings
        self.rel2id = json.load(open(rel2id_path))
        self.word2id = json.load(open(word2id_path))
        self.ner2id = json.load(open(ner2id_path))

        self.output_file_prefix = output_file_prefix
        self.output_dir = output_dir

        self.config = config if config else LsrConfig()
        self.max_length = self.config.max_length
        self.num_relations = self.config.num_relations
        self.use_bert = self.config.use_bert

        self.max_node_num = max_node_num
        self.max_node_per_sent = max_node_per_sent
        self.max_sent_num = max_sent_num
        self.max_sent_len = max_sent_len
        self.max_entity_num = max_entity_num
        self.h_t_limit = h_t_limit
        self.is_train = is_train
        self.device = device if device else get_default_device()

        # Load spacy model
        self.nlp = spacy.load("en_core_web_sm")

        # Load Bert if needed
        if self.use_bert:
            self.bert = Bert('bert-base-uncased')

        # Build dis2idx
        self.dis2idx = self._get_dis2idx()

    def __call__(self, data_batch, save_output=False):
        np_vectors = self.get_numpy_vectors(data_batch, save_output)
        tensor_batch = self.get_tensor_batch(np_vectors)
        return tensor_batch

    def get_numpy_vectors(self, data_batch, save_output=False):
        mdp_vectors = self.get_mdp_vectors(data_batch)
        sentence_vectors = self.get_sentence_vectors(mdp_vectors['data'])
        np_vectors = {**mdp_vectors, **sentence_vectors}
        if self.use_bert:
            bert_vectors = self.get_bert_vectors(data_batch)
            np_vectors = {**np_vectors, **bert_vectors}
        if save_output:
            self.save_preprocessed_data(np_vectors)

        return np_vectors

    def get_sentence_vectors(self, data_batch):
        # Process for sentences
        batch_size = len(data_batch)

        sen_word = np.zeros((batch_size, self.max_length), dtype=np.int64)
        sen_pos = np.zeros((batch_size, self.max_length), dtype=np.int16)
        sen_ner = np.zeros((batch_size, self.max_length), dtype=np.int16)
        sen_seg = np.zeros((batch_size, self.max_length), dtype=np.int16)

        for i, data_instance in enumerate(data_batch):
            words = []
            sen_seg[i][0] = 1
            for sent in data_instance['sents']:
                words += sent
                sen_seg[i][len(words) - 1] = 1

            for j, word in enumerate(words):
                word = word.lower()
                if j < self.max_length:
                    if word in self.word2id:
                        sen_word[i][j] = self.word2id[word]
                    else:
                        sen_word[i][j] = self.word2id['UNK']
                    if sen_word[i][j] < 0:
                        raise ValueError("the id should not be negative")

            # Fill remaining vector with blank token id
            for j in range(j + 1, self.max_length):
                sen_word[i][j] = self.word2id['BLANK']

            vertex_set = data_instance['vertexSet']

            for idx, vertex in enumerate(vertex_set, 1):
                for v in vertex:
                    sen_pos[i][v['pos'][0]:v['pos'][1]] = idx
                    ner_type_B = self.ner2id[v['type']]
                    ner_type_I = ner_type_B + 1
                    sen_ner[i][v['pos'][0]] = ner_type_B
                    sen_ner[i][v['pos'][0] + 1:v['pos'][1]] = ner_type_I

        return {
            'word': sen_word,
            'pos': sen_pos,
            'ner': sen_ner,
            'seg': sen_seg
        }

    def get_bert_vectors(self, data_batch):
        batch_size = len(data_batch)
        bert_token = np.zeros((batch_size, self.max_length), dtype=np.int64)
        bert_mask = np.zeros((batch_size, self.max_length), dtype=np.int64)
        bert_starts = np.zeros((batch_size, self.max_length), dtype=np.int64)

        for i, data_instance in enumerate(data_batch):
            bert_token[i], bert_mask[i], bert_starts[i] = self.bert.subword_tokenize_to_ids(
                join_document(data_instance))

        return {
            'bert_token': bert_token,
            'bert_mask': bert_mask,
            'bert_starts': bert_starts
        }

    def get_mdp_vectors(self, data_batch):
        data = []
        batch_size = len(data_batch)

        node_position = np.zeros((batch_size, self.max_node_num, self.max_length), dtype=np.int16)
        node_sent_num = np.zeros((batch_size, self.max_sent_num), dtype=np.int16)
        node_num = np.zeros((batch_size, 1), dtype=np.int16)
        entity_position = np.zeros((batch_size, self.max_entity_num, self.max_length), dtype=np.int16)
        sdp_position = np.zeros((batch_size, self.max_entity_num, self.max_length), dtype=np.int16)
        sdp_num = np.zeros((batch_size, 1), dtype=np.int16)

        for i, data_instance in enumerate(data_batch):
            sentence_start_idx = [0]
            sentence_start_idx_counter = 0
            for sent in data_instance['sents']:
                sentence_start_idx_counter += len(sent)
                sentence_start_idx.append(sentence_start_idx_counter)

            node_num[i] = self.get_node_position(data_instance, node_position[i], node_sent_num[i], entity_position[i],
                                                 sentence_start_idx)
            self.extract_mdp_node(data_instance, sdp_position[i], sdp_num[i], sentence_start_idx)

            vertex_set = copy.deepcopy(data_instance['vertexSet'])
            # point position added with sent start position
            for j in range(len(vertex_set)):
                for k in range(len(vertex_set[j])):
                    vertex_set[j][k]['sent_id'] = int(vertex_set[j][k]['sent_id'])

                    sent_id = vertex_set[j][k]['sent_id']
                    dl = sentence_start_idx[sent_id]
                    pos1 = vertex_set[j][k]['pos'][0]
                    pos2 = vertex_set[j][k]['pos'][1]
                    vertex_set[j][k]['pos'] = [pos1 + dl, pos2 + dl]

            item = {'vertexSet': vertex_set}
            labels = data_instance.get('labels', [])

            train_triple = set()
            new_labels = []
            for label in labels:
                label['r'] = self.rel2id[label['r']]  # Replace with id
                train_triple.add((label['h'], label['t']))
                new_labels.append(label)

            item['labels'] = new_labels

            na_triple = []
            for h, t in h_t_idx_generator(len(vertex_set)):
                if (h, t) not in train_triple:
                    na_triple.append((h, t))

            item['na_triple'] = na_triple
            item['Ls'] = sentence_start_idx
            item['sents'] = data_instance['sents']
            data.append(item)

        return {
            'data': data,
            'node_position': node_position,
            'node_num': node_num,
            'node_sent_num': node_sent_num,
            'entity_position': entity_position,
            'sdp_position': sdp_position,
            'sdp_num': sdp_num
        }

    def get_node_position(self, data, node_position, node_sent_num, entity_position, sentence_start_idx):
        nodes = [[] for _ in range(len(data['sents']))]
        nodes_sent = [[] for _ in range(len(data['sents']))]

        for ns_no, ns in enumerate(data['vertexSet']):
            for n in ns:
                sent_id = int(n['sent_id'])
                doc_pos_s = n['pos'][0] + sentence_start_idx[sent_id]
                doc_pos_e = n['pos'][1] + sentence_start_idx[sent_id]
                assert (doc_pos_e <= sentence_start_idx[-1])
                nodes[sent_id].append([sent_id] + [ns_no] + [doc_pos_s, doc_pos_e])
                nodes_sent[sent_id].append([sent_id] + n['pos'])
        id = 0

        for ns in nodes:
            for n in ns:
                n.insert(0, id)  # Adds an index to the start of each node
                id += 1

        assert (id <= self.max_node_num)

        entity_num = len(data['vertexSet'])

        # generate entities(nodes) mask for document
        for ns in nodes:
            for n in ns:
                node_position[n[0]][n[3]:n[4]] = 1  # n[0] refers to the index added in the above loop

        # generate entities(nodes) mask for sentences in a document
        for sent_no, ns in enumerate(nodes_sent):
            assert (len(ns) < self.max_node_per_sent)
            node_sent_num[sent_no] = len(ns)

        # entity matrices
        for e_no, es in enumerate(data['vertexSet']):
            for e in es:
                sent_id = int(e['sent_id'])
                doc_pos_s = e['pos'][0] + sentence_start_idx[sent_id]
                doc_pos_e = e['pos'][1] + sentence_start_idx[sent_id]
                entity_position[e_no][doc_pos_s:doc_pos_e] = 1

        total_mentions = id

        total_num_nodes = total_mentions + entity_num
        assert (total_num_nodes <= self.max_node_num)

        return total_mentions  # only mentions

    def extract_mdp_node(self, data, sdp_pos, sdp_num, sentence_start_idx):
        """Extract meta dependency paths (MDP) node for each document
        """
        sents = data["sents"]
        nodes = [[] for _ in range(len(data['sents']))]
        sdp_lists = []
        # create mention's list for each sentence
        for ns_no, ns in enumerate(data['vertexSet']):
            for n in ns:
                sent_id = int(n['sent_id'])
                nodes[sent_id].append(n['pos'])

        for sent_no in range(len(sents)):
            spacy_sent = self.nlp(' '.join(sents[sent_no]))
            edges = []
            if len(spacy_sent) != len(sents[sent_no]):
                sdp_lists.append([])
                continue
            # make sure the length of sentence parsed by spacy is the same as original sentence.
            for token in spacy_sent:
                for child in token.children:
                    edges.append(('{0}'.format(token.i), '{0}'.format(child.i)))

            graph = nx.Graph(edges)  # Get the length and path

            mention_num = len(nodes[sent_no])
            sdp_list = []
            # get the shortest dependency path of all mentions in a sentence
            entity_indices = []
            for m_i in range(mention_num):  # m_i is the mention number
                indices_i = [nodes[sent_no][m_i][0] + offset for offset in
                             range(nodes[sent_no][m_i][1] - nodes[sent_no][m_i][0])]
                entity_indices = entity_indices + indices_i
                for m_j in range(mention_num):  #
                    if m_i == m_j:
                        continue
                    indices_j = [nodes[sent_no][m_j][0] + offset for offset in
                                 range(nodes[sent_no][m_j][1] - nodes[sent_no][m_j][0])]
                    for index_i in indices_i:
                        for index_j in indices_j:
                            try:
                                sdp_path = nx.shortest_path(graph, source='{0}'.format(index_i),
                                                            target='{0}'.format(index_j))
                            except (nx.NetworkXNoPath, nx.NodeNotFound):
                                continue
                            sdp_list.append(sdp_path)
            # get the sdp indices in a sentence
            sdp_nodes_flat = [sdp for sub_sdp in sdp_list for sdp in sub_sdp]
            entity_set = set(entity_indices)
            sdp_nodes_set = set(sdp_nodes_flat)
            # minus the entity node
            sdp_list = list(set([int(n) for n in sdp_nodes_set]) - entity_set)
            sdp_list.sort()
            sdp_lists.append(sdp_list)

        # calculate the sdp position in a document
        if len(sents) != len(sdp_lists):
            print("len mismatch")
        for i in range(len(sents)):
            if len(sdp_lists[i]) == 0:
                continue
            for j, sdp in enumerate(sdp_lists[i]):
                if j > len(sdp_lists[i]) - 1:
                    print("list index out of range")
                sdp_lists[i][j] = sdp + sentence_start_idx[i]

        flat_sdp = [sdp for sub_sdp in sdp_lists for sdp in sub_sdp]

        # set the sdp position as 1. for example, if the sdp_pos size is 100 X 512,
        # then we will set the value in each row as 1 according to flat_sdp[i]
        for i in range(len(flat_sdp)):
            if i > self.max_entity_num - 1:
                continue
            sdp_pos[i][flat_sdp[i]] = 1

        sdp_num[0] = len(flat_sdp)

    def save_preprocessed_data(self, preprocessed_data):
        for key, value in preprocessed_data.items():
            if key == 'data':
                json.dump(value, open(os.path.join(self.output_dir, self.output_file_prefix + '.json'), "w"))
            else:
                np.save(os.path.join(self.output_dir, self.output_file_prefix + f'_{key}.npy'), value)

    def _get_dis2idx(self):
        """Used to categorize distance between head-to-tail and tail-to-head entities."""
        dis2idx = np.zeros(self.max_length, dtype='int64')
        dis2idx[1] = 1
        dis2idx[2:] = 2
        dis2idx[4:] = 3
        dis2idx[8:] = 4
        dis2idx[16:] = 5
        dis2idx[32:] = 6
        dis2idx[64:] = 7
        dis2idx[128:] = 8
        dis2idx[256:] = 9
        return dis2idx

    def get_tensor_batch(self, np_vectors):
        batch_size = len(np_vectors['data'])

        # Init tensors
        h_mapping = torch.zeros(batch_size, self.h_t_limit, self.max_length, device=self.device)
        t_mapping = torch.zeros(batch_size, self.h_t_limit, self.max_length, device=self.device)
        relation_mask = torch.zeros(batch_size, self.h_t_limit, device=self.device)
        ht_pair_pos = torch.zeros(batch_size, self.h_t_limit, dtype=torch.long, device=self.device)

        if self.use_bert:
            context_idxs = torch.tensor(np_vectors['bert_token'], dtype=torch.long, device=self.device)
            context_masks = torch.tensor(np_vectors['bert_mask'], dtype=torch.long, device=self.device)
            context_starts = torch.tensor(np_vectors['bert_starts'], dtype=torch.long, device=self.device)
        else:
            context_idxs = torch.tensor(np_vectors['word'], dtype=torch.long, device=self.device)
        context_pos = torch.tensor(np_vectors['pos'], dtype=torch.long, device=self.device)
        context_ner = torch.tensor(np_vectors['ner'], dtype=torch.long, device=self.device)
        context_seg = torch.tensor(np_vectors['seg'], dtype=torch.long, device=self.device)
        node_position = torch.tensor(np_vectors['node_position'], dtype=torch.float, device=self.device)
        node_sent_num = torch.tensor(np_vectors['node_sent_num'], dtype=torch.long, device=self.device)
        node_num = torch.tensor(np_vectors['node_num'], dtype=torch.long, device=self.device)
        entity_position = torch.tensor(np_vectors['entity_position'], dtype=torch.float, device=self.device)
        sdp_position = torch.tensor(np_vectors['sdp_position'], dtype=torch.float, device=self.device)
        # Flattens a nested list.
        sdp_num = torch.tensor(np_vectors['sdp_num'], dtype=torch.long, device=self.device).flatten()
        sdp_num = sdp_num.clamp(max=self.max_entity_num).tolist()

        if self.is_train:
            relation_multi_label = torch.zeros(batch_size, self.h_t_limit, self.num_relations).to(
                device=self.device)

        max_h_t_cnt = 0
        entity_num = []
        sentence_num = []
        node_num_per_sent = []
        labels = []
        for i in range(batch_size):
            instance = np_vectors['data'][i]
            entity_num.append(len(instance['vertexSet']))
            sentence_num.append(len(instance['sents']))
            node_num_per_sent.append(max(node_sent_num[i].tolist()))

            if self.is_train:
                ht_idx2label_idx = defaultdict(list)
                for label in instance['labels']:
                    ht_idx2label_idx[(label['h'], label['t'])].append(label['r'])

            vertex_set_length = len(instance['vertexSet'])
            if vertex_set_length > 1:  # Need at least 2 entities to have relations
                for j, (h_idx, t_idx) in enumerate(h_t_idx_generator(vertex_set_length)):
                    hlist = instance['vertexSet'][h_idx]
                    tlist = instance['vertexSet'][t_idx]
                    for h in hlist:
                        h_mapping[i, j, h['pos'][0]:h['pos'][1]] = 1.0 / len(hlist) / (h['pos'][1] - h['pos'][0])
                    for t in tlist:
                        t_mapping[i, j, t['pos'][0]:t['pos'][1]] = 1.0 / len(tlist) / (t['pos'][1] - t['pos'][0])

                    relation_mask[i, j] = 1
                    delta_dis = hlist[0]['pos'][0] - tlist[0]['pos'][0]
                    if delta_dis < 0:
                        ht_pair_pos[i, j] = -int(self.dis2idx[-delta_dis])
                    else:
                        ht_pair_pos[i, j] = int(self.dis2idx[delta_dis])

                    if self.is_train:
                        # Fill relation labels
                        relation_indices = ht_idx2label_idx[(h_idx, t_idx)]
                        for r_idx in relation_indices:
                            relation_multi_label[i, j, r_idx] = 1

                max_h_t_cnt = max(max_h_t_cnt, j + 1)  # Because j is 0-based index, + 1 to get actual count

            # Labels for calculating metrics
            # Modified from original
            # Original stored a boolean value of whether the fact is found in training data
            label_set = set()
            for label in instance['labels']:
                label_set.add((label['h'], label['t'], label['r']))
            labels.append(label_set)

        input_lengths = (context_idxs > 0).sum(dim=1)
        max_c_len = int(input_lengths.max())  # max length of a document
        entity_mention_num = list(map(add, entity_num, node_num.squeeze(1).tolist()))
        max_sdp_num = max(sdp_num)
        max_entity_num = max(entity_num)
        max_sentence_num = max(sentence_num)
        b_max_mention_num = int(node_num.max())
        all_node_num = torch.LongTensor(list(map(add, sdp_num, entity_mention_num)))
        dis_h_2_t = ht_pair_pos + 10
        dis_t_2_h = -ht_pair_pos + 10

        tensor_batch = {
            'context_idxs': context_idxs[:, :max_c_len],
            'context_pos': context_pos[:, :max_c_len],
            'context_ner': context_ner[:, :max_c_len],
            'h_mapping': h_mapping[:, :max_h_t_cnt, :max_c_len],
            't_mapping': t_mapping[:, :max_h_t_cnt, :max_c_len],
            'relation_mask': relation_mask[:, :max_h_t_cnt],
            'dis_h_2_t': dis_h_2_t[:, :max_h_t_cnt],
            'dis_t_2_h': dis_t_2_h[:, :max_h_t_cnt],
            'context_seg': context_seg[:, :max_c_len],
            'node_position': node_position[:, :b_max_mention_num, :max_c_len],
            'entity_position': entity_position[:, :max_entity_num, :max_c_len],
            'node_sent_num': node_sent_num[:, :max_sentence_num],
            'all_node_num': all_node_num,
            'entity_num_list': entity_num,
            'sdp_position': sdp_position[:, :max_sdp_num, :max_c_len],
            'sdp_num_list': sdp_num,
            'labels': labels  # used by metrics only
        }

        if self.use_bert:
            tensor_batch['context_masks'] = context_masks[:, :max_c_len]
            tensor_batch['context_starts'] = context_starts[:, :max_c_len]

        if self.is_train:
            tensor_batch['relation_multi_label'] = relation_multi_label[:, :max_h_t_cnt]

        return tensor_batch
