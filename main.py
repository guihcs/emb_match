from rdflib.term import BNode

from om.ont import get_n, tokenize
from om.util import WordMap, Cross
from om.match import onts, Step, Runner
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pymagnitude import Magnitude
from match import MatchDataset
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import sys
import os

def build_modules(options, configs, module_builders, config=None):
    if config is None:
        config = {}
    modules = []

    for o, cf, mb in zip(options, configs, module_builders):
        config.update(cf[o])
        modules.append(mb[o](config))

    return modules, config


def embed_graph(g1, modules):
    features = {x: x for x in
                filter(lambda x: type(x) is not BNode, set(g1.subjects()).union(g1.predicates()).union(g1.objects()))}

    for m in modules:
        for x in features:
            features[x] = m(x, features, g1)
    return features


def entity_feature(x, f, g):
    return [get_n(x, g)]


def word_feature(x, f, g):
    n = get_n(x, g)
    return list(map(str.lower, tokenize(n)))


def char_feature(x, f, g):
    n = get_n(x, g)
    return list(map(str.lower, n))


class LearnEmbedding(nn.Module):

    def __init__(self, vocab, emb_size, **kwargs):
        super(LearnEmbedding, self).__init__()
        self.emb = nn.Embedding(len(vocab) + 1, emb_size, 0)
        self.wm = WordMap(vocab)

    def forward(self, x, f, g):
        im = list(map(lambda q: self.wm[q] + 1 if q in self.wm else 0, f[x]))
        return self.emb(torch.LongTensor(im))


class PretrainedEmbedding(nn.Module):

    def __init__(self, emb_path, **kwargs):
        super(PretrainedEmbedding, self).__init__()
        self.emb = Magnitude(emb_path)

    def forward(self, x, f, g):
        embs = map(lambda x: self.emb.query(x), f[x])
        tembs = map(lambda x: torch.from_numpy(x).float().unsqueeze(0), embs)
        res = torch.cat(list(tembs), dim=0)
        return res


def sum_aggregation(x, f, g):
    return torch.sum(f[x], dim=0, keepdim=True)


def mean_aggregation(x, f, g):
    return torch.mean(f[x], dim=0, keepdim=True)


class LSTMAggregation(nn.Module):

    def __init__(self, emb_size, **kwargs):
        super(LSTMAggregation, self).__init__()
        self.emb_size = emb_size
        self.lstm = nn.LSTM(emb_size, emb_size, batch_first=True)

    def forward(self, x, f, g):
        inp = f[x].unsqueeze(0)

        output, (hn, cn) = self.lstm(inp)
        return output.squeeze(0)[-1].unsqueeze(0)


class TransformerAggregation(nn.Module):

    def __init__(self, emb_size, **kwargs):
        super(TransformerAggregation, self).__init__()
        self.emb_size = emb_size
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

    def forward(self, x, f, g):
        out = self.transformer_encoder(f[x])
        return torch.sum(out, dim=0, keepdim=True)


class GNN(nn.Module):

    def __init__(self, emb_size, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.nw = nn.Linear(self.emb_size * 3, self.emb_size)
        self.hw = nn.Linear(self.emb_size * 2, self.emb_size)

    def forward(self, x, f, g):
        h = f[x]
        nb = []
        for s, p, o in g.triples((x, None, None)):
            hf = f[p] if p in f else torch.zeros((1, self.emb_size))
            nf = f[o] if o in f else torch.zeros((1, self.emb_size))
            nb.append(torch.cat([h, hf, nf], dim=1))

        if len(nb) <= 0:
            nb.append(torch.zeros(1, self.emb_size * 3))

        nb = torch.cat(nb)
        nbf = nn.functional.leaky_relu(self.nw(nb), 0.2)
        nbf = torch.sum(nbf, dim=0, keepdim=True)
        res = torch.cat([h, nbf], dim=1)
        fh = nn.functional.leaky_relu(self.hw(res), 0.2)
        return fh


class GCN(nn.Module):

    def __init__(self, emb_size, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.nw = nn.Linear(self.emb_size, self.emb_size)

    def forward(self, x, f, g):
        nb = [f[x]]
        i = 1
        for s, p, o in g.triples((x, None, None)):
            nf = f[o] if o in f else torch.zeros((1, self.emb_size))
            nb.append(nf)
            i += 1

        nb = torch.sum(torch.cat(nb), dim=0, keepdim=True) / torch.sqrt_(torch.Tensor([i]))
        res = nn.functional.leaky_relu(self.nw(nb), 0.2)
        return res


class GAH(nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size
        self.w = nn.Linear(self.emb_size, self.emb_size)
        self.a = nn.Linear(self.emb_size * 2, 1)

    def forward(self, x, f, g):
        h = f[x]
        nb = [h]

        for s, p, o in g.triples((x, None, None)):
            nf = f[o] if o in f else torch.zeros((1, self.emb_size))
            nb.append(nf)

        nb = torch.cat(nb)
        wh = self.w(h).repeat(nb.shape[0], 1)
        cnb = torch.cat([wh, self.w(nb)], dim=1)

        at = nn.functional.leaky_relu(self.a(cnb), 0.2)
        res = at * nb
        return torch.sum(res, dim=0, keepdim=True)


class GAT(nn.Module):

    def __init__(self, emb_size, h=1, **kwargs):
        super().__init__()
        self.emb_size = emb_size
        self.ah = nn.ModuleList([GAH(emb_size) for _ in range(h)])

    def forward(self, x, f, g):
        r = [ah(x, f, g) for ah in self.ah]

        return torch.mean(torch.cat(r, dim=0), dim=0, keepdim=True)


def identity(x, f, g):
    return f[x]


def fit_modules(datasets, modules, each_config):
    ml = nn.ModuleList([modules[i] for i in range(len(modules)) if 'supervised' in each_config[i]])
    optimizer = optim.Adam(ml.parameters(), lr=0.003, weight_decay=1.0)
    crit = nn.CosineEmbeddingLoss(margin=0.5)

    lh = []

    for e in range(2):
        el = []
        random.shuffle(datasets)
        for dataset in datasets:
            dl = []
            for e1, e2, s in DataLoader(dataset, batch_size=bs, shuffle=True):
                optimizer.zero_grad()
                features_g1 = embed_graph(dataset.g1, modules)
                features_g2 = embed_graph(dataset.g2, modules)

                embs1 = list(map(lambda x: features_g1[x], e1))
                embs2 = list(map(lambda x: features_g2[x], e2))
                em1 = torch.cat(embs1)
                em2 = torch.cat(embs2)

                loss = crit(em1, em2, s)
                loss.backward()
                dl.append(loss.item())
                nn.utils.clip_grad_norm_(ml.parameters(), 1.0)
                optimizer.step()

            el.append(sum(dl) / len(dl))

        lh.append(sum(el) / len(el))

    return lh


class ModuleMatcher(Step):

    def __init__(self, modules):
        self.cross = Cross()
        self.modules = modules

    def forward(self, dataset, i):

        ents = self.cross(dataset)
        rang = np.arange(0.1, 1, 0.01)
        res = [[] for r in rang]
        with torch.no_grad():
            emb1 = embed_graph(dataset.g1, self.modules)
            emb2 = embed_graph(dataset.g2, self.modules)

        for e1, e2 in ents:

            with torch.no_grad():
                eb1 = emb1[e1]
                eb2 = emb2[e2]
                s = torch.cosine_similarity(eb1, eb2)
            for i in range(len(rang)):
                sim = 1 if s >= rang[i] else 0
                res[i].append((e1, e2, sim))

        return res, {}


def rank(results):
    res = []
    for k in results:
        data = []
        for r in results[k]:
            m = r[['precision', 'recall', 'f1']].mean()
            data.append([m['precision'], m['recall'], m['f1']])
        res.append([k] + max(data, key=lambda x: x[2]))

    df = pd.DataFrame(res, columns=['Name', 'Precision', 'Recall', 'F1'])

    return df.sort_values('F1', ascending=False)


def get_vocab_t(g, t):
    vocab = set()
    for s, p, o in g:
        vocab.update(set(t(s, {}, g)))
        vocab.update(set(t(p, {}, g)))
        vocab.update(set(t(o, {}, g)))

    return vocab


base = sys.argv[2]
reference = sys.argv[3]
result_file = f'emb_result/result_{sys.argv[1]}.csv'
emb_base = sys.argv[4]

if not os.path.exists('emb_result'):
    os.mkdir('emb_result')

options = list(map(int, list(sys.argv[1])))
feature_view_configs = [{}, {}, {}]
embedding_model_configs = [{'emb_size': 50, 'supervised': True},
                           {'emb_size': 300, 'emb_path': emb_base + '/glove.magnitude'},
                           {'emb_size': 300, 'emb_path': emb_base + '/word2vec.magnitude'},
                           {'emb_size': 300, 'emb_path': emb_base + '/fasttext.magnitude'}]
aggregation_configs = [{}, {}, {'supervised': True, 'batch': True}, {'supervised': True}]
graph_aggregation_configs = [{}, {'supervised': True}, {'supervised': True}, {'supervised': True}]

feature_view = [lambda x: entity_feature, lambda x: word_feature, lambda x: char_feature]
embedding_model = [lambda x: LearnEmbedding(**x), lambda x: PretrainedEmbedding(**x),
                   lambda x: PretrainedEmbedding(**x), lambda x: PretrainedEmbedding(**x)]
aggregation = [lambda x: sum_aggregation, lambda x: mean_aggregation, lambda x: LSTMAggregation(**x),
               lambda x: TransformerAggregation(**x)]
graph_aggregation = [lambda x: identity, lambda x: GNN(**x), lambda x: GCN(**x), lambda x: GAT(**x)]

configs = [feature_view_configs, embedding_model_configs, aggregation_configs, graph_aggregation_configs]
module_builders = [feature_view, embedding_model, aggregation, graph_aggregation]

supervised = any(map(lambda x: 'supervised' in x[1][x[0]], zip(options, configs)))

if supervised:
    each_config = list(map(lambda x: x[1][x[0]], zip(options, configs)))
    ds = list(onts(base, reference))
    res = KFold(n_splits=3, shuffle=True).split(ds)
    bs = 128
    ranks = []
    for fi, (train, test) in enumerate(res):

        tokenizer = feature_view[options[0]]({})
        vocab = set()
        datasets = []
        for t in train:
            dataset = MatchDataset(*ds[t], tf=800)
            vocab.update(get_vocab_t(dataset.g1, tokenizer))
            vocab.update(get_vocab_t(dataset.g2, tokenizer))
            datasets.append(dataset)

        config = {'vocab': vocab}

        modules, config = build_modules(options, configs, module_builders, config)
        lh = fit_modules(datasets, modules, each_config)

        refs = [ds[i][0] for i in test]
        runner = Runner(base, reference, matcher=ModuleMatcher(modules))
        result = runner.run(parallel=False, refs=refs)
        for i in range(len(result)):
            result[i]['th'] = 0.1 + i / 100

        fold = pd.concat(result)
        fold['fold'] = fi
        ranks.append(fold)

    final_result = pd.concat(ranks)
    final_result['mode'] = 'supervised'

    final_result.to_csv(result_file)
else:
    modules, config = build_modules(options, configs, module_builders)
    runner = Runner(base, reference, matcher=ModuleMatcher(modules))
    result = runner.run(parallel=False)
    for i in range(len(result)):
        result[i]['th'] = 0.1 + i / 100

    fold = pd.concat(result)
    fold['fold'] = 0
    fold['mode'] = 'unsupervised'
    fold.to_csv(result_file)
