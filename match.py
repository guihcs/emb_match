from torch.utils.data import Dataset

from om.match import aligns

from rdflib import Graph
from rdflib.term import URIRef, BNode


class MatchDataset(Dataset):

    def __init__(self, ref, o1, o2, tf=1):
        self.transform = None
        self.ref = ref
        self.o1 = o1
        self.o2 = o2
        als = aligns(ref)
        als = map(lambda x: (URIRef(x[0]), URIRef(x[1])), als)
        self.als = set(als)
        self.g1 = Graph()
        self.g1.parse(o1)
        self.g2 = Graph()
        self.g2.parse(o2)

        self.data = []
        t = 0
        f = 0

        for e1 in set(self.g1.subjects()):
            if type(e1) is BNode:
                continue
            for e2 in set(self.g2.subjects()):
                if type(e2) is BNode:
                    continue
                if (e1, e2) in self.als:
                    for _ in range(tf):
                        t += 1
                        self.data.append((e1, e2, 1))
                else:
                    f += 1
                    self.data.append((e1, e2, -1))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
