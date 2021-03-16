import codecs
import os

import numpy as np



NUM_TOP_WORDS = 30
VOCABS_PATH = os.path.join('Datasets', 'vocabularies.txt')
TOP_WORDS_PATH_FMT = os.path.join('Datasets', 'top-' + str(NUM_TOP_WORDS) + '_k{}.txt')



if __name__ == '__main__':
    beta = []
    with open('beta.txt', 'r') as f:
        for l in f:
            l = l.strip()
            if not len(l):
                continue

            new_line = l.replace('  ', ' ')
            while len(new_line) != len(l):
                l = new_line
                new_line = l.replace('  ', ' ')

            beta.append([ float(b) for b in l.split()[2:] ])
    beta = np.array(beta)
    idx = np.argsort(beta)

    vocabularies = [ l for l in codecs.open(VOCABS_PATH, 'r', encoding='utf-8') ]
    for k in range(idx.shape[0]):
        NUM_TOP_WORDS_IDX = NUM_TOP_WORDS + 1
        print(idx[k][-1:-NUM_TOP_WORDS_IDX:-1])
        with codecs.open(TOP_WORDS_PATH_FMT.format(k), 'w', encoding='utf-8') as f:
            top_words = u''.join([ vocabularies[v] for v in idx[k][-1:-NUM_TOP_WORDS_IDX:-1] ])
            f.write(top_words)

