import codecs
import os
import sys

import collect_vocabularies as cv



MIN_WORDS_IN_DOC = 10

# Read vocabularies.
VOCABS_PATH = os.path.join('Datasets', 'vocabularies.txt')
VOCABS = [l.split('\t')[1].strip() for l in codecs.open(VOCABS_PATH, 'r', encoding='utf-8')]
VOCABS_COUNT = [int(l.split('\t')[0].strip()) for l in codecs.open(VOCABS_PATH, 'r', encoding='utf-8')]

NUM_TRAINING_DOCS = 200 if len(sys.argv) < 2 else int(sys.argv[1])
NUM_TEST_DOCS = len(VOCABS) - NUM_TRAINING_DOCS



def VocabIndexSorter(a, b):
    a_idx = VOCABS.index(a[0].strip())
    b_idx = VOCABS.index(b[0].strip())
    if VOCABS_COUNT[a_idx] < VOCABS_COUNT[b_idx]:
        return 1
    if VOCABS_COUNT[a_idx] > VOCABS_COUNT[b_idx]:
        return -1
    if a[0] < b[0]:
        return 1
    if a[0] > b[0]:
        return -1
    return 0



def WriteDocs(f, doc_paths, num_training):
    num_corpus_docs = 0
    for doc_idx in range(len(doc_paths)):
        # Read document.
        t = doc_paths[doc_idx]
        doc = [(l.strip().split('\t')[1], l.strip().split('\t')[0]) for l in codecs.open(t, 'r', encoding='utf-8')]

        # Remove unwanted words
        i = 0
        while i < len(doc):
            if doc[i][0] not in VOCABS:
                del doc[i]
                continue
            i += 1

        if len(doc) < MIN_WORDS_IN_DOC:
            print('Document `{}\' length is less than {}!'.format(t, MIN_WORDS_IN_DOC))
            continue

        # Write to corpus.
        words = sorted(doc, cmp=VocabIndexSorter)
        f.write('{} '.format(len(words)))
        f.write('{}:{}'.format(VOCABS.index(words[0][0]), words[0][1]))
        for w in words[1:]:
            f.write(' {}:{}'.format(VOCABS.index(w[0]), w[1]))
        if num_corpus_docs + 1 < num_training:
            f.write('\n')

        num_corpus_docs += 1
        if num_corpus_docs >= num_training:
            break

    return num_corpus_docs



if __name__ == '__main__':
    print('Num vocabularies: {}'.format(len(VOCABS)))
    print('Num training set: {}'.format(NUM_TRAINING_DOCS))

    # Collect file name.
    word_text_paths = cv.CollectFileNames(os.path.join('Datasets', 'world', 'texts'))
    print('Num word docs:    {}'.format(len(word_text_paths)))
    economy_text_paths = cv.CollectFileNames(os.path.join('Datasets', 'economy', 'texts'))
    print('Num economy docs: {}'.format(len(economy_text_paths)))

    # Make corpus.
    with open(os.path.join('Datasets', 'farsnews_corpus.txt'), 'w') as f:
        print('Writing word documents . . .')
        print(WriteDocs(f, word_text_paths, NUM_TRAINING_DOCS))
        f.write('\n')
        print('Writing economy documents . . .')
        print(WriteDocs(f, economy_text_paths, NUM_TRAINING_DOCS))

