import os
import codecs



MIN_WORD_COUNT = 25



def CollectFileNames(base_dir):
    path_list = []
    for dirpath, dirnames, filenames in os.walk(base_dir):
        path_list += [os.path.join(dirpath, file_name) for file_name in filenames]
    return path_list



def CountSorter(a, b):
    if a[1] < b[1]:
        return 1
    if a[1] > b[1]:
        return -1
    if a[0] < b[0]:
        return 1
    if a[0] > b[0]:
        return -1
    return 0



if __name__ == '__main__':
    # Collect file names.
    text_paths = CollectFileNames(os.path.join('Datasets', 'world', 'texts'))
    text_paths += CollectFileNames(os.path.join('Datasets', 'economy', 'texts'))
    print('Num files : {}'.format(len(text_paths)))

    # Concat all words and counts in all documents.
    words_dict = {}
    for text_path in text_paths:
        with codecs.open(text_path, 'r', encoding='utf-8') as f:
            for l in f:
                sp = l.strip().split('\t')
                if words_dict.has_key(sp[1]):
                    words_dict[sp[1]] += int(sp[0])
                else:
                    words_dict[sp[1]] = int(sp[0])

    # Write vocabularies to file.
    sorted_words = sorted([(k, words_dict[k]) for k in words_dict], cmp=CountSorter)
    num_unique_words = 0
    with open(os.path.join('Datasets', 'vocabularies.txt'), 'wb') as f:
        for w in sorted_words:
            if w[1] < MIN_WORD_COUNT:
                continue

            s = u'{:3d}\t'.format(w[1]) + w[0] + u'\r\n'
            f.write(s.encode('utf-8'))
            num_unique_words += 1

    print('Num unique words : {}'.format(num_unique_words))

