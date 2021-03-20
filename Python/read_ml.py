import os
import sys

import numpy as np



MOVIE_LENS_PATH = os.path.join('Datasets', 'MovieLens', 'ml-1m', 'ratings.dat')
TEST_INDICES_PATH = os.path.join('Datasets', '1m-indices.txt')
TEST_PATH = os.path.join('Datasets', '1m-test.txt')
TRAIN_PATH = os.path.join('Datasets', '1m-train.txt')



def ReadMovieLens(path):
    data_set = []
    mx_film = -1
    f = open(path, 'r')
    for l in f:
        l = l.strip()

        # User ID
        u_st = l.index('::')
        u_idx = int(l[0:u_st])

        # Movie ID
        m_st = l.index('::', u_st + 2)
        m_idx = int(l[u_st + 2 : m_st])
        mx_film = max(mx_film, m_idx)

        # Rating
        r_st = l.index('::', m_st + 2)
        u_rate = int(l[m_st + 2 : r_st])

        data_set.append((u_idx - 1, m_idx - 1, u_rate))
    f.close()
    return data_set, mx_film



if __name__ == '__main__':
    # Read dataset.
    ml_dataset, mx_film = ReadMovieLens(MOVIE_LENS_PATH)
    print('len(dataset) : ', len(ml_dataset))
    print('dataset[0]   :', ml_dataset[0])
    print('dataset[-1]  :', ml_dataset[-1])
    print('max movie    :', mx_film + 1)

    # Make rating matrix.
    num_users = ml_dataset[-1][0] + 1
    num_movies = mx_film + 1
    print(num_users, num_movies)
    mat = np.zeros((num_users, num_movies), dtype=int)
    for rec in ml_dataset:
        u_idx = rec[0]
        m_idx = rec[1]
        rate = rec[2]
        if mat[u_idx][m_idx] != 0:
            print('mat[u_idx:{}][m_idx:{}] = rate:{}'.format(u_idx, m_idx, rate))
        mat[u_idx][m_idx] = rate

    # Count num of ratings and find positive ratings.
    rate_counts = np.count_nonzero(mat, 1)
    print('mat.shape         : {}'.format(mat.shape))
    print('rate_counts.shape : {}'.format(rate_counts.shape))
    num_neg_users = 0
    neg_users = []
    for y in range(mat.shape[0]):
        rate_cnt = 0
        num_gt_4_rates = 0
        for x in range(mat.shape[1]):
            rate = mat[y][x]
            rate_cnt += 1 if rate != 0 else 0
            num_gt_4_rates += 1 if rate > 3 else 0
        if num_gt_4_rates < 10:
            num_neg_users += 1
            neg_users.append(y)
            print('negative user #{}  ->  user:{}  rate_cnt:{}   num+4:{}'.format(num_neg_users, y, rate_cnt, num_gt_4_rates))
    print('min_rate_cnt', min(rate_counts), 'max_rate_cnt', max(rate_counts))
    print('num_neg_users', num_neg_users)

    # Calculate number of test users.
    num_rem_users = num_users - num_neg_users
    num_tests = 200
    percent = int(float(num_tests) / num_rem_users * 100.0)
    print('{}% users for test : {} of {}'.format(percent, num_tests, num_rem_users))

    # Select test users.
    test_users = []
    while len(test_users) < num_tests:
        r = int(np.random.uniform(0, num_users))
        if r in neg_users + test_users:
            continue
        test_users.append(r)
    test_users = sorted(test_users)
    print('len(test_users):', len(test_users), 'num_tests:', num_tests)
    print('min(test_users):', min(test_users))
    print('max(test_users):', max(test_users))

    # Dump test users indices and test corpus.
    f_idx = open(TEST_INDICES_PATH, 'w')
    f_corpus = open(TEST_PATH, 'w')
    for u in range(len(test_users)):
        u_idx = test_users[u]

        # Write selected test users indices and select a held-out movie.
        held_out = int(np.random.uniform(0, rate_counts[u_idx]))
        f_idx.write('{} {}'.format(u_idx, held_out))

        # Write ratings as corpus.
        w = np.where(mat[u_idx] != 0)[0]
        r_idx = w[0]
        f_corpus.write('{} '.format(rate_counts[u_idx]))
        f_corpus.write('{}:{}'.format(r_idx, mat[u_idx][r_idx]))
        for r in range(1, len(w)):
            r_idx = w[r]
            f_corpus.write(' {}:{}'.format(r_idx, mat[u_idx][r_idx]))

        if u + 1 < len(test_users):
            f_idx.write('\n')
            f_corpus.write('\n')
    f_idx.close()
    f_corpus.close()

    # Dump train corpus.
    filtered_users = test_users + neg_users
    print('filtered_users : ', len(filtered_users), len(set(filtered_users)))
    print('total train    : ', mat.shape[0] - len(filtered_users))
    f_corpus = open(TRAIN_PATH, 'w')
    for y in range(mat.shape[0]):
        if y in filtered_users:
            continue

        u_idx = y
        w = np.where(mat[u_idx] != 0)[0]
        r_idx = w[0]
        f_corpus.write('{} '.format(rate_counts[u_idx]))
        f_corpus.write('{}:{}'.format(r_idx, mat[u_idx][r_idx]))
        for r in range(1, len(w)):
            r_idx = w[r]
            f_corpus.write(' {}:{}'.format(r_idx, mat[u_idx][r_idx]))

        if y + 1 < mat.shape[0]:
            f_corpus.write('\n')
    f_corpus.close()

