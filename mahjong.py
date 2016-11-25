# coding: utf-8

import numpy as np
from scipy import signal
import random
import os
import csv

n_dots = 9  # 筒牌[0,..,8]
n_bamboos = 9  # 条牌[9,..,17]
n_characters = 9  # 万牌[18,..,26]
n_winds = 4  # 风牌 东、南、西、北[27,..,30]
n_dragons = 3  # 中、发、白[31,32,33]
n_honors = n_dragons + n_winds
n_suits = n_dots + n_bamboos + n_characters

r_dots = list(range(0, 9))
r_bamboos = list(range(9, 18))
r_characters = list(range(18, 27))
r_winds = list(range(27, 31))
r_dragons = list(range(31, 34))
uni_tiles = [r_dots, r_bamboos, r_characters, r_winds, r_dragons]
flatten_tiles = [i for sub in uni_tiles for i in sub]
n_tiles = len(flatten_tiles)
all_tiles = flatten_tiles * 4

tiles_str = u'\U0001F019\U0001F01A\U0001F01B\U0001F01C\U0001F01D\U0001F01E\U0001F01F\U0001F020\U0001F021\U0001F010\U0001F011\U0001F012\U0001F013\U0001F014\U0001F015\U0001F016\U0001F017\U0001F018\U0001F007\U0001F008\U0001F009\U0001F00A\U0001F00B\U0001F00C\U0001F00D\U0001F00E\U0001F00F\U0001F000\U0001F001\U0001F002\U0001F003\U0001F004\U0001F005\U0001F006'
the_king = '* '

n_player_tiles = 14

# any win patterns must meet $k DD + m AAA + n ABC$, which $k, m, n \in \mathbb{Z}_{\geq 0}$
# k = 1, m + n = 4
# <font color='grey'>~~k = 4, m + n = 2~~</font>
# k = 7, m + n = 0
# proper values:
#
# | k | m | n
# |:- |:- |:-
# | 1 | 0 | 4
# | 1 | 1 | 3
# | 1 | 2 | 2
# | 1 | 3 | 1
# | 1 | 4 | 0
# | 7 | 0 | 0
#

# all improper patterns must break all above equations.

# In[4]:

def deal_tiles():
    b = np.random.choice(all_tiles, n_player_tiles, replace=False)
    return np.sort(b)

def who_is_the_king():
    return np.random.randint(n_tiles)

def to_str(tiles):
    return ''.join([tiles_str[i] for i in tiles])

def gen_pairs():
    return [[i] * 2 for i in flatten_tiles]

def gen_triplets():
    return [[i] * 3 for i in flatten_tiles]

def gen_kongs():
    return [[i] * 4 for i in flatten_tiles]

def gen_seqs():
    pd = [[i, i + 1, i + 2] for i in r_dots[:-2]]
    pb = [[i, i + 1, i + 2] for i in r_bamboos[:-2]]
    pc = [[i, i + 1, i + 2] for i in r_characters[:-2]]
    return pd + pb + pc

def _find_subseq(seq, sub):
    print(seq)
    print(sub)
    assert seq.size >= sub.size
    target = np.dot(sub, sub)
    # print('target', target)
    candidates = np.where(np.correlate(seq, sub) == target)[0]
    # print('candidates', candidates)
    # some of the candidates entries may be false positives, double check
    check = candidates[:, np.newaxis] + np.arange(len(sub))
    # print('check', check)
    mask = np.all((np.take(seq, check) == sub), axis=-1)
    # print('mask', mask)
    return candidates[mask]

def gen_7_pairs():
    s = np.random.choice(flatten_tiles, 7, replace=False)
    r = [[i] * 2 for i in s]
    r = [i for e in r for i in e]
    assert len(r) == n_player_tiles
#     r = sorted(r)
    return r

def gen_4_triplets():
    pass

def gen_a_proper_pat():
    # DD + mAAA + nABC, m + n == 4, m n are elements of Z*

    all_seqs = gen_seqs()
    m = random.randint(0, 4)
    n = 4 - m

    x = random.sample(flatten_tiles, m + 1)
    d = x[0]
    r = [[d, d]]
    aaa = x[1:]
    r += [[i] * 3 for i in aaa]

    for i in range(n):
        while True:
            ju = random.sample(all_seqs, 1)
            cur = r + ju
            cur = np.array([i for e in cur for i in e])
            bc = np.bincount(cur)
            if np.all(bc <= 4):
                r += ju
                break

#     print(r)
    r = [i for e in r for i in e]
    assert len(r) == n_player_tiles
#     r = sorted(r)
    return r


# a = gen_a_proper_pat()
# print(a)

# print(tiles_str)
# a = np.array([8,9,11,11,12,17,17,18,21,22,22,23,28,32])
# a = deal_tiles()
# print(a)
# print(to_str(a))

# a = gen_seqs()
# print(gen_seqs())
# print(gen_7_pairs())
# a = np.array(a)
# print(a.shape)
# b = np.dot(a, a.T)

# sub = np.array([21,22,23])
# _find_subseq(a, sub)
# print(np.dot(sub, sub))
# print(np.correlate(a, sub, mode='valid'))
# print(np.convolve(a, sub[::-1], mode='valid'))
# print(signal.correlate(a, sub, mode='valid'))
# print(signal.convolve(a, sub[::-1], mode='valid'))
# print(signal.correlate2d(a[np.newaxis, :], sub[np.newaxis, :], mode='valid'))
# print(signal.convolve2d(a[np.newaxis, :], sub[np.newaxis, ::-1], mode='valid'))

# a = np.array([[ 1,  3,  4,  4,  5, 12, 21, 21, 22, 25, 25, 27, 29, 31],
#               [ 4,  6,  9, 17, 20, 21, 22, 22, 23, 27, 27, 28, 32, 33],
#               [ 0,  2,  4,  7,  8, 10, 12, 14, 21, 22, 22, 22, 23, 31],
#               [ 2,  5,  7, 10, 12, 13, 16, 21, 22, 23, 26, 28, 30, 31],
#               [ 4, 10, 11, 13, 16, 20, 21, 22, 24, 26, 27, 28, 29, 32]])
# print(signal.correlate2d(a, sub[np.newaxis, :]))


def check_hu(tiles):
    bc = np.zeros(n_tiles, dtype=int)
    b = np.bincount(tiles)
    bc[:b.shape[0]] = b
    clusters = []
    return reduce1(bc, 0, clusters)

def reduce_seq(bc, i, suits):
    if i not in suits:
        return False, None
    if i > suits[-1] - 2:
        return False, None
    if bc[i + 1] == 0 or bc[i + 2] == 0:
        return False, None
    bc[i] -= 1
    bc[i + 1] -= 1
    bc[i + 2] -= 1
    seq = [i, i + 1, i + 2]
#     print('-seq', seq, i)
    return True, seq

def find_seq(bc, i):
    r, t = False, None
    if i in r_dots:
        r, t = reduce_seq(bc, i, r_dots)
    elif i in r_bamboos:
        r, t = reduce_seq(bc, i, r_bamboos)
    elif i in r_characters:
        r, t = reduce_seq(bc, i, r_characters)
    return r, t

def reduce_n(bc, i, n):
    assert i in flatten_tiles
    assert n == 2 or n == 3
    if bc[i] < n:
        return False, None
    bc[i] -= n
    gang_of_n = [i] * n
#     print(' -%d:' % n, gang_of_n, i)
    return True, gang_of_n

def backtracking(bc, subpat):
    for i in subpat:
        bc[i] += 1
#     print('undo:', bc)

def reap_again(bc, i, clusters, t):
    clusters.append(t)
    r = reduce(bc, i, clusters)
    backtracking(bc, t)
    del clusters[-1]
    return r

# if a triplet was found, it is unnecessary to find other patterns further
# 2 = 1 + 1
# 3 = 1 + 2
# 4 = 1 + 3 = 1 + 1 + 2 = 2 + 2 = 1 + 1 + 1 + 1
def reduce(bc, start_i, clusters):
    assert bc.size == n_tiles

    sum0 = np.sum(bc)
#     print(bc, sum0)
    if sum0 == 0:
        n = len(clusters)
        if n != 5 and n != 7:
            return False
        print('ok:', clusters)
        return True

    for i in range(start_i, n_tiles):
        if bc[i] == 0:
            continue

        ok = False
        r, t = find_seq(bc, i)
        if r:
            ok = reap_again(bc, i, clusters, t) or ok
#             if ok: return ok
        if bc[i] == 2 or bc[i] == 4:
            r, t = reduce_n(bc, i, 2)
            if r:
                ok = reap_again(bc, i, clusters, t) or ok
#                 if ok: return ok
        if bc[i] == 3:
            r, t = reduce_n(bc, i, 3)
            if r:
                ok = reap_again(bc, i, clusters, t) or ok
#                 if ok: return ok
        return ok

def reduce1(bc, start_i, clusters):
    assert bc.size == n_tiles

    sum0 = np.sum(bc)
#     print(bc, sum0)
    if sum0 == 0:
        n = len(clusters)
        if n != 5 and n != 7:
            return False
        print('ok:', clusters)
        return True

    i = start_i
    while True:
        if i >= n_tiles:
            break
        if bc[i] != 0:
            break
        i += 1

    if i >= n_tiles:
        return False

    got_it = False
    tried = np.zeros(3, bool)
    while True:
        if np.all(tried):
            break

        r, t = False, []
        for _ in range(1):
            if not tried[0]:
                r, t = find_seq(bc, i)
                tried[0] = True
                if r:
                    break
            if not tried[1]:
                if bc[i] == 2 or bc[i] == 4:
                    r, t = reduce_n(bc, i, 2)
                tried[1] = True
                if r:
                    break
            if not tried[2]:
                if bc[i] == 3:
                    r, t = reduce_n(bc, i, 3)
                tried[2] = True
                if r:
                    break
        if r:
            clusters.append(t)
            r = reduce1(bc, i, clusters)
            backtracking(bc, t)
            del clusters[-1]
            if r:
                got_it = True
                just_one_solution = True
                if just_one_solution:
                    break
    return got_it


def make_dataset(amount):
    propor_1pair_hu = 0.45
    porpor_7pairs = propor_1pair_hu * 0.01
    n_1pair_hu = int(amount * propor_1pair_hu)
    n_7pairs = int(amount * porpor_7pairs)

    ds_tiles, ds_hu_labels = [], []
    for i in range(n_1pair_hu):
        a = gen_a_proper_pat()
        ds_tiles.append(a)
        ds_hu_labels.append((1,))
    for i in range(n_7pairs):
        a = gen_7_pairs()
        ds_tiles.append(a)
        ds_hu_labels.append((1,))
    while len(ds_tiles) < amount:
        a = deal_tiles()
        hu = check_hu(a)
        ds_tiles.append(a)
        ds_hu_labels.append((int(hu),))

    return np.array(ds_tiles), np.array(ds_hu_labels)

def make_dataset_to_files(amount, file_dir, file_name_prefix):
    ds_tiles, ds_labels = make_dataset(amount)
    assert ds_tiles.shape[0] == ds_labels.shape[0]
    index = np.arange(ds_tiles.shape[0])
    np.random.shuffle(index)
    
    p = int(index.shape[0] * 0.8)
    test = index[p:]
    index = index[:p]
    p = int(index.shape[0] * 0.8)
    valid = index[p:]
    train = index[:p]
    
    file = os.path.join(file_dir, file_name_prefix + '_train_tiles.txt')
    save_to_file(file, ds_tiles[train])
    file = os.path.join(file_dir, file_name_prefix + '_train_labels.txt')
    save_to_file(file, ds_labels[train])
    file = os.path.join(file_dir, file_name_prefix + '_valid_tiles.txt')
    save_to_file(file, ds_tiles[valid])
    file = os.path.join(file_dir, file_name_prefix + '_valid_labels.txt')
    save_to_file(file, ds_labels[valid])
    file = os.path.join(file_dir, file_name_prefix + '_test_tiles.txt')
    save_to_file(file, ds_tiles[test])
    file = os.path.join(file_dir, file_name_prefix + '_test_labels.txt')
    save_to_file(file, ds_labels[test])
     
def save_to_file(out_file, a):
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for r in a:
            writer.writerow(r)
            
# make_dataset_to_files(500000, '/home/splendor/fusor/dataset_mahjong', 'mj')   

# ds_tiles, ds_labels = make_dataset(10000)
# print(ds_tiles.shape, ds_labels.shape)
# assert ds_tiles.shape[1] == n_player_tiles
# assert ds_tiles.shape[0] == ds_labels.shape[0]
# print(ds_tiles)
# print(ds_labels)

# from timeit import default_timer as timer
# start = timer()

cnt = 0
total = 1
for i in range(total):
    a = gen_a_proper_pat()

#     a = np.array([1, 1, 1, 2, 3, 4, 5, 5, 5, 6, 7, 12, 13, 14])
#     a = np.array([1, 1, 2, 3, 4, 5, 5, 5, 5, 6, 7, 12, 13, 14])
#     a = np.array([0, 0, 1, 1, 2, 2, 5, 5, 5, 5, 6, 6, 7, 7])
#     a = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6])
#     a = np.array([1, 1, 1, 2, 3, 5, 5, 5, 6, 6, 7, 7, 8, 8])
#     a = np.array([1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 8, 8, 8])
#     a = np.array([1, 1, 1, 2, 3, 5, 5, 6, 6, 7, 7, 8, 8, 8])
#     a = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4])
#     a = np.array([3, 4, 4, 5, 5, 5, 5, 6, 6, 6, 7, 7, 8, 8])
#     a = np.array([4, 4, 4, 5, 5, 5, 0, 0, 2, 2, 7, 7, 9, 9])
    a = np.array([2,3,6,9,10,10,11,11,15,20,24,24,29,31])
#     a = np.array([8,8,5,5,5,29,29,29,6,7,8,10,11,12])

#     a = np.array([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 3])
#     a[-1] = i % 10

    if check_hu(a):
        cnt += 1
        print(a)
#         print()
    else:
        print(a)

# assert cnt == total
 
# end = timer()
# print(end - start)
