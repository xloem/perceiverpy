import torch 


# given two ranges of indices, a generator that produces sufficient set combinations to isolate the influence
# of any single index.
# i think some of the copying internally could be optimized away; my memory was taxed during dev
def batch_pair_combinations(batch1idx, batch2idx, batchlenlog2, shuffle = True):
    # for now we only consider combinations of 2 groups of data.  the algorithm is extendable for larger groups of data.
    if int(batchlenlog2) != batchlenlog2:
        raise AssertionError('batchlen not power of 2')
    batchlen = 1 << int(batchlenlog2)
    idxs1 = torch.arange(batchlen) + batch1idx
    idxs2 = idxs1 + (batch2idx - batch1idx)
    orig = torch.cat((idxs1, idxs2))
    if shuffle:
        orig = orig[torch.randperm(len(orig))]
    comb = orig.clone()

    checkersize = batchlen
    orthosize = 1
    while True:
        if shuffle:
            yield comb[:batchlen][torch.randperm(batchlen)]
            yield comb[batchlen:][torch.randperm(batchlen)]
        else:
            yield comb[:batchlen]
            yield comb[batchlen:]

        if checkersize <= 1:
            break

        nextchecker = checkersize >> 1
        nextortho = orthosize << 1
        comb_view = comb.view(nextortho, checkersize)
        orig_view = orig.view(nextortho, checkersize)
        comb_view[:, :nextchecker] = orig_view[:, :nextchecker]
        comb_view[orthosize:, nextchecker:] = orig_view[:orthosize, nextchecker:]
        comb_view[:orthosize, nextchecker:] = orig_view[orthosize:, nextchecker:]
        checkersize = nextchecker
        orthosize = nextortho

if __name__ == '__main__':
    for comb in batch_pair_combinations(0, 4, 2):
        print(comb)
