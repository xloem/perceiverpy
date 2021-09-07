import os
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
            yield comb[:batchlen][torch.cat((torch.randperm(batchlen), torch.randperm(batchlen) + batchlen))]
            yield comb[batchlen:][torch.cat((torch.randperm(batchlen), torch.randperm(batchlen) + batchlen))]
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

def name2path(*names):
    return os.path.join('.', 'models', *names[:-1], names[-1] + '.pystate')

def path2name(path):
    return os.path.splitext(os.path.basename(path))[0]

def stack2name(maxdepth=2):
    import inspect
    result = []
    depth = 0
    lastfile = None
    for frame in inspect.stack()[1:]:
        name = path2name(frame.filename)
        if '<' not in name:
            if name != lastfile:
                lastfile = name
                depth += 1
                if depth > maxdepth:
                    break
            #name += '.' + str(frame.lineno)
            if '<' not in frame.function:
                name += '.' + frame.function
            result.append(name)
    result.reverse()
    return '-'.join(result)

def dict2name(dict):
    def shorten(val):
        if type(val) is str:
            result = ''
            lastchr = ''
            for chr in val:
                if lastchr == '' or lastchr == '_':
                    result += chr
                lastchr = chr
            return result
        elif type(val) is bool:
            return 'fT'[val]
        else:
            return str(val)
    return ''.join((
        shorten(key) + shorten(value)
        for key, value in dict.items()
    ))

if __name__ == '__main__':
    for comb in batch_pair_combinations(0, 4, 2):
        print(comb)
