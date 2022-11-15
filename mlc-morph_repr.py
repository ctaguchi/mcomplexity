import sys
import os.path
from collections import Counter
import argparse
import numpy as np
from multiprocessing import Pool
import zlib
import random
import re
import glob
import time

from conllu import conllu_sentences

# filename = sys.argv[1]

def read_treebank(tbdir, ja=False):
    sentences = []
    if ja:
        # files = ["pud_tested_suw_1.conllu",
        #     "pud_tested_suw_2.conllu",
        #     "pud_tested_luw_1.conllu",
        #     "pud_tested_luw_2.conllu"]
        # for tbf in tbdir:
        sentences.extend(conllu_sentences(tbdir))
    else:
        for tbf in glob.glob(tbdir + '/*.conllu'):
            sentences.extend(conllu_sentences(tbf))
    return sentences

def sample_nodes(sentences, sample_size=1000, random_sample=True,
        filter_num=True, filter_pos={'X', 'PUNCT'}):
    """ Filter/sample sentences from given treebank sentences.

    Arguments:
    sample_size:    The size of the samples in number of nodes. If None (or
                    anything that evaluates to False, the whole
                    corpus is used.
    random_sample:  Sample formed by chosing sentences randomly
                    with replacement.  The order within the sentences
                    are preserved. If False, the order is not reandomized.
    filter_pos:     Set of POS tags to skip while creating the
                    node list.
    filter_num:     Skip the numbers (if written as arabic numerals).

    """
    nodes = []
    i = -1
    while not sample_size or len(nodes) < sample_size:
        if random_sample:
            i = random.randrange(len(sentences))
        else:
            i = (i + 1) % len(sentences)
        for n in sentences[i].nodes[1:]:
            if filter_pos and n.upos in filter_pos:
                continue
            elif filter_num and n.upos == 'NUM' and not n.form.isalpha():
                continue
            elif n.form is None or n.lemma is None: # error in some treebanks
                continue
            nodes.append(n)
        if not sample_size and i == len(sentences):
            break
    if sample_size:
        nodes = nodes[:sample_size]
    return nodes

def get_ttr(sentences, sample_size=1000, random_sample=True,
        lowercase=True, filter_num=True, filter_pos={'X', 'PUNCT'},
        **kwargs):
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    if lowercase:
        words = [n.form.lower() for n in nodes]
    return len(set(words)) / len(words)

def get_msp(sentences, sample_size=1000, random_sample=True,
        lowercase=True, filter_num=True, filter_pos={'X', 'PUNCT'},
        **kwargs):
    """ Calculate the 'mean size of paradigm' on a sample of the given treebank.

    Arguments:
    lowercase:      Convert the words to lowercase.

    Other arguments are as defined in sample_nodes().

    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    if lowercase:
        nlemmas = len(set((x.lemma.lower() for x in nodes)))
        nwords = len(set((x.form.lower() for x in nodes)))
    else:
        nlemmas = len(set((x.lemma for x in nodes)))
        nwords = len(set((x.form for x in nodes)))
    return (nwords / nlemmas)

def get_wh_lh(sentences, sample_size=1000, random_sample=True,
        lowercase=True, filter_num=True, filter_pos={'X', 'PUNCT'},
        smooth=None, **kwargs):
    """ Calculate the unigram words and lemma entropy.

    Arguments:
    lowercase:  Convert the words to lowercase.
    smooth:     Apply smoothing. A numeric value indicates 'add alpha'
                smoothing, 'GT' means absolute discouting based
                on Good-Tring. [these are currently not
                (re)implemented here as they are not used in the
                paper.]

    Other arguments are as defined in sample_nodes().

    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    if lowercase:
        clemmas = Counter((x.lemma.lower() for x in nodes))
        cwords = Counter((x.form.lower() for x in nodes))
    else:
        clemmas = Counter((x.lemma for x in nodes))
        cwords = Counter((x.form for x in nodes))
    nlemmas = sum(clemmas.values())
    nwords = sum(cwords.values())
    wh, lh = 0, 0 # init
    for w in cwords:
        p = cwords[w] / nwords
        wh -= p * np.log2(p)
    for l in clemmas:
        p = clemmas[l] / nlemmas
        lh -= p * np.log2(p)
    return wh, lh

def get_mfh(sentences, sample_size=1000, random_sample=True,
        filter_num=True, filter_pos={'X', 'PUNCT'},
        smooth=None, **kwargs):
    """ Calculate the morphological feature (and POS) entropy.
    POS en

    Arguments:
    smooth:     Apply smoothing. A numeric value indicates 'add alpha'
                smoothing, 'GT' means absolute discouting based
                on Good-Tring. [these are currently not
                (re)implemented here as they are not used in the
                paper.]

    Other arguments are as defined in sample_nodes().
    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    cfeat = Counter()
    cpos = Counter()
    npos, nfeat = 0, 0
    for node in nodes:
        if node.feats:
            feats = node.feats.split('|')
            cfeat.update(feats)
        cpos.update([node.upos])
    npos = sum(cpos.values())
    nfeat = sum(cfeat.values())
    ph, mfh = 0, 0
    for pos in cpos:
        p = cpos[pos] / npos
        ph -= p * np.log2(p)
    for feat in cfeat:
        p = cfeat[feat] / nfeat
        mfh -= p * np.log2(p)
    return mfh, ph

def random_words(words, uniform=False):
    alphabet = {str(i):i for i in range(10)}
    if uniform:
        chcount = Counter(set((ch for w in words for ch in w)))
    else:
        chcount = Counter((ch for w in words for ch in w ))
    if len(chcount) > 256:
        # Non-alphabetic scripts are not comparable,
        # we calculate a value for the sake of robustness
        print("Warning: more than 255 characters", file=sys.stderr)
        chcount = Counter(dict(chcount.most_common(255)))
    n = sum(chcount.values())
    p = [chcount[i]/n for i in chcount]
    worddict = set(words)
    rdict = {w:''.join(np.random.choice(list(chcount), size=len(w),
                    replace=True, p=p)) for w in worddict}
    return [rdict[w] for w in words]

def get_ws(sentences, sample_size=1000, random_sample=True,
        lowercase=True, filter_num=True, filter_pos={'X', 'PUNCT'},
        **kwargs):
    """Calculate the information loss when word-internal structure is destroyed.

    Arguments:
    lowercase:  Convert the words to lowercase.

    Other arguments are as defined in sample_nodes().

    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample,
                 filter_num=filter_num, filter_pos=filter_pos)
    if lowercase:
        words = [n.form.lower() for n in nodes]
    else:
        words = [n.form for n in nodes]
    rwords = random_words(words)

    alphabet = {' ': 0}
    for w in rwords:
        for ch in w:
            if ch not in alphabet:
                alphabet[ch] = len(alphabet) 

    text = ' '.join(words)      # original text
    rtext = ' '.join(rwords)    # randomized `cooked' text
    # We binarize them to remove the effects of Unicode encoding
    bintext =  bytearray([alphabet.get(ch, 0) for ch in text])
    comptext = zlib.compress(bintext, level=9)
    cr = len(bintext)/len(comptext)
    rbintext =  bytearray([alphabet.get(ch, 1) for ch in rtext])
    rcomptext = zlib.compress(rbintext, level=9)
    rcr = len(rbintext)/len(rcomptext)
    return cr - rcr

def get_is(sentences, sample_size=1000, random_sample=True, 
        **kwargs): #ignore unsused arguments
    """Calculate maximum number of inflectional markers per verb.

    All arguments passed to sample_nodes().

    """
    nodes = sample_nodes(sentences, sample_size=sample_size,
                 random_sample=random_sample)
    fset = set()    # set of features
    fvset = set()   # set of feature-value pairs
    featcount = []  # number of features marked on each verb
    for node in nodes:
        if node.upos == 'VERB' and node.feats:
            fvlist = node.feats.split('|')
            fvset.update(fvlist)
            feats = (fv.split('=')[0] for fv in fvlist)
            fset.update(feats)
            featcount.append(len(fvlist))
    avg = 0
    if featcount:
        avg = sum(featcount)/len(featcount)
#    return len(fset), len(fvset), avg
    return len(fset)

def get_wh(*args, **kwargs):
    return get_wh_lh(*args, **kwargs)[0]

def get_lh(*args, **kwargs):
    return get_wh_lh(*args, **kwargs)[1]

measures = {
    'ttr':  ('Type/token ratio', get_ttr), 
    'msp':  ('Means size of paradigm', get_msp), 
    'ws':   ('Word structure information', get_ws), 
    'wh':   ('Word entropy (unigram)', get_wh), 
    'lh':   ('Lemma entropy', get_lh), 
    'is':   ('Inflectional synthesis', get_is), 
    'mfh':  ('Morphological feature entropy', get_mfh),
}

def get_score(jobdesc):
    func = measures[jobdesc[0]][1] # mc measure functions defined above
    tb = read_treebank(jobdesc[1])
    kwargs = jobdesc[2]
    print("Calculating scores for {}, {}...".format(jobdesc[1], func))
    return jobdesc, func(tb, **kwargs)

def get_score_ja(jobdesc):
    func = measures[jobdesc[0]][1] # mc measure functions defined above
    tb = read_treebank(jobdesc[1], ja=True)
    kwargs = jobdesc[2]
    print("Calculating scores for {}, {}...".format(jobdesc[1], func))
    return jobdesc, func(tb, **kwargs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('treebanks', nargs='+')
    ap.add_argument('-j', '--nproc', default=1, type=int,
                        help='number of processes')
    ap.add_argument('-s', '--samples', default=10, type=int,
                        help='number of samples')
    ap.add_argument('-S', '--sample-size', default=1000, type=int)
    ap.add_argument('--separator', default='\t')
    ap.add_argument('-n', '--normalize', action='store_true')
    ap.add_argument('-m', '--measures', default='all',
                        help='comma separated measures, or all')
    ap.add_argument('-o', '--output', default='measures.txt')
    ap.add_argument('-ja', '--japanese', default=False, type=bool)
    args = ap.parse_args()

    if args.measures == 'all':
        mlist = tuple(measures.keys())
    else:
        mlist = args.measures.split(',')

    kwargs = {'sample_size': args.sample_size}
    joblist = []
    if args.japanese == False:
        for m in mlist:
            for tbdir in glob.glob(args.treebanks[0] + "/UD_*/"):
                for _ in range(args.samples):
                    # joblist.append((m, tbdir, kwargs))
                    joblist.append((m, tbdir, kwargs))
    else:
        files = ["../ud_ja_standardize/pud_tested_suw_1_romanized.conllu",
            "../ud_ja_standardize/pud_tested_suw_2_romanized.conllu",
            "../ud_ja_standardize/pud_tested_luw_1_romanized.conllu",
            "../ud_ja_standardize/pud_tested_luw_2_romanized.conllu"]
        for m in mlist:
            for tbdir in files:
                for _ in range(args.samples):
                    joblist.append((m, tbdir, kwargs))

    # print(joblist)
    start = time.time()

    pool = Pool(processes=args.nproc) # for parallel processing (faster?)
    if args.japanese == False:
        res = pool.map(get_score, joblist) # the second argument is iterable
        print(res[1])
    else:
        res = pool.map(get_score_ja, joblist)
        print(res[1])

    scores = dict()
    for (m, tb, _), sc in res:
        tb = os.path.basename(tb.rstrip('/')).replace('UD_','')
        if (m, tb) not in scores:
            scores[(m, tb)] = []
        scores[(m, tb)].append(sc)
    print(scores)
    print(time.time() - start)
    
    tblist = [os.path.basename(tb.rstrip('/')).replace('UD_','') \
            for tb in glob.glob(args.treebanks[0] + "/UD_*/")]
    ja_tb = ["pud_tested_suw_1_romanized.conllu",
            "pud_tested_suw_2_romanized.conllu",
            "pud_tested_luw_1_romanized.conllu",
            "pud_tested_luw_2_romanized.conllu"]
            
    fmt = "\t{}" * (2*len(mlist))
    head = [x for pair in zip(mlist, [m + "_sd" for m in mlist]) for x in pair]
    with open(args.output, "wt") as fp:
        print("treebank", fmt.format(*head), file=fp)
        if args.japanese == False:
            for tb in tblist:
                print(tb, end="", file=fp)
                sclist = []
                for m in mlist:
                    sc = np.array(scores[(m, tb)])
                    sclist.extend((sc.mean(), sc.std()))
                print(fmt.format(*sclist), file=fp)
        else:
            for tb in ja_tb:
                print(tb, end="", file=fp)
                sclist = []
                for m in mlist:
                    sc = np.array(scores[((m, tb))])
                    sclist.extend((sc.mean(), sc.std()))
                print(fmt.format(*sclist), file=fp)

    # for job in joblist:
    #     sentences = read_treebank(job)

        # print("Type-Token Ratio (TTR)               : {}".format(get_ttr(sentences)))
        # print("Mean Size of Paradigm (MSP)          : {}".format(get_msp(sentences)))
        # wh, lh = get_wh_lh(sentences)
        # print("Word Entropy (WH)                    : {}".format(wh))
        # print("Lemma Entropy (LH)                   : {}".format(lh))
        # print("Morphological Feature Entropy (MFH)  : {}".format(get_mfh(sentences)[0]))
        # print("Information in Word Structure (WS)   : {}".format(get_ws(sentences)))
        # print("Inflectional Synthesis (IS)          : {}".format(get_is(sentences)))

    # elapsed = time.time() - start
    # print(elapsed) 