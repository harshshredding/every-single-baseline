#!/usr/bin/env python
import logging as log
import sys
import os

class Tweet(object):
    '''Class for storing tweet'''
    def __init__(self, twid, text=""):
        self.twid = twid
        self.text = text
        self.has_adr = False
        self.anns = []

class Ann(object):
    '''Class for storing annotation spans'''
    def __init__(self, adr, atype, start, end, ptid=""):
        self.adr = adr
        self.atype = atype
        self.start = int(start)
        self.end = int(end)
        self.ptid = ptid

class Meddra(object):
    '''Basic Entity object'''
    def __init__(self, ptid, lltid, text):
        self.ptid = ptid
        self.lltid = lltid
        self.text = text

def get_meddra_dict(meddra_llt):
    """load corpus data and write resolution files"""
    pt_dict, llt_dict = {}, {}
    for line in open(meddra_llt, 'r'):
        elems = line.split("$")
        if len(elems) > 2:
            ptid, lltid, text = elems[2], elems[0], elems[1]
            entry = Meddra(ptid, lltid, text)
            if ptid == lltid:
                pt_dict[ptid] = entry
            llt_dict[lltid] = entry
    return pt_dict, llt_dict

def load_dataset(spfile, tweet_data, load_ids=True, lltfile=None):
    """Loads dataset given the span file and the tweets file
    Arguments:
        spfile {string} -- path to span file
        twfile {string} -- path to tweets file
    Returns:
        dict -- dictionary of tweet-id to Tweet object
    """
    if lltfile:
        pt_dict, llt_dict = get_meddra_dict(lltfile)
    tw_int_map = {}
    # Load tweets
    for twid in tweet_data:
        text = tweet_data[twid]
        tweet = Tweet(twid, text)
        tw_int_map[twid] = tweet
    # Load annotations
    for line in open(spfile, 'r'):
        parts = [x.strip() for x in line.split("\t")]
        if len(parts) != 5 and len(parts) != 6:
            print("Length too long:" + str({len(parts)}))
            continue
        ptid = ""
        if len(parts) == 5:
            twid, start, end, atype, adr = parts
        elif len(parts) == 6:
            twid, start, end, atype, adr, ptid = parts
            # reset ptid in case it is detection only
            if load_ids:
                # if participants added lltid then convert to ptid
                if ptid in llt_dict:
                    ptid = llt_dict[ptid].ptid
                elif ptid in ["-", ""]:
                    pass
                else:
                    log.warning("MedDRA id '"+ptid+"' not found.")
            else:
                ptid = ""
        if twid in tw_int_map:
            tweet = tw_int_map[twid]
        else:
            #log.warning("Tweet id %s not in dataset. Ignoring.", twid)
            continue
        #if atype == "ADR": 
        if atype in ['ENFERMEDAD']: # Spanish task
            ann = Ann(adr.strip(), atype, start, end, ptid)
            tweet.anns.append(ann)
            #tweet.has_adr = (tweet.has_adr or atype == "ADR")
    num_anns = sum([len(x.anns) for _, x in tw_int_map.items()])
    log.info("Loaded dataset %s tweets. %s annotations.", len(tw_int_map), num_anns)
    return tw_int_map

def is_overlap(a, b):
    return b.start <= a.start <= b.end or a.start <= b.start <= a.end

def is_overlap_match(a, b):
    #return is_overlap(a, b) and a.ptid == b.ptid
    return is_overlap(a, b) and a.ptid == b.ptid and a.atype == b.atype # Spanish task

def is_strict_match(a, b):
    #return a.start == b.start and a.end == b.end and a.ptid == b.ptid
    return a.start == b.start and a.end == b.end and a.ptid == b.ptid and a.atype == b.atype # Spanish task

def is_match(a, b, strict):
    return is_strict_match(a, b) if strict else is_overlap_match(a, b)

def perf(gold_ds, pred_ds, strict=True):
    """Calculates performance and returns P, R, F1
    Arguments:
        gold_ds {dict} -- dict contaning gold dataset
        pred_ds {dict} -- dict containing prediction dataset
        strict {boolean} -- boolean indication if strict evaluation is to be used
    """
    g_tp, g_fn = [], []
    # find true positives and false negatives
    for gold_id, gold_tw in gold_ds.items():
        gold_anns = gold_tw.anns
        pred_anns = pred_ds[gold_id].anns
        for g in gold_anns:
            g_found = False
            for p in pred_anns:
                if is_match(p, g, strict):
                    g_tp.append(p)
                    g_found = True
            if not g_found:
                g_fn.append(g)
    p_tp, p_fp = [], []
    # find true positives and false positives
    for pred_id, pred_tw in pred_ds.items():
        pred_anns = pred_tw.anns
        gold_anns = gold_ds[pred_id].anns
        for p in pred_anns:
            p_found = False
            for g in gold_anns:
                if is_match(p, g, strict):
                    p_tp.append(p)
                    p_found = True
            if not p_found:
                p_fp.append(p)
    # both true positive lists should be same
    if len(g_tp) != len(p_tp):
        log.warning("Error: True Positives don't match. %s != %s", g_tp, p_tp)
    log.info("TP:%s FP:%s FN:%s", len(g_tp), len(p_fp), len(g_fn))
    # now calculate p, r, f1
    precision = 1.0 * len(g_tp)/(len(g_tp) + len(p_fp) + 0.000001)
    recall = 1.0 * len(g_tp)/(len(g_tp) + len(g_fn) + 0.000001)
    f1sc = 2.0 * precision * recall / (precision + recall + 0.000001)
    log.info("Precision:%.3f Recall:%.3f F1:%.3f", precision, recall, f1sc)
    return precision, recall, f1sc

def score_task2(pred_file, gold_file, tweet_data, out_file):
    """Score the predictions and print scores to files
    Arguments:
        pred_file {string} -- path to the predictions file
        gold_file {string} -- path to the gold annotation file
        tweet_file {string} -- path to the tweet file
        out_file {string} -- path to the file to write results to
    """
    # load gold dataset
    gold_ds = load_dataset(gold_file, tweet_data, load_ids=False)
    # load prediction dataset
    pred_ds = load_dataset(pred_file, tweet_data, load_ids=False)
    o_prec, o_rec, o_f1 = perf(gold_ds, pred_ds, strict=False)
    out = open(out_file, 'w')
    out.write("Task10overlapF:%.3f\n" % o_f1)
    out.write("Task10overlapP:%.3f\n" % o_prec)
    out.write("Task10overlapR:%.3f\n" % o_rec)
    print(f"Overlap - F1 : {o_f1}, P: {o_prec}, R: {o_rec}")
    s_prec, s_rec, s_f1 = perf(gold_ds, pred_ds, strict=True)
    out.write("Task10strictF:%.3f\n" % s_f1)
    out.write("Task10strictP:%.3f\n" % s_prec)
    out.write("Task10strictR:%.3f\n" % s_rec)
    print(f"Strict - F1 : {s_f1}, P: {s_prec}, R: {s_rec}")
    out.flush()

def score_task3(pred_file, gold_file, tweet_file, out_file, llt_file):
    """Score the predictions and print scores to files
    Arguments:
        pred_file {string} -- path to the predictions file
        gold_file {string} -- path to the gold annotation file
        tweet_file {string} -- path to the tweet file
        out_file {string} -- path to the file to write results to
    """
    # load gold dataset
    gold_ds = load_dataset(gold_file, tweet_file, True, llt_file)
    # load prediction dataset
    pred_ds = load_dataset(pred_file, tweet_file, True, llt_file)
    o_prec, o_rec, o_f1 = perf(gold_ds, pred_ds, strict=False)
    out = open(out_file, 'w')
    out.write("Task5Fr:%.3f\n" % o_f1)
    out.write("Task5Pr:%.3f\n" % o_prec)
    out.write("Task5Rr:%.3f\n" % o_rec)
    s_prec, s_rec, s_f1 = perf(gold_ds, pred_ds, strict=True)
    out.write("Task5Fs:%.3f\n" % s_f1)
    out.write("Task5Ps:%.3f\n" % s_prec)
    out.write("Task5Rs:%.3f\n" % s_rec)
    '''
    out.write("Task3relaxedF:%.3f\n" % o_f1)
    out.write("Task3relaxedP:%.3f\n" % o_prec)
    out.write("Task3relaxedR:%.3f\n" % o_rec)
    s_prec, s_rec, s_f1 = perf(gold_ds, pred_ds, strict=True)
    out.write("Task3strictF:%.3f\n" % s_f1)
    out.write("Task3strictP:%.3f\n" % s_prec)
    out.write("Task3strictR:%.3f\n" % s_rec)
    '''
    out.flush()

def evaluate():
    """Runs the evaluation function"""
    # load logger
    LOG_FILE = '/tmp/ADE_Eval.log'
    ####LOG_FILE = 'C:\\Users\\iflores\\Documents\\Projects\\SMM4H2020\\Datasets_Eval_Scripts\\EvalScripts\\SMM4H20scoringTask3\\ADE_Eval.log'
    ###
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    handlers=[log.StreamHandler(sys.stdout), log.FileHandler(LOG_FILE)])
    # as per the metadata file, input and output directories are the arguments

    ###
    ###if len(sys.argv) != 4:
    ###    log.error("Invalid input parameters. Format:\
    ###              \n python evaluation_util.py [input_dir] [output_dir] [Detection/Resolution]")
    ###    sys.exit(0)


    ###[_, input_dir, output_dir, eval_type] = sys.argv
    [_, input_dir, output_dir ] = sys.argv

    ###
    ###
    ####input_dir = "C:\\Users\\iflores\\Documents\\Projects\\SMM4H2020\\Datasets_Eval_Scripts\\EvalScripts\\SMM4H20scoringTask3"
    ####output_dir = "C:\\Users\\iflores\\Documents\\Projects\\SMM4H2020\\Datasets_Eval_Scripts\\EvalScripts\\SMM4H20scoringTask3"
    eval_type = "Detection"
    ###
    ###


    # get files in prediction zip file
    pred_dir = os.path.join(input_dir, 'res')
    pred_files = [x for x in os.listdir(pred_dir) if not os.path.isdir(os.path.join(pred_dir, x))]
    pred_files = [x for x in pred_files if x[0] not in ["_", "."]]
    if not pred_files:
        log.error("No valid files found in archive. \
                  \nMake sure file names do not start with . or _ characters")
        sys.exit(0)
    if len(pred_files) > 1:
        log.error("More than one valid files found in archive. \
                  \nMake sure only one valid file is available.")
        sys.exit(0)
    # Get path to the prediction file
    pred_file = os.path.join(pred_dir, pred_files[0])
    # Get path to the gold standard annotation file
    tweet_file = os.path.join(input_dir, 'ref/gold_tweets_test.tsv')  # REPLACE by your GoldStandard annotations
    ###llt_file = os.path.join(input_dir, 'ref/llt.asc')
    if eval_type == 'Detection':
        gold_file = os.path.join(input_dir, 'ref/det_gold_test.tsv')
    elif eval_type == 'Resolution':
        gold_file = os.path.join(input_dir, 'ref/res_gold.tsv')
    else:
        log.fatal("Unknown parameter: [{}]. Expecting [Detection, Resolution]".format(eval_type))
        sys.exit(0)

    log.info("Pred file:%s, Gold file:%s", pred_file, gold_file)

    out_file = os.path.join(output_dir, 'scores.txt')
    log.info("Tweet file:%s, Output file:%s", tweet_file, out_file)

    log.info("Start scoring")
    if eval_type == 'Detection':
        score_task2(pred_file, gold_file, tweet_file, out_file)
    else:
        score_task3(pred_file, gold_file, tweet_file, out_file, llt_file)

    log.info("Finished scoring")

if __name__ == '__main__':
    evaluate()
