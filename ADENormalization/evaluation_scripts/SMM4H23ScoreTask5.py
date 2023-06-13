#!/usr/bin/env python
import logging as log
import sys
import os

class Tweet(object):
    '''Class for storing tweet'''
    def __init__(self, twid, text=""):
        self.twid = twid
        self.text = text
        
class Meddra(object):
    '''Basic Meddra Entity object'''
    def __init__(self, ptid, lltid, text):
        self.ptid = ptid
        self.lltid = lltid
        self.text = text
        
class Ann(object):
    '''Class for storing annotation spans'''
    def __init__(self, tid, ptid=""):
        self.tid = tid
        self.ptid = ptid.split(" & ")
        
    def count_ptid(self):
        return len(self.ptid)
    
    def valid_ptid(self,pt_dict, llt_dict):
        # if participants added lltid, then convert to ptid
        ptids = []
        for ptid_single in self.ptid:
            
            if ptid_single in llt_dict:
                log.warning("MedDRA id '"+ptid_single+"' is lower lever term id.")
                ptid = llt_dict[ptid_single].ptid
                ptids.append(ptid)
            
            elif ptid_single in pt_dict:
                ptids.append(ptid_single)
            else:
                log.warning("MedDRA id '"+ptid_single+"' not found.")
        self.ptid = list(set(ptids))
        # self.ptid = ptids   commented for unique ptid
        
    def count_ptid_unseeen(self,seen_concepts):
        return len([ptid_single for ptid_single in self.ptid if ptid_single not in seen_concepts])
        
        
def ann_match(gold, pred):
    #   return len([item for item in gold if item in pred])         commented for unique ptid matching
    return len(set(gold).intersection(set(pred)))

def ann_match_unseen(gold, pred, seen_concepts):
    #ptid_match =  [item for item in gold if item in pred]         commented for unique ptid matching
    ptid_match = list(set(gold).intersection(set(pred)))
    return len([ptid_single for ptid_single in ptid_match if ptid_single not in seen_concepts])
    

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

def readfrom_txt(path):
    data = open(path).read()
    txt_list = list()
    for line in data.splitlines():
        txt_list.append(line)
    return txt_list


def load_dataset(annfile, twfile, load_ids=True, lltfile=None):
    """Loads dataset given the span file and the tweets file
    Arguments:
        twfile {string} -- path to raw twitter file
    Arguments:
        annfile {string} -- path to annotation file
    Arguments:
        load_ids {boolean} -- whether to load meddra id file
    Arguments:
        lltfile {string} -- path to meddra id file

    Returns:
        dict -- dictionary of tweet-id to Tweet object
    """
    # Load meddra file
    if load_ids and lltfile:
        pt_dict, llt_dict = get_meddra_dict(lltfile)
        
    tw_int_map = {}
    # Load tweets
    if twfile:
        for line in open(twfile, 'r'):
            parts = line.split("\t")
            twid, text = parts[0], parts[1]
            if twid == "tweet_id":
                continue
            tweet = Tweet(twid, text)
            if twid in tw_int_map:
                log.warning("Possible duplicate %s from raw twitter file", twid)
            tw_int_map[twid] = tweet

    uniq = set()
    ptids_count = 0
    tw_ann = {}
    # Load annotations
    for line in open(annfile, 'r'):
        # duplicate lines in annotation
        if line.strip() in uniq:
            log.warning("Possible duplicate %s from annotation file", line)
            continue
        uniq.add(line.strip())
        parts = [x.strip() for x in line.split("\t")]
        if len(parts) != 2:
            log.warning("Length too long:" + str(len(parts)))
            continue
        if len(parts) == 2:
            twid,  ptid = parts
            if twid == "tweet_id":
                continue
            if twid not in tw_int_map:
                log.warning("Tweet id %s not in dataset. Ignoring.", twid)
                continue
            else:
                ann = Ann(twid, ptid)
                # map lltids to ptids
                if load_ids:
                    ann.valid_ptid(pt_dict, llt_dict) 
                tw_ann[twid] = ann
                ptids_count +=ann.count_ptid()
                
    log.info("Loaded dataset %s tweets. %s annotations. %s ptids.", len(tw_int_map), len(tw_ann), ptids_count)
    return tw_ann

def perf(gold_anns, pred_anns, seen_concepts):
    """Calculates performance and returns P, R, F1
    Arguments:
        gold_anns {dict} -- dict contaning gold dataset
        pred_anns {dict} -- dict containing prediction dataset
        seen_concepts {list} --None or a list of seen concepts 
    """
    
    g_tp, g_all = 0, 0
    # gold annotations: find true positives and the overall counts
    for twid, gold_ann in gold_anns.items():
                
        if seen_concepts:
            g_all += gold_ann.count_ptid_unseeen(seen_concepts)
        else:
            g_all += gold_ann.count_ptid()
        if twid in pred_anns:
            pred_ann = pred_anns[twid]
            if seen_concepts:
                g_tp += ann_match_unseen(gold_ann.ptid, pred_ann.ptid,seen_concepts)
            else:
                g_tp += ann_match(gold_ann.ptid, pred_ann.ptid)
    # predictions: find true positives and the overall counts
    p_tp, p_all = 0, 0
    for twid, pred_ann in pred_anns.items():
        if seen_concepts:
            p_all += pred_ann.count_ptid_unseeen(seen_concepts)
        else:
            p_all += pred_ann.count_ptid()
        if twid in gold_anns:
            gold_ann = gold_anns[twid]
            if seen_concepts:
                p_tp += ann_match_unseen(gold_ann.ptid, pred_ann.ptid,seen_concepts)
            else:
                p_tp += ann_match(gold_ann.ptid, pred_ann.ptid)
    # both true positive lists should be same
    if g_tp != p_tp:
        log.warning("Error: True Positives don't match. %s != %s", g_tp, p_tp)
    log.info("TP:%s FP:%s FN:%s", g_tp, p_all - g_tp, g_all - g_tp)
    # now calculate p, r, f1
    precision = 1.0 * g_tp /( p_all + 0.000001)
    recall = 1.0 * g_tp /( g_all + 0.000001)
    f1sc = 2.0 * precision * recall / (precision + recall + 0.000001)
    if seen_concepts:
        log.info("Unseen: Precision:%.3f Recall:%.3f F1:%.3f", precision, recall, f1sc)
    else:
        log.info("Overall: Precision:%.3f Recall:%.3f F1:%.3f", precision, recall, f1sc)
    return precision, recall, f1sc

def score_task(pred_file, gold_file, tweet_file, out_file, llt_file, seencfile):
    """Score the predictions and print scores to files
    Arguments:
        pred_file {string} -- path to the predictions file
        gold_file {string} -- path to the gold annotation file
        tweet_file {string} -- path to the tweet file
        out_file {string} -- path to the file to write results
        llt_file {string} -- path to the meddra file
        seencfile {string} -- path to the seen concepts file
        
    """
    # load gold dataset
    gold_anns = load_dataset(gold_file, tweet_file, True, llt_file)
    # load prediction dataset
    pred_anns = load_dataset(pred_file, tweet_file, True, llt_file)
    # load seen concepts
    seen_concepts = readfrom_txt(seencfile) 
    o_prec, o_rec, o_f1 = perf(gold_anns, pred_anns, seen_concepts=None)
    out = open(out_file, 'w')
    out.write("Task5OverallF:%.3f\n" % o_f1)
    out.write("Task5OverallP:%.3f\n" % o_prec)
    out.write("Task5OverallR:%.3f\n" % o_rec)
    s_prec, s_rec, s_f1 = perf(gold_anns, pred_anns, seen_concepts=seen_concepts)
    out.write("Task5UnseenF:%.3f\n" % s_f1)
    out.write("Task5UnseenR:%.3f\n" % s_prec)
    out.write("Task5UnseenR:%.3f\n" % s_rec)
    out.flush()
    
def evaluate():
    """Runs the evaluation function"""
    # load logger
    LOG_FILE = '/tmp/ADE_Eval.log'
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    handlers=[log.StreamHandler(sys.stdout), log.FileHandler(LOG_FILE)])
    log.info("-------------------------------------------")
    # as per the metadata file, input and output directories are the arguments
    if len(sys.argv) != 4:
       log.error("Invalid input parameters. Format:\
                 \n python evaluation.py [input_dir] [output_dir] [Normalization]")
       sys.exit(0)

    [_, input_dir, output_dir, eval_type] = sys.argv

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
    tweet_file = os.path.join(input_dir, 'ref/tweets.tsv')
    llt_file = os.path.join(input_dir, 'ref/llt.asc')
    seenc_file = os.path.join(input_dir, 'ref/seen_concepts.txt')
    if eval_type == 'Normalization':
        gold_file = os.path.join(input_dir, 'ref/norms.tsv')
    else:
        log.fatal("Unknown parameter: [{}]. Expecting [Normalization]".format(eval_type))
        sys.exit(0)
    log.info("Pred file:%s, Gold file:%s, Seen concepts file:%s ", pred_file, gold_file, seenc_file)
    out_file = os.path.join(output_dir, 'scores.txt')
    log.info("Tweet file:%s, Output file:%s", tweet_file, out_file)
    score_task(pred_file, gold_file, tweet_file, out_file, llt_file, seenc_file)
    log.info("Finished scoring")

if __name__ == '__main__':
    evaluate()
