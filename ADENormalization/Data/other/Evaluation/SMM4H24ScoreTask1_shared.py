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
    def __init__(self, twid, ade, ptid=""):
        self.twid = twid
        self.ades = [ade]
        self.ptids = [ptid]
    
    def count_ptid(self):
        return len(self.ptids)

    def update(self, ade, ptid):
        self.ades.append(ade)
        self.ptids.append(ptid)
    
    def valid_ptid(self,pt_dict, llt_dict):
        # if participants added lltid, then convert to ptid
        ptids = []
        for ptid_single in self.ptids:
            
            if ptid_single in llt_dict:
                # log.info("Info: MedDRA id '"+ptid_single+"' is lower lever term id.")
                ptid = llt_dict[ptid_single].ptid
                ptids.append(ptid)
            
            elif ptid_single in pt_dict:
                ptids.append(ptid_single)
            else:
                log.warning("MedDRA id '"+ptid_single+"' not found.")
        self.ptids = list(set(ptids))


    def count_ptid_unseeen(self,seen_concepts):
        return len([ptid_single for ptid_single in self.ptids if ptid_single not in seen_concepts])
        
        
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
        if len(parts) != 3:
            log.error("Invalid prediction format:\
                 \n Your submission should follow the format of $tweet_id$\t$ADE$\t$MedDRA_ID$ ")
            sys.exit(0)
            log.warning("Missing ADE span or MedDRA ID:" +" ".join(parts))
            continue
        if len(parts) == 3:
            twid, ade,  ptid = parts
            if twid == "tweet_id":
                continue
            if twid not in tw_int_map:
                log.ERROR("Error: Tweet id %s not in dataset. Ignoring.", twid)
                continue
            else:
                if twid not in tw_ann:
                    ann = Ann(twid, ade, ptid)
                    tw_ann[twid] = ann
                else:
                    tw_ann[twid].update(ade, ptid)

                 
    for twid, ann in tw_ann.items():
        ptids_count +=ann.count_ptid()
        # map lltids to ptids if it has llt
        if load_ids:
            tw_ann[twid].valid_ptid(pt_dict, llt_dict)
        
                
    log.info("Loaded dataset %s tweets. %s annotations. %s ptids.", len(tw_int_map), len(tw_ann), ptids_count)
    return tw_ann

def perf(gold_anns, pred_anns, seen_concepts):
    """Calculates performance and returns P, R, F1
    Arguments:
        gold_anns {dict} -- dict contaning gold dataset
        pred_anns {dict} -- dict containing prediction dataset
        seen_concepts {list} --None or a list of seen concepts 
    """
    
    g_norm_tp, g_norm_all = 0, 0
    g_ade_tp, g_ade_all = 0, 0

    # gold annotations: find true positives and the overall counts
    for twid, gold_ann in gold_anns.items():
                
        if seen_concepts:
            g_norm_all += gold_ann.count_ptid_unseeen(seen_concepts)
        else:
            g_norm_all += gold_ann.count_ptid()
            g_ade_all += len(gold_ann.ades)

        if twid in pred_anns:
            pred_ann = pred_anns[twid]
            if seen_concepts:
                g_norm_tp += ann_match_unseen(gold_ann.ptids, pred_ann.ptids,seen_concepts)
            else:
                g_norm_tp += ann_match(gold_ann.ptids, pred_ann.ptids)
                g_ade_tp += ann_match(gold_ann.ades, pred_ann.ades)
    
    
    # predictions: find true positives and the overall counts
    p_norm_tp, p_norm_all = 0, 0
    p_ade_tp, p_ade_all = 0, 0

    for twid, pred_ann in pred_anns.items():
        if seen_concepts:
            p_norm_all += pred_ann.count_ptid_unseeen(seen_concepts)
        else:
            p_norm_all += pred_ann.count_ptid()
            p_ade_all += len(pred_ann.ades)

        if twid in gold_anns:
            gold_ann = gold_anns[twid]
            if seen_concepts:
                p_norm_tp += ann_match_unseen(gold_ann.ptids, pred_ann.ptids,seen_concepts)
            else:
                p_norm_tp += ann_match(gold_ann.ptids, pred_ann.ptids)
                p_ade_tp += ann_match(gold_ann.ades, pred_ann.ades)
    # both true positive lists should be same
                
    if g_norm_tp != p_norm_tp:
        log.ERROR("Error: True Positives of ADE normaliaztion don't match. %s != %s (gold vs. prediction)", g_norm_tp, p_norm_tp)
    
    if g_ade_tp != p_ade_tp:
        log.ERROR("Error: True Positives of ADE extraction don't match. %s != %s (gold vs. prediction)", g_ade_tp, p_ade_tp)

    if seen_concepts:
        log.info("Normalization Unseen TP:%s FP:%s FN:%s", g_norm_tp, p_norm_all - g_norm_tp, g_norm_all - g_norm_tp)
    else:
        log.info("Normalization Overall TP:%s FP:%s FN:%s", g_norm_tp, p_norm_all - p_norm_tp, g_norm_all - g_norm_tp)
        log.info("Extraction TP:%s FP:%s FN:%s", g_ade_tp, p_ade_all - g_ade_tp, g_ade_all - g_ade_tp)

    # now calculate p, r, f1
    precision_norm = 1.0 * g_norm_tp /( p_norm_all + 0.000001)
    recall_norm = 1.0 * g_norm_tp /( g_norm_all + 0.000001)
    f1_norm = 2.0 * precision_norm * recall_norm / (precision_norm + recall_norm + 0.000001)


    precision_ade = 1.0 * g_ade_tp /( p_ade_all + 0.000001)
    recall_ade = 1.0 * g_ade_tp /( g_ade_all + 0.000001)
    f1_ade = 2.0 * precision_ade * recall_ade / (precision_ade + recall_ade + 0.000001)


    if seen_concepts:
        log.info("ADE Normalization Unseen: Precision:%.3f Recall:%.3f F1:%.3f", precision_norm, recall_norm, f1_norm)
        return precision_norm, recall_norm, f1_norm, precision_ade, recall_ade, f1_ade
    else:
        log.info("ADE Normalization Overall: Precision:%.3f Recall:%.3f F1:%.3f", precision_norm, recall_norm, f1_norm)
        log.info("ADE Extraction Overall: Precision:%.3f Recall:%.3f F1:%.3f", precision_ade, recall_ade, f1_ade)
        return precision_norm, recall_norm, f1_norm, precision_ade, recall_ade, f1_ade

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
    log.info("Load gold annotations: ")
    gold_anns = load_dataset(gold_file, tweet_file, True, llt_file)
    # load prediction dataset
    log.info("Load predictions: ")
    pred_anns = load_dataset(pred_file, tweet_file, True, llt_file)
    # load seen concepts
    seen_concepts = readfrom_txt(seencfile) 
    precision_norm, recall_norm, f1_norm, precision_ade, recall_ade, f1_ade  = perf(gold_anns, pred_anns, seen_concepts=None)
    
    out = open(out_file, 'w')

    out.write("Task1: ADE Extraction F1:%.3f\n" % f1_ade)
    out.write("Task1: ADE Extraction Precision:%.3f\n" % precision_ade)
    out.write("Task1: ADE Extraction Recall:%.3f\n" % recall_ade)

    out.write("Task1: ADE Normalization F1:%.3f\n" % f1_norm)
    out.write("Task1: ADE Normalization Precision:%.3f\n" % precision_norm)
    out.write("Task1: ADE Normalization Recall:%.3f\n" % recall_norm)

    precision_norm_unseen, recall_norm_unseen, f1_norm_unseen, _, _, _= perf(gold_anns, pred_anns, seen_concepts=seen_concepts)
    out.write("Task1: ADE Normalization on Unseen MedDRA IDs F1:%.3f\n" % f1_norm_unseen)
    out.write("Task1: ADE Normalization on Unseen MedDRA IDs Precision::%.3f\n" % precision_norm_unseen)
    out.write("Task1: ADE Normalization on Unseen MedDRA IDs Recall::%.3f\n" % recall_norm_unseen)
    out.flush()
    
def evaluate():
    """Runs the evaluation function"""
    # load logger
    LOG_FILE = './ADE_Eval.log'
    log.basicConfig(level=log.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    handlers=[log.StreamHandler(sys.stdout), log.FileHandler(LOG_FILE)])
    log.info("-------------------------------------------")
    # as per the metadata file, input and output directories are the arguments
    if len(sys.argv) != 4:
       log.error("Invalid input parameters. Format:\
                 \n python evaluation.py [prediction_path] [gold_dir] [output_dir]")
       sys.exit(0)

    [_, prediction_path, gold_dir, output_dir] = sys.argv


    
    # Get path to the gold standard annotation file
    tweet_file = os.path.join(gold_dir, 'tweets.tsv')
    llt_file = os.path.join(gold_dir, 'llt.asc')
    seenc_file = os.path.join(gold_dir, 'seen_concepts.txt')
    gold_file = os.path.join(gold_dir, 'gold_annotations_for_evaluation.tsv')

    log.info("Pred file:%s, Gold file:%s, Seen concepts file:%s ", prediction_path, gold_file, seenc_file)
    out_file = os.path.join(output_dir, 'scores.txt')
    log.info("Tweet file:%s, Output file:%s", tweet_file, out_file)
    score_task(prediction_path, gold_file, tweet_file, out_file, llt_file, seenc_file)
    log.info("Finished scoring")

if __name__ == '__main__':
    evaluate()
