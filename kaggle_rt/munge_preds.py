#python 2.7
# example use: python munge_preds.py data/ypred.txt
import pandas as pd
import numpy as np
import sys

class CheckDup():   
    def __init__(self, traindf):
        self.n = len(traindf.Phrase)
        self.d = {traindf.Phrase[i]:i for i in range(self.n)}
        self.traindf = traindf
        self.dup_ct = 0
    def contains_id(self, phrase):
        """ 
        Returns the sentiment label from the training data if found, else 
        returns None. I use a dictionary for speed, but this only does 
        exact string matching; I might also want to check containment, 
        similarity to training instances, etc."""
        try:
            idx = self.d[phrase] 
            self.dup_ct += 1
            return self.traindf.ix[idx,"Sentiment"]
        except KeyError:
            return None


def fill_dup(traindf, testdf, resultdf):
    """Fills in the duplicates from the training set"""
    cd = CheckDup(traindf)
    assert testdf.shape[0]==resultdf.shape[0]
    dup_loss = 0 #counts the misclassified duplicated
    for i in xrange(resultdf.shape[0]):
        phrase= testdf.ix[i,'Phrase'] 
        train_lab = cd.contains_id(phrase)
        if train_lab is None:
            continue
        if resultdf.ix[i, 'Sentiment']!= train_lab:
            dup_loss += 1
        resultdf.ix[i,'Sentiment'] = train_lab
    print "number of duplicates found:", cd.dup_ct
    print "percentage of duplicates misclassified (compare to training error)", float(dup_loss)/cd.dup_ct


if __name__=="__main__":


    # read predictions
    try:
        pred_fp = sys.argv[1]
    except:
        print 'first argument should be path to predictions'
        sys.exit(1)

    # read data and check duplicates
    try:
        testdf = pd.read_csv('data/test.tsv', sep="\t")
        traindf = pd.read_csv('data/train.tsv', sep="\t")
    except IOException:
        print "data should be in './data/*'"
        sys.exit(1)
    try:
        preds = pd.read_csv(pred_fp, sep="\n", header=None).ix[:,0]
    except IOException:
        print 'could not open file: ', pred_fp
        sys.exit(1)
    ids = testdf.PhraseId
    resultdf = pd.DataFrame.from_dict({"PhraseId": list(ids), "Sentiment":list(preds)})
    fill_dup(traindf, testdf, resultdf)

    # make submission
    try:
        outputfp=sys.argv[2]
    except:
        outputfp = 'submissions/curr.csv'
    pd.DataFrame.to_csv(resultdf, outputfp, index=False)
    print "wrote to file ", outputfp



