

import difflib
import numpy as np
import pandas as pd

class tree_stack(object):
    def __init__(self):
        self.stack = list()
    def push(self, item, depth, phrase):
        self.stack.append((item,depth, phrase))
    def pop(self):
        return self.stack.pop(len(self.stack)-1)
# testing tree_stack

def my_split_difference(s1, s2):
    '''s1 contains s2 with addition to beginning or end'''
    split_diff = ""
    for s in difflib.ndiff(s1,s2):
        if s[0]=='-':
            split_diff+=(s[2])
    return split_diff

def generate_tree_data(df):
    ''' Generates tree_df and split_df from the raw Dataframe.'''
    tree_df = df.copy()
    tree_df['Tree_Depth']=0
    tree_df['Parent_PhraseId']=np.nan
    split_df = pd.DataFrame(columns=['PhraseId1', 'PhraseId2', 'SentenceId', 
                                     'Phrase1', 'Phrase2','SplitDifference', 
                                     'Sentiment1', 'Sentiment2'])
    sentence_tree_stack = tree_stack()
    curr_depth=0
    for i in range(1,tree_df.shape[0]):
        curr_row = tree_df.iloc[i-1,:]
        next_row = tree_df.iloc[i,:]
        if curr_row.SentenceId != next_row.SentenceId:
            sentence_tree_stack = tree_stack()
            curr_depth=0
            continue
        if next_row.Phrase in curr_row.Phrase:
            tree_df.ix[i,'Parent_PhraseId'] = curr_row.PhraseId 
            sentence_tree_stack.push(curr_row.PhraseId, curr_depth, curr_row.Phrase)
            curr_depth += 1
            tree_df.ix[i, 'Tree_Depth'] = curr_depth
            split_difference = my_split_difference(curr_row.Phrase, next_row.Phrase)
            split_df.loc[split_df.shape[0]] = [curr_row.PhraseId, next_row.PhraseId, curr_row.SentenceId,
                                               curr_row.Phrase, next_row.Phrase, split_difference,
                                               curr_row.Sentiment, next_row.Sentiment]
            continue
        while(True):
            try:
                parent_phraseid, parent_depth, parent_phrase = sentence_tree_stack.pop()
            except IndexError:
                print i, next_row.Phrase + ' | ' +parent_phrase
                tree_df.ix[i, 'Tree_Depth'] = np.nan
                tree_df.ix[i, 'Parent_PhraseId'] = np.nan
                break;
            if next_row.Phrase in parent_phrase:
                curr_depth = parent_depth + 1
                tree_df.ix[i, 'Tree_Depth'] = curr_depth
                tree_df.ix[i, 'Parent_PhraseId'] = parent_phraseid
                break
    return (tree_df, split_df)


if __name__=="__main__":
    # test tree
    s = tree_stack()
    s.push(3); s.push(4); s.push(5)
    print s.pop(), s.pop(), s.pop(), s.pop()

    #test my_split_difference
    print my_split_difference('An intermittently pleasing but mostly', 'mostly')



