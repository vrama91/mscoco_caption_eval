#!/usr/bin/env python
# Ramakrishna Vedantam <vrama91@vt.edu>
#
# Date: Wed Jul 22 17:58:14 EDT 2015
#
# Description: Class to compute skip thought vectors given
# candidates and references

import skthoughts
import numpy as np
import pdb

class skipScorer():
    def __init__(self, test=None, refs=None):
        ''' singular instance '''
        self.crefs = []
        self.ctest = []
        self.ref_len = None

    def __iadd__(self, other):
        '''add an instance (e.g., from another sentence).'''

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def cook_append(self, test, refs):
        '''called by constructor and __iadd__ to avoid creating new instances.'''

        if refs is not None:
            self.crefs.append(refs)
            if test is not None:
                self.ctest.append(test) ## N.B.: -1
            else:
                self.ctest.append(None) # lens of crefs and ctest have to match

    def compute_skip(self):
        def sublists(original, lenRefs):
            newList = []

            lenRefs = np.cumsum(np.array([0] + lenRefs))

            for it in xrange(len(lenRefs)-1):
                newList.append(original[lenRefs[it]:lenRefs[it+1]])
            return newList

        import skipthoughts
        # TO-DO: Add cpu/gpu as a parameter to skip thoughts

        model = skipthoughts.load_model()
        print "Getting skip vectors for candidates"
        candVecs = skipthoughts.encode(model, self.ctest)
        print "Done"

        print "Getting skip vectors for references"
        lenRefs = [len(x) for x in self.crefs]
        # assert that all refs should have number of sentences
        assert(len(set(lenRefs))==1)
        allRefs = [y for x in self.crefs for y in x]
        refVecs = skipthoughts.encode(model, allRefs)
        print "Done"
        # get original references for each image back
        refVecs = sublists(refVecs, lenRefs)

        assert(len(refVecs)==len(candVecs))

        scores = []
        for test, refs in zip(candVecs, refVecs):
            score = []
            # compute cosine similarity between ref and cand
            # TO-DO: Optimize performance of similarity
            for ref in refs:
                sim = np.dot(test, ref)
                mag_test = np.sqrt(np.dot(test, test))
                mag_ref = np.sqrt(np.dot(ref, ref))
                sim /= (mag_test*mag_ref)
                score.append(sim)
            scores.append(score)
            print "At iteration %d/%d" % (len(scores), len(candVecs))
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute skip thought score
        score = self.compute_skip()
        return np.mean(np.array(score)), np.array(score)
