# Filename: skthought.py
#
# Description: Class to compute evaluation metrics using skip thought vectors
#
# Creation Date: Wed Jul 22 17:54:39 EDT 2015
#
# Authors: Ramakrishna Vedantam <vrama91@vt.edu>

import pdb
from skthought_scorer import skipScorer

class SkThought():
    """
    Main Class to compute the SkipThought metric

    """
    def compute_score(self, gts, res):
        """
        Main function to compute SkipThought score
        :param  hypo_for_image (dict) : dictionary with key <image> and value <tokenized hypothesis / candidate sentence>
                ref_for_image (dict)  : dictionary with key <image> and value <tokenized reference sentence>
        :return: cider (float) : computed CIDEr score for the corpus
        """
        assert(gts.keys() == res.keys())
        imgIds = gts.keys()

        skip_scorer = skipScorer()

        for id in imgIds:
            hypo = res[id]
            ref = gts[id]
          # Sanity check.
            assert(type(hypo) is list)
            assert(len(hypo) == 1)
            assert(type(ref) is list)
            assert(len(ref) > 0)
            skip_scorer += (hypo[0], ref)

        (score, scores) = skip_scorer.compute_score()

        return score, scores

    def method(self):
        return "SkipThoughts"
