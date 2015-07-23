__author__ = 'vrama91'
from tokenizer.ptbtokenizer import PTBTokenizer
from bleu.bleu import Bleu
from meteor.meteor import Meteor
from rouge.rouge import Rouge
from cider.cider import Cider
from skipthought.skthought import SkThought
import numpy as np
import pdb


class RelCOCOEvalCap:
    def __init__(self, coco, cocoResOne, cocoResTwo):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        self.coco = coco
        self.cocoResOne = cocoResOne
        self.cocoResTwo = cocoResTwo
        self.params = {'image_id': coco.getImgIds()}

    def evaluate(self):
        imgIds = self.params['image_id']
        # imgIds = self.coco.getImgIds()
        gts = {}
        resOne = {}
        resTwo = {}
        for imgId in imgIds:
            gts[imgId] = self.coco.imgToAnns[imgId]
            try:
                resOne[imgId] = self.cocoResOne.imgToAnns[imgId]
                resTwo[imgId] = self.cocoResTwo.imgToAnns[imgId]
            except:
                resOne[imgId] = [self.cocoResOne.imgToAnns[imgId][0]]
                resTwo[imgId] = [self.cocoResTwo.imgToAnns[imgId][0]]

        # =================================================
        # Set up scorers
        # =================================================

        print 'tokenization...'
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        resOne = tokenizer.tokenize(resOne)
        resTwo = tokenizer.tokenize(resTwo)

        # =================================================
        # Set up scorers
        # =================================================

        print 'setting up scorers...'
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(),"METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
        ]
        """
        scorers = [
            (Cider(), "CIDEr")
        ]
        # =================================================
        # Compute scores
        # =================================================
        eval = {}
        diff = {}
        for scorer, method in scorers:
            print 'computing %s relative score...'%(scorer.method())
            _, scrOne = scorer.compute_score(gts, resOne)
            _, scrTwo = scorer.compute_score(gts, resTwo)

            if not isinstance(method, list):
                assert(len(scrOne)==len(scrTwo))
                scrOne = np.array(scrOne)
                scrTwo = np.array(scrTwo)
                diff[method] = np.greater(scrOne, scrTwo)
            else:
                for index, ngrams in enumerate(method):
                    tempOne = np.array(scrOne[index])
                    tempTwo = np.array(scrTwo[index])
                    diff[ngrams] = np.greater(tempOne, tempTwo)

            self.eval = diff
