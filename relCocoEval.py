
# coding: utf-8

from pycocotools.coco import COCO
from pycocoevalcap.relEval import RelCOCOEvalCap
import json
from json import encoder
import numpy as np
import pdb
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# set up file names and paths
dataDir = '.'
dataType = 'valid2014'
algName = ('human', 'valid')

annFile='%s/annotations/captions_%s.json'%(dataDir,dataType)

subtypes=['results', 'evalImgs', 'eval']
[resFileOne, evalImgsFile, evalFile]= ['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName[0],subtype) for subtype in subtypes]
[resFileTwo, evalImgsFile, evalFile]= ['%s/results/captions_%s_%s_%s.json'%(dataDir,dataType,algName[1],subtype) for subtype in subtypes]

coco = COCO(annFile)
cocoResOne = coco.loadRes(resFileOne)
cocoResTwo = coco.loadRes(resFileTwo)

# create cocoEval object by taking coco and cocoRes
cocoEval = RelCOCOEvalCap(coco, cocoResOne, cocoResTwo)
# evaluate results
cocoEval.evaluate()

for metric, score in cocoEval.eval.items():
    # print how many times method 1 scored more than method 2
    rel = sum(score)/float(0.5*len(score))
    print "Metric %s scores %s higher than %s %f times" % (metric, algName[0],
                                                          algName[1], rel)
