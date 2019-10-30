import numpy as np
import glob
from NeuralRST.in_out.instance import CResult
from NeuralRST.in_out.rst_feature import RSTFeature
import threading
import math
from multiprocessing import Process

THREADS = 40
SOURCE_PATH = '/home/ffajri/Data/Petition/US/processed/output_tree/*'
TARGET_PATH = '/home/ffajri/Data/Petition/US/processed/output_shallow/'
allfiles = glob.glob(SOURCE_PATH)

def run_thread(files):
    for filepath in files:
        filename = filepath.split('/')[-1].replace('.npy', '')
        cresult = CResult()
        cresult.subtrees = list(np.load(filepath))
        tree = cresult.obtain_tree()
        rst_feature = RSTFeature()
        feat = rst_feature.generate_heuristic_feature(tree)
        np.save(TARGET_PATH+filename, feat)

partitions  = []
size = int(math.ceil(1.0*len(allfiles)/THREADS))
processes = list()
for i in range(THREADS):
    start = i * size
    end = start + size
    if end > len(allfiles):
        end = len(allfiles)
    p = allfiles[start:end]
    
    process = Process(target=run_thread, args=(p,))
    process.start()
    processes.append(process)
    if end == len(allfiles):
        break
for process in processes:
    process.join()





