import glob
import math
THREADS=7
files = glob.glob(DATA_PATH)
DATA_PATH = '/home/ffajri/Data/segmenter/*'
size = int(math.ceil(1.0*len(files)/THREADS))

allfiles = []
for i in range(THREADS):
    start = i * size
    end = start + size
    if end > len(files):
        end = len(files)
    p = files[start:end]
    allfiles.append(p)
    if end == len(files):
        break

for idx in range(len(allfiles)):
    f = open(str(idx)+'.list', 'w')
    for l in allfiles[idx]:
        f.write(l+'\n')
    f.close()
