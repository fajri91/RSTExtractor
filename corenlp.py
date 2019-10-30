from subprocess import call
import os
import glob
import threading
import math
import argparse


scriptdir = '.'
THREADS = 27

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", help="provide source path", type=str)
parser.add_argument("-t", "--target", help="provide target path", type=str)

args = parser.parse_args()
if args.source:
    PATH=args.source
if args.target:
    TARGET=args.target

files = glob.glob(PATH)
targets = glob.glob(TARGET+'/*')
TOTAL_FILES = len(files)
BATCH_SIZE = int(math.ceil(TOTAL_FILES/THREADS))

def run_thread(ftmp):
    os.system('/usr/bin/java -mx150g -cp "stanford-corenlp/*" edu.stanford.nlp.pipeline.StanfordCoreNLP -annotators tokenize,ssplit,pos,lemma,ner,parse -ssplit.eolonly -tokenize.whitespace true -filelist '+ftmp+' -outputFormat xml -outputDirectory '+TARGET)

def generate_listfile(start, end, ftmp):
    sliced_files = files[start:end]
    w = open (ftmp, 'w')
    for f in sliced_files:
        fname = TARGET+'/' + f.split('/')[-1] + '.xml'
        if not fname in targets:
            w.write(f+'\n')
    w.close()

for i in range(THREADS):
    start = i * BATCH_SIZE
    end = start + BATCH_SIZE
    if end > TOTAL_FILES:
        end = TOTAL_FILES
    
    ftmp = 'tmp'+str(i)+'.txt'
    generate_listfile(start,end,ftmp)
    
    t = threading.Thread(target=run_thread, args=(ftmp,))
    t.start()
    print (start, end)







