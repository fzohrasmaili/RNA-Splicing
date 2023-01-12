#Similar to extract_events.py but having all events in one data set (no training/testing split)
import sys
import numpy as np
import re
import os
import torch
import sklearn
import random
#####################################################################################
# I. Build testing and training dataset
#1. Extract positive events
data_type = sys.argv[1] # e.g "BRCA" #to be changed accordingly
input_file = f"results_alt_splice_difftest_gdc/testing_{data_type}-T_vs_{data_type}-N_MERG/test_results_C2_exon_skip.gene_unique.tsv"
file = open (input_file,'r')
header_line = next(file)
test_positive_events =[]
train_positive_events = []
splicing_label ={} #significant vs non-significant
binary_splicing_label={} #Positive vs. Negative
for line in file:
    columns =line.split()
    # Extract and fix event ID
    event_string = columns[0]
    #event_search = re.search('exon_skip_(\d+)',event_string)
    #if event_search:
    #    id_num=event_search.group(1)
    #    event_id='exon_skip.' + id_num
    event_id = event_string
    p_value = float (columns[2])
    adjus_p_value =float(columns[3])
    log2FC = float(columns [6])
    #Extract testing events
    if (adjus_p_value <= 0.9):
        #Filter based on log2FC:
        if (log2FC <= -0.5):
            test_positive_events.append(event_id)
            splicing_label[event_id] = 1 #significant
            binary_splicing_label[event_id] = 0 #Negative
        elif (log2FC >= 0.5):
            test_positive_events.append(event_id)
            splicing_label[event_id] = 1 #significant
            binary_splicing_label[event_id] = 1 #Positive
    #Extract training events
    elif (p_value<=0.5):
        #Filter based on log2FC:
        if (log2FC <= -0.5):
            train_positive_events.append(event_id)
            splicing_label[event_id] = 1 #significant
            binary_splicing_label[event_id] = 0 #Negative
        elif (log2FC >= 0.5):
            train_positive_events.append(event_id)
            splicing_label[event_id] = 1 #significant
            binary_splicing_label[event_id] = 1 #Positive
test_pos_length = len(test_positive_events)
train_pos_length = len (train_positive_events)

print ("Extraction of positive events complete")
#2. Extract negative events
coord_file = "exon_skip_coordinates.txt"
neg_file = coord_file
neg_f = open (neg_file, 'r')
general_negatives =[]
test_negative_events=[]
train_negative_events=[]
event_chromosome ={}
event_coordinates={}
header_line = next (neg_f)
for line in neg_f:
    cols=line.split()
    ev_id= cols[0]
    event_chromosome[ev_id]= cols[2]
    event_coordinates [ev_id] = cols [4]
    #Check if event is in the positive dataset
    if (ev_id not in test_positive_events) and (ev_id not in train_positive_events):
        # Add event to negative dataset
        general_negatives.append(ev_id)
        splicing_label [ev_id] = 0


#II.Split negatives into testing and training
#1. Sort out indices
sz_neg=len(general_negatives)
shuffled_indices=torch.randperm(sz_neg)
general_negatives = np.asarray(general_negatives)
test_indices = shuffled_indices[0:test_pos_length]
train_start_idx = test_pos_length+1
train_end_idx = test_pos_length+train_pos_length+1
train_indices = shuffled_indices[train_start_idx:train_end_idx]
#2. Build negative dataset
test_negative_events = general_negatives [test_indices]
train_negative_events = general_negatives [train_indices]
print ("Extraction of negative events complete")
print ("Dataset size:")
print ("Number of training positive events: ", str(train_pos_length))
print ("Number of training negative events: ", str(len(train_negative_events)))
print ("Number of testing positive events: ", str(test_pos_length))
print ("Number of testing negative events: ", str(len(test_negative_events)))
#Merge train and test events
train_events = [*train_positive_events, *train_negative_events]
random.shuffle(train_events)
test_events = [*test_positive_events, *test_negative_events]
random.shuffle(test_events)
events_list = [*train_events, *test_events]
#Save data
def save_d(filename, data):
  with open(filename, 'w') as f:
      for ev1 in data:
          f.write(f'{ev1}\t{splicing_label[ev1]}\n')
event_type = "exon_skip" # to be changed accordingly to alt_3prime alt_5prime mutex_exon intron_retention

save_d(f'data/{data_type}/events_data.txt', events_list)

###################################################################################

# III. generate BED files:
#Fasta file: /Users/FatimaZohra/Desktop/McGill/Research/Splicing/BC_Data/hg38_genome.fasta

# bedfile_name = 'data/'+data_type+'/'+event_type+'/seqs.bed'
bedfile_name = 'data/'+data_type+'/seqs.bed'
bed_file =open (bedfile_name,'w')
for e in events_list:
    coors = event_coordinates[e].split(':')
    chro = "chr"+event_chromosome[e]
    # start = int (coors[0])-250
    # end = int (coors[1])+250
    start = int (coors[0])-100
    end = int (coors[1])+100
    #print (chro,str(start),str(end),e )
    bed_line = chro+"\t"+str(start)+"\t"+str(end)+"\t"+e+"\n"
    bed_file.write(bed_line)
#run bedtools to extract sequences:
#for TCGA in "BLCA" "BRCA" "COAD" "HNSC" "KIRC" "KIRP" "LIHC" "LUAD" "LUSC" "PRAD" "READ" "STAD" "THCA" "UCEC"; do bedtools getfasta -fi hg38_genome.fasta -bed data/${TCGA}/seqs.bed -fo data/${TCGA}/seqs.fa.out -name; done
#$ bedtools getfasta -fi /Users/FatimaZohra/Desktop/McGill/Research/Splicing/BC_Data/hg38_genome.fasta -bed seqs.bed -fo seqs.fa.out -name
