#Similar to process_sequences.py but for all events in one set (no train/test separation)
import sys
import os
import re
import numpy
import math
# Extract and process event sequences
#event_type = "exon_skip" # to be changed accordingly to alt_3prime alt_5prime mutex_exon intron_retention
data_type = sys.argv[1] #e.g. "BLCA"
seq_map ={}
#sequence_file = 'data/'+data_type+'/'+event_type+'/seqs.fa.out'
sequence_file = 'data/'+data_type+'/all_seqs.fa.out'
seq_FH = open (sequence_file,'r')
for line in seq_FH:
     header_search =re.search('^>(\S+)::\S+',line)
     if header_search:
         event_name = header_search.group(1)
         sequence = (next (seq_FH)).strip().lower()
         sequence = sequence.replace('a','1').replace('t','2').replace('c','3').replace('g','4')
         lgth = len(sequence)
         #Extract introns
         left_intron = sequence [:100]
         ri_intr_st = len (sequence) - 100
         right_intron = sequence [ri_intr_st:]
         #Extract exon_skip
         left_exon =''
         right_exon =''
         mid_point = math.ceil(lgth/2)
         first_lgth = mid_point-99
         second_lgth = ri_intr_st - mid_point + 1
         #Extract left exon:
         if (first_lgth >=100):
            left_exon = sequence[100:200]
         else:
            diff = 100 - first_lgth
            left_exon = sequence[100:mid_point+1]+('0'*diff);
        #Extract right exon:
         if (second_lgth >=100):
            right_exon = sequence [-200:-100]
         else:
            diff = 100-second_lgth
            right_exon=('0'*diff)+sequence[mid_point:ri_intr_st+1]
         seq_map [event_name] = left_intron+left_exon+right_intron+right_exon
         #Check points:
         # print ("Length of left intron: ",str(len(left_intron)))
         # print ("Length of right intron: ",str(len(right_intron)))
         # print ("Length of left exon: ",str(len(left_exon)))
         # print ("Length of right exon: ",str(len(right_exon)))
print ("Sequence created")
# for event, seque in seq_map.items():
#     print (event, " -> ", seque)

# Build Training/Testting dataset:
#Training data set
events_dataset = 'data/'+data_type+'/events_data.txt'
X_file = open('data/'+data_type+'/events_X.txt',"w")
Y_file = open('data/'+data_type+'/events_Y.txt',"w")
#E_file = open('data/'+data_type+'/Training/Train_Events.txt',"w")
with open(events_dataset, 'r') as event_data:
  for line in event_data:
      ev_nm , label = line.split()
      if ev_nm  in seq_map:
          ev_seq = seq_map[ev_nm]
          X_file.write(ev_seq + "\n")
          Y_file.write(label + "\n")
          #E_file.write(ev_nm + "\t" + label + "\n")
print ("Data saved")
