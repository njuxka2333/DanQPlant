"""Build the DeepSEA dataset."""
import os
import random
import argparse
import numpy as np
from Bio import SeqIO
from process_data import extract_label,generate_data,save_to_mat, complement_dna
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# get training-valid data, test data and the label file
training_valid_record = list(SeqIO.parse('Original_data/sample_train&valid_data.fa', 'fasta'))
test_record = list(SeqIO.parse('Original_data/sample_test_data.fa','fasta'))

# get the label file
labels = []
with open('Original_data/sample_tag.txt','r') as file:
    for line in file:
        labels.append(line.strip())

#print labels
for i in range(0, len(labels), 8):
    line = " ".join(str(x) for x in labels[i:i + 8])
    print(line)

# generate training and valid data
train_data,train_labels = [],[]
valid_data,valid_labels = [],[]
valid_size = int(len(training_valid_record)*2*0.05)

for record in training_valid_record:
    if len(record.seq) == 1024:
        chrom = record.id.split('::')[1].split(':')[0] # get chromosome name
        label = extract_label(record.id,labels) # generate label matrix
        sequence = str(record.seq)  #get sequence
        complement_sequence = complement_dna(sequence) # generate complement sequence
        if np.any(label):
            if len(valid_data) < valid_size and chrom == '9':  # select a chromosome for validation(osa:9,zma:12,ath:Chr5)
                valid_data.append(generate_data(sequence))
                valid_labels.append(label)
                valid_data.append(generate_data(complement_sequence))
                valid_labels.append(label)
            else:
                train_data.append(generate_data(sequence))
                train_labels.append(label)
                train_data.append(generate_data(complement_sequence))
                train_labels.append(label)

train_data = np.stack(train_data,axis=0)
train_labels =np.stack(train_labels,axis=0)

valid_data = np.stack(valid_data,axis=0)
valid_labels = np.stack(valid_labels,axis=0)

print('{} train sequences with shape {}'.format(len(train_data),train_data.shape))
print('{} train labels with shape {}'.format(len(train_labels),train_labels.shape))
print('{} valid sequences with shape {}'.format(len(valid_data),valid_labels.shape))
print('{} valid labels with shape {}'.format(len(valid_labels),valid_labels.shape))

# generate test data
test_data = []
test_labels = []
for record in test_record:
   if len(record.seq) == 1024:
       label = extract_label(record.id,labels)
       sequence = str(record.seq)
       complement_sequence = complement_dna(sequence)
       if np.any(label):
           test_data.append(generate_data(sequence))
           test_labels.append(label)
           test_data.append(generate_data(complement_sequence))
           test_labels.append(label)

test_data = np.stack(test_data, axis=0)
test_labels = np.stack(test_labels, axis=0)
print('{} test sequences with shape {}'.format(len(test_data),test_data.shape))
print('{} test labels with shape {}'.format(len(test_labels),test_labels.shape))

os.makedirs('data' exist_ok=True)


# save training dataset,valid dataset and test dataset to train.mat, valid.mat and test.mat respectively
save_to_mat('data/train.mat', train_data, train_labels, "train")
print('training file saved ')
save_to_mat('data/valid.mat', valid_data, valid_labels, "valid")
print('valid file saved')
save_to_mat('data/test.mat', test_data, test_labels, "test")
print('test file saved')
