#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 21:20:50 2021

@author: phgeorgis
"""
import os, glob
from collections import defaultdict
from loadLangs import *

def write_data(data_dict, output_file, sep='\t'):
    features = list(data_dict[list(data_dict.keys())[0]].keys())
    with open(output_file, 'w') as f:
        header = sep.join(features)
        f.write(f'{header}\n')
        for i in data_dict:
            try:
                values = sep.join([data_dict[i][feature] for feature in features])
            except TypeError:
                print(data_dict[i])
            f.write(f'{values}\n')

def vocablist2cldf(family_name, lang_files, dest_dir):
    #start_dir = os.getcwd()
    #os.chdir(list_dir)
    #lang_files = glob.glob('*.txt')
    #lang_files = [file for file in lang_files
    #             if file.split('.')[0] in language_list]
    
    data = defaultdict(lambda:{})
    i = 0
    for file in lang_files:
        with open(file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) > 0:
                    line = line.split('/')
                    gloss = line[0]
                    orth = line[1]
                    tr = line[-1] if len(line) < 4 else line[2]
                    lang = file.split('/')[-1].split('.')[0]
                    i += 1
                    entry = data[i]
                    entry['ID'] = lang
                    entry['Language_ID'] = lang
                    entry['Glottocode'] = ''
                    entry['ISO 639-3'] = ''
                    entry['Parameter_ID'] = gloss
                    entry['Value'] = orth
                    entry['Form'] = tr
                    entry['Segments'] = ' '.join(segment_word(tr))
                    entry['Cognate_ID'] = gloss
                    entry['Loan'] = 'TRUE' if '*' in gloss else ''
                    entry['Comment'] = line[2] if len(line) > 3 else ''
                    entry['Source'] = ''
    
    write_data(data, output_file=dest_dir+f'/{family_name}_data.csv', sep='\t')
    #os.chdir(start_dir)
    
    
    