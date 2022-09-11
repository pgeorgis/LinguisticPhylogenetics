import os, glob, re
from collections import defaultdict
import pandas as pd
import numpy as np
from pathlib import Path
local_dir = Path(str(os.getcwd()))
parent_dir = local_dir.parent


def write_lang_index(dataset_file, language_metadata, output_file):
    data = pd.read_csv(dataset_file, sep='\t')
    data = data.replace(np.nan, '', regex=True)
    langs = {}
    words = defaultdict(lambda:defaultdict(lambda:[]))
    for i in range(len(data)):
        lang_name = data['Language_ID'][i]
        glottocode = data['Glottocode'][i]
        iso_code = data['ISO 639-3'][i]
        concept = data['Parameter_ID'][i]
        word = data['Form'][i]
        lang_id = data['ID'][i].split('_')[0]
        langs[lang_name] = lang_id, glottocode, iso_code
        words[lang_name][concept].append(word)
    with open(output_file, 'w') as f:
        f.write(','.join('ID Name Glottocode Glottolog_Name ISO639P3code Macroarea Latitude Longitude Family Concept_Count Word_Count'.split()))
        f.write('\n')
        for lang_name in sorted(list(langs.keys())):
            lang_id, glottocode, iso_code = langs[lang_name]
            metadata_index = lang_metadata.index[lang_metadata['Name'] == lang_name].tolist()[0]
            glottolog_name = lang_metadata['Glottolog Name'][metadata_index]
            macroarea = lang_metadata['Macroarea'][metadata_index]
            latitude = str(lang_metadata['Latitude'][metadata_index])
            longitude = str(lang_metadata['Longitude'][metadata_index])
            family = lang_metadata['Family'][metadata_index]
            concept_count = str(len(words[lang_name]))
            word_count = str(sum([len(set(words[lang_name][concept])) for concept in words[lang_name]]))
            f.write(','.join([lang_id, lang_name, glottocode, glottolog_name,
                              iso_code, macroarea, latitude, longitude, family,
                              concept_count, word_count]))
            f.write('\n')

#%%
#LOAD LANGUAGE METADATA
lang_metadata_file = str(parent_dir) + '/Datasets/Languages.csv'
lang_metadata = pd.read_csv(lang_metadata_file, sep='\t')

#Replace NA values in dataframe with empty string
lang_metadata = lang_metadata.replace(np.nan, '', regex=True)

#Ensure that all languages have unique names, print warning message otherwise
warnings = []
for i in range(len(list(lang_metadata['Name']))):
    name = lang_metadata['Name'][i]
    name_entries = lang_metadata.index[lang_metadata['Name'] == name].tolist()
    if len(name_entries) > 1:
        if name not in warnings:
            print(f'Warning: "{name}" appears {len(name_entries)} times in dataset (indices {", ".join([str(j) for j in name_entries])})!')
            warnings.append(name)

#%%
#WRITE LANGUAGE INDICES
for dataset in ['Arabic',
                'Balto-Slavic',
                'Bantu',
                'Dravidian',
                'Hellenic',
                'Hokan',
                'Italic',
                'Japonic',
                'Polynesian',
                'Quechuan',
                'Sinitic',
                'Turkic',
                'Uralic',
                'Uto-Aztecan',
                'Vietic'                
                ]:
    directory = str(parent_dir) + f'/Datasets/{dataset}'
    os.chdir(directory)
    dataset = re.sub('-', '_', dataset.lower())
    dataset_file = directory + '/' + dataset + '_data.csv'
    write_lang_index(dataset_file, lang_metadata, output_file='languages.csv')
    os.chdir(local_dir)



