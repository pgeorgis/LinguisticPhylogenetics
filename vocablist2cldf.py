import os, glob
from collections import defaultdict
from phonSim import segment_ipa
import argparse

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

def vocablist2cldf(lang_files, combine_diphthongs=True):
    # start_dir = os.getcwd()
    # os.chdir(list_dir)
    # lang_files = glob.glob('*.txt')
    # lang_files = [file for file in lang_files
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
                    entry['Segments'] = ' '.join(segment_ipa(tr, combine_diphthongs=combine_diphthongs))
                    entry['Cognate_ID'] = gloss
                    entry['Loan'] = 'TRUE' if '*' in gloss else ''
                    entry['Comment'] = line[2] if len(line) > 3 else ''
                    entry['Source'] = ''
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--family', required=True, help='Family name')
    parser.add_argument('--dir', required=True, help='Directory from which to extract txt files')
    parser.add_argument('--langs', required=True, nargs='+', help='Language names to extract (.txt extension added automatically)')
    parser.add_argument('--dest', required=True, help='Destination directory for CLDF file')
    args = parser.parse_args()

    lang_files = [os.path.join(args.dir, l + '.txt') for l in args.langs]
    data = vocablist2cldf(lang_files)

    write_data(data, output_file=os.path.join(args.dest, args.family+'_data.csv'), sep='\t')