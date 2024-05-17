import os
import shutil

def replace(lrs3_path):
    langs = ['en-en','en-es','en-fr','en-it','en-pt']
    for lang in langs:
        src, tgt = lang.split('-')
        if src == tgt:
            task = 'vsr'
            lang = src
        else:
            task = 'vst'
            lang = f"{src}/{tgt}"
            
        file_path = f'../../labels/{task}/{lang}/test.tsv'       
        f = open(file_path,'r')
        lines = f.readlines()
        f.close()
        tsv_list = ['/\n']
        wrd_list = []
        for idx, line in enumerate(lines[1:]): 
            items = line.split('\t')
            video_path = items[1]
            new_video_path = video_path.replace('/path/to/lrs3',lrs3_path)
            items[1] = new_video_path
            audio_path = items[2]
            new_audio_path = audio_path.replace('/path/to/lrs3',lrs3_path)
            items[2] = new_audio_path
            tsv_list.append('\t'.join(items))

        out_path = f"./{task}/{lang}"
        os.makedirs(out_path,exist_ok=True)
        with open(f"{out_path}/test.tsv",'w') as wf:
            wf.write(''.join(tsv_list))
        shutil.copyfile(f"../../labels/{task}/{lang}/test.wrd", f"{out_path}/test.wrd")
        shutil.copyfile(f"../../labels/{task}/{lang}/test.cluster_counts", f"{out_path}/test.cluster_counts")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='replace the path in manifest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='path to preprocessed lrs3 dataset')
    args = parser.parse_args()
    replace(args.lrs3)
