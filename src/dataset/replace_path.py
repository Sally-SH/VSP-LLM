import os
import shutil

def replace(lrs3_path, outdir):
    splits = ['train','test']
    for split in splits:
        file_path = f'./{split}.tsv'       
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

        with open(f"{outdir}/{split}.tsv",'w') as wf:
            wf.write(''.join(tsv_list))
        shutil.copyfile(f"./{split}.wrd", f"{outdir}/{split}.wrd")
        shutil.copyfile(f"./{split}.cluster_counts", f"{outdir}/{split}.cluster_counts")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='replace the path in manifest', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lrs3', type=str, help='path to preprocessed lrs3 dataset')
    parser.add_argument('--outdir', type=str, help='path to save converted manifest')
    args = parser.parse_args()
    os.makedirs(args.outdir,exist_ok = True)
    replace(args.lrs3, args.outdir)
