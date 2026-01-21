import os
import argparse
import time
import glob

from tqdm import tqdm
from functools import partial
import multiprocessing



def preprocessing(patch_item):



def prepare_statistic(data, stats):



def run(args):
    start_time = time.time()
    patch_dict = {}

    # files
    for path in glob.glob(os.path.join(args.input, 'df_shapes_*.csv')):
        filename = os.path.basename(path)
        patch = filename.split('df_shapes_')[1].split('_')[0]
        patch_dict[patch]['shapes'].append(path)

    os.makedirs(args.output, exist_ok=True)

    patch_items = list(patch_dict.items())
    pool_worker = partial(preprocessing)
    filtered_patch_items = {}

    # preprocessing
    with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
        for patch_id, df_shapes in tqdm(pool.imap_unordered(pool_worker, patch_items),
                                        total=len(patch_items),
                                        desc='Preprocessing'):
            if df_shapes is not None:
                filtered_patch_items[patch_id] = df_shapes

    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Completed in {minutes} minutes and {seconds} seconds")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='../../data/15_prct')
    parser.add_argument('-o', '--output', type=str, default='../output/')
    parser.add_argument('-w', '--workers', type=int, default=multiprocessing.cpu_count()-1)

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
