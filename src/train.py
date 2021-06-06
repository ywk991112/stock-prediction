from solver import get_solver
from feature import get_feature
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
from os import listdir
from os.path import isfile, join, basename, normpath
import pickle
import argparse
import yaml

filename = 'data/Combined_News_DJIA.csv'

def main(config_path):
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.CLoader)
        stream.close()
    features = get_feature(filename, config['feature'], config['feature_param'])
    solver = get_solver(config['model'])(features, **config['model_param'])
    solver.fit()
    return config, solver.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Stock Prediction with Text Information")
    parser.add_argument('--config_dir', type=str, default='config',
                        help='Config path for the experiment')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Run all configs in parallel')
    args = parser.parse_args()
    config_paths = [join(args.config_dir, f) for f in listdir(args.config_dir) if isfile(join(args.config_dir, f))]
    if args.parallel:
        results = Parallel(n_jobs=-1)(delayed(main)(config_path) for config_path in tqdm(config_paths))
    else:
        results = [main(config_path)for config_path in tqdm(config_paths)]
    save_path = join('results', f'{basename(normpath(args.config_dir))}.tar')
    print(f'Results are saved in {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
