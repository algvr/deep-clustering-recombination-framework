import argparse
import itertools
import os

from datetime import datetime
from method import Method
from utils import flatten, resolve_relative_path


def is_dir_or_file(path):
    path = resolve_relative_path(path, base_path=None)  # use directory of script as base_path
    if os.path.isfile(path) or os.path.isdir(path):
        return path
    else:
        raise FileNotFoundError(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Processes deep clustering methods.')
    parser.add_argument('--one_log_file', default='',
                        help='path to single log file to use (uses separate log files if omitted)')
    parser.add_argument('--no_log_files', action='store_true',
                        help='do not create log files')
    parser.add_argument('--no_log_timestamps', action='store_true',
                        help='do not prefix log messages with timestamps')
    parser.add_argument('config_paths', type=is_dir_or_file, nargs='+',
                        help='list of method configuration files and directories')
    parser.add_argument('--resume_on_error', action='store_true',
                        help='if an exception is raised while processing a method, process the next method ' +
                             'instead of crashing')
    args = parser.parse_args()

    log_dir_base = os.path.dirname(args.one_log_file) if args.one_log_file != '' else None

    if log_dir_base is not None and not os.path.isdir(log_dir_base):
        os.makedirs(log_dir_base)

    file_paths = [path for path in args.config_paths if os.path.isfile(path)] +\
        flatten([[os.path.join(directory, path) for path in os.listdir(directory) if path.lower().endswith('.json')]
                 for directory in args.config_paths if os.path.isdir(directory)])

    for path in file_paths:
        # %s in log_file_path will be replaced by name of method
        path_dir = os.path.dirname(path)
        log_file_path =\
            None if args.no_log_files\
            else args.one_log_file if args.one_log_file != ''\
            else os.path.join(path_dir, datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + '%s.log')
        method = Method(config_file_path=path, log_file_path=log_file_path, log_timestamps=not args.no_log_timestamps)
        if args.resume_on_error:
            try:
                method.process()
            except Exception as e:
                delimiter = '\n\n%s\n\n' % ('**********' * 5)
                print('%sAn exception was raised while processing "%s": %s%s' % (delimiter, path, str(e), delimiter))
        else:
            method.process()
