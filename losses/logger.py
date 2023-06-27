from __future__ import absolute_import

__all__ = ['Logger']

import sys
import os
import os.path as osp
import errno
def mkdir_if_missing(dirname):
    """Creates dirname if it is missing."""
    if not osp.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class Logger(object):
    """Writes console output to external text file.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py>`_

    Args:
        fpath (str): directory to save logging file.

    Examples::
       # >>> import sys
       # >>> import os
       # >>> import os.path as osp
       # >>> from logger import Logger
       # >>> save_dir = 'log/resnet50-softmax-market1501'
       # >>> log_name = 'train.log'
       # >>> sys.stdout = Logger(osp.join(args.save_dir, log_name))
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            # self.file = open(fpath, 'w')
            self.file = open(fpath, 'a')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        # msg += '\n'
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)
            # self.flush()

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        # self.console.close()
        if self.file is not None:
            self.file.close()

