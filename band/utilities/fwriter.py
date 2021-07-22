"""
Easy logging to file when training a neural network
Sam Nguyen
"""

import os


class FWriter:
    """Open file and log to file"""

    def __init__(self, is_debug, fn):
        self.is_debug = is_debug
        if (not is_debug):
            self.f = open(fn, 'a+', encoding='utf8')
        else:
            self.f = None

        # TODO: add exception handling x

    def write(self, s, force=True, p=False, nl=True):
        """
            write s to file and flush it immediately
            with Force = True, call os.sync to force flushing
        """
        if (self.is_debug or p):
            # if debugging, then print instead
            print(s)
        else:
            if (nl):
                s += '\n'
            self.f.write(s)
            self.f.flush()
            if force:
                os.fsync(self.f.fileno())

    def close(self):
        """Close file"""
        if (self.f):
            self.f.close()
