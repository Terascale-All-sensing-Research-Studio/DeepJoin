import os, sys
import logging
import traceback
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


class GracefulProcessPoolExecutor(ProcessPoolExecutor):
    """
    ProcessPoolExecutor with some added functionalities to make submitting
    new jobs and handling errors less cumbersome.
    """

    def __enter__(self):
        self._graceful_exceptions_log = defaultdict(lambda: 0)
        self._graceful_futures_queue = {}
        return super().__enter__()

    def _graceful_wait(self, empty_queue=False):
        # Will force every task to finish
        if empty_queue:
            for future in as_completed(self._graceful_futures_queue):
                try:
                    future.result()
                except:
                    logging.error(sys.exc_info())
                    logging.debug("".join(traceback.format_exception(*sys.exc_info())))
                    self._graceful_exceptions_log[sys.exc_info()[0]] += 1
            self._graceful_futures_queue.clear()
        else:
            # Will return when a single task finishes
            if len(self._graceful_futures_queue) >= self._max_workers:
                future = next(as_completed(self._graceful_futures_queue))
                try:
                    future.result()
                except:
                    logging.error(sys.exc_info())
                    logging.debug("".join(traceback.format_exception(*sys.exc_info())))
                    self._graceful_exceptions_log[sys.exc_info()[0]] += 1
                del self._graceful_futures_queue[future]

    def graceful_submit(self, *args, **kwargs):
        self._graceful_futures_queue[self.submit(*args, **kwargs)] = None
        self._graceful_wait(empty_queue=False)

    def graceful_finish(self):
        self._graceful_wait(empty_queue=True)

    def __exit__(self, *args, **kwargs):
        self._graceful_wait(empty_queue=True)
        return super().__exit__(self, args, kwargs)

    @property
    def exceptions_log(self):
        return self._graceful_exceptions_log


class GracefulProcessPoolExecutorDebug:
    """
    Standin for singlethreaded pool. Use for debugging.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def _graceful_wait(self, empty_queue=False):
        pass

    def graceful_submit(self, *args, **kwargs):
        args[0](*args[1:], **kwargs)

    def graceful_finish(self):
        pass

    def __exit__(self, *args, **kwargs):
        pass

    @property
    def exceptions_log(self):
        return {}


def random_seeder(seed):
    """Seed a thread's random generator"""
    np.random.seed(seed)


def get_file(d, fe, fn=None):
    """
    Given a root directory and a list of file extensions, recursively
    return all files in that directory that have that extension.
    """
    for f in os.listdir(d):
        fp = os.path.join(d, f)
        if os.path.isdir(fp):
            yield from get_file(fp, fe, fn)
        elif os.path.splitext(fp)[-1] in fe:
            if fn is None:
                yield fp
            elif fn == os.path.splitext(f)[0]:
                yield fp


def split_train_test(object_id_list, train_ratio):

    # Get object classes
    class_list = list(set([o.class_id for o in object_id_list]))

    # Sort into class lists
    id_by_class_train = []
    id_by_class_test = []
    flipper = False
    for c in class_list:
        o_in_class = [o for o in object_id_list if o.class_id == c]
        random.shuffle(o_in_class)

        if flipper:
            id_by_class_train.append(o_in_class[: int(len(o_in_class) * train_ratio)])
            id_by_class_test.append(o_in_class[int(len(o_in_class) * train_ratio) :])
        else:
            id_by_class_train.append(
                o_in_class[: int(np.ceil(len(o_in_class) * train_ratio))]
            )
            id_by_class_test.append(
                o_in_class[int(np.ceil(len(o_in_class) * train_ratio)) :]
            )
        flipper = not flipper

    # Flatten
    id_train_list = []
    id_test_list = []
    for idx in range(len(class_list)):
        id_train_list.extend(id_by_class_train[idx])
        id_test_list.extend(id_by_class_test[idx])

    return id_train_list, id_test_list


def rebalance_split_train_test_val(
    new_train_list,
    new_test_list,
    train_val_test_ratio,
):

    total_num = len(new_train_list) + len(new_test_list)

    assert sum(train_val_test_ratio) == 1
    train_val_num = int((train_val_test_ratio[0] + train_val_test_ratio[1]) * total_num)
    test_num = total_num - train_val_num

    if len(new_train_list) < train_val_num:
        diff_num = train_val_num - len(new_train_list)
        id_to_move = random.sample(new_test_list, diff_num)
        for id in id_to_move:
            new_test_list.remove(id)
        new_train_list += id_to_move
    else:
        diff_num = len(new_train_list) - train_val_num
        if diff_num != 0:
            id_to_move = random.sample(new_train_list, diff_num)
            for id in id_to_move:
                new_train_list.remove(id)
            new_test_list += id_to_move

    random.shuffle(new_train_list)
    train_perc = train_val_test_ratio[0] / (train_val_test_ratio[0] + train_val_test_ratio[1])
    train_perc = int(train_perc * len(new_train_list))
    new_train_list, new_val_list = new_train_list[:train_perc], new_train_list[train_perc:]

    print("num train samples: {}".format(len(new_train_list)))
    print("num val samples: {}".format(len(new_val_list)))
    print("num test samples: {}".format(len(new_test_list)))
    assert len(set(new_train_list).union(set(new_val_list)).union(set(new_test_list))) == total_num

    return new_train_list, new_val_list, new_test_list
