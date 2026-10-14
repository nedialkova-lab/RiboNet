"""Parallel processing helper used by the preprocessing and interpretation steps."""

import logging
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from tqdm.auto import tqdm

log = logging.getLogger(__name__)


def parallel_map(func, iterable, threads=1, chunksize=1, desc=None):
    '''Apply ``func`` to every item of ``iterable`` using up to ``threads`` processes.

    Results are returned as a list in the order of ``iterable``. Unlike
    ``multiprocessing.Pool``, which waits forever if a worker process is killed
    (e.g. by the system when running out of memory), an error is raised as soon
    as a worker dies. With ``threads=1``, items are processed in the current
    process, which also eases debugging in interactive sessions.

    Parameters
    ----------
    func : callable
        Function applied to each item. Must be picklable (e.g. a module-level
        function, a method, or a ``functools.partial`` of those).
    iterable : iterable
        Items to process.
    threads : int, default = 1
        Maximum number of worker processes.
    chunksize : int, default = 1
        Number of items sent to a worker at once.
    desc : str, default = None
        If given, show a progress bar with this description.

    Returns
    -------
    list
        ``[func(item) for item in iterable]``
    '''
    items = list(iterable)
    progress = dict(total=len(items), desc=desc, disable=desc is None)
    if threads <= 1 or len(items) <= 1:
        return [func(item) for item in tqdm(items, **progress)]
    try:
        with ProcessPoolExecutor(max_workers=min(threads, len(items))) as executor:
            return list(tqdm(executor.map(func, items, chunksize=max(1, chunksize)), **progress))
    except BrokenProcessPool as e:
        err_message = ("A worker process terminated abruptly, most likely because it ran out of memory. "
                       "Reduce the number of threads (--threads) or increase the available memory.")
        log.error(err_message)
        raise RuntimeError(err_message) from e
