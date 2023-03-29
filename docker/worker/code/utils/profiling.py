# profile
import cProfile, pstats, io
from pstats import SortKey


def profiling_before() -> cProfile.Profile:
    """
    :return: pr
    """
    pr = cProfile.Profile()
    pr.enable()
    return pr


def profiling_after(pr: cProfile.Profile):
    pr.disable()
    s_out = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s_out).sort_stats(sortby)
    ps.print_stats(400)
    text = s_out.getvalue().split('\n')
    libs = 'docker/worker/code'
    print('Internal:')
    for x in text:
        if libs in x:
            print(x)
    print('External:')
    i = 0
    for x in text:
        if libs not in x and i < 30:
            i += 1
            print(x)

