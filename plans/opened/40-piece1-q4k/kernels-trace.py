# kernels-trace.py CYCLE_DIR: Nsight Systems' kernel trace of `adad profile --repeat 3` (nsys_cuda_gpu_trace.csv) matched
# with its launch lines (nsys-launches.txt: a line per kernel of each shape's lane, each lane run 3 times): per shape and
# lane, each kernel's duration in the first run and the median of the next two, the device's idle gaps between the run's
# kernels, and the run's span (first kernel's start to last kernel's end).
import csv, os, re, sys, statistics
d = sys.argv[1]
rows = list(csv.DictReader(open(os.path.join(d, 'nsys_cuda_gpu_trace.csv'), encoding='utf-8')))
kern = [r for r in rows if not r['Name'].startswith('[CUDA')]
launch = [l.rstrip('\n') for l in open(os.path.join(d, 'nsys-launches.txt'), encoding='utf-8') if l.startswith('launch ')]
groups, cur = [], None
for l in launch:
    m = re.match(r'launch \d+: (.*): (\S+)$', l)
    key, k = m.group(1), m.group(2)
    if cur is None or cur[0] != key:
        cur = (key, [])
        groups.append(cur)
    cur[1].append(k)
at = 0
print('shape and lane | kernels: first run / median of runs 2-3 (us) | idle gaps between kernels (us, median run) | span (us, median run)')
for key, names in groups:
    runs = []
    for r in range(3):
        run = kern[at:at + len(names)]
        at += len(names)
        assert [x['Name'] for x in run] == names, (key, [x['Name'] for x in run], names)
        runs.append(run)
    dur = lambda run, i: int(run[i]['Duration (ns)']) / 1000
    start = lambda run, i: int(run[i]['Start (ns)']) / 1000
    parts = '; '.join(f'{n} {dur(runs[0], i):.1f} / {statistics.median([dur(runs[1], i), dur(runs[2], i)]):.1f}' for i, n in enumerate(names))
    gaps = [statistics.median([start(run, i + 1) - start(run, i) - dur(run, i) for run in runs[1:]]) for i in range(len(names) - 1)]
    span = statistics.median([start(run, len(names) - 1) + dur(run, len(names) - 1) - start(run, 0) for run in runs[1:]])
    print(f'{key} | {parts} | {", ".join(f"{g:.1f}" for g in gaps) or "-"} | {span:.1f}')
print(f'kernels matched: {at} of {len(kern)}')
