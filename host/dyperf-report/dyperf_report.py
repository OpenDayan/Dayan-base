#!/usr/bin/env python3
#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# Dayan-base is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

"""Report task elapsed cycles from dyperf Chrome Trace JSON (Python 3.8+)."""
import argparse
import csv
from decimal import Decimal, localcontext
import json
from pathlib import Path
import re
import sys

PERIOD = 1 << 48
SERVICES = {'scheduler', 'rpc-receive', 'rpc-send', 'profile-flush', 'PCU snapshots'}


def integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError('%s must be a nonnegative integer' % label)
    return value


def analyze_trace(data, source, task=None):
    if not isinstance(data, dict) or not isinstance(data.get('traceEvents'), list):
        raise ValueError('expected an object containing a traceEvents array')
    events = data['traceEvents']
    names, cores = {}, {}
    for event in events:
        if not isinstance(event, dict):
            raise ValueError('traceEvents contains a non-object')
        if event.get('ph') != 'M':
            continue
        if event.get('name') not in ('thread_name', 'process_name'):
            continue
        pid = integer(event.get('pid'), 'pid')
        args = event.get('args')
        if not isinstance(args, dict) or not isinstance(args.get('name'), str):
            raise ValueError('invalid name metadata')
        if event['name'] == 'thread_name':
            key = pid, integer(event.get('tid'), 'tid')
            if key in names and names[key] != args['name']:
                raise ValueError('conflicting thread names for %s' % (key,))
            names[key] = args['name']
        else:
            match = re.fullmatch(r'cluster (\d+) hart (\d+)', args['name'])
            if match:
                cores[pid] = tuple(map(int, match.groups()))

    clocks, tasks, missing_names = {}, {}, set()
    for event in events:
        if event.get('cat') != 'proc_state' or event.get('ph') not in ('B', 'E'):
            continue
        pid = integer(event.get('pid'), 'pid')
        tid = integer(event.get('tid'), 'tid')
        args = event.get('args')
        if not isinstance(args, dict):
            raise ValueError('proc_state event has no args')
        raw = integer(args.get('counter'), 'counter')
        if raw >= PERIOD:
            raise ValueError('expected the 48-bit raw counter emitted by dyperf')
        previous, epoch = clocks.get(pid, (raw, 0))
        if raw < previous:
            if previous - raw <= PERIOD // 2:
                raise ValueError('proc_state counters are out of order for pid=%s' % pid)
            epoch += PERIOD
        clocks[pid] = raw, epoch
        cycle = epoch + raw
        if event['ph'] != 'B':
            continue
        key = pid, tid
        if key not in names:
            # dyperf's synthetic scheduler has tid == pid, without thread metadata.
            if tid != pid:
                missing_names.add(key)
            continue
        name = names[key]
        if name in SERVICES or (task is not None and name != task):
            continue
        info = tasks.setdefault(key, {'name': name, 'start': None, 'end': None,
                                      'exited': False})
        if info['exited']:
            continue
        if event.get('name') == 'RUNNING' and info['start'] is None:
            info['start'] = cycle
        elif event.get('name') == 'WAIT_EXIT':
            info['exited'] = True
            info['end'] = cycle

    intervals = []
    excluded = {'incomplete': 0, 'missing_start': 0, 'not_started': 0,
                'missing_name': len(missing_names)}
    for (pid, tid), info in tasks.items():
        if info['start'] is None:
            excluded['missing_start' if info['exited'] else 'not_started'] += 1
        elif not info['exited']:
            excluded['incomplete'] += 1
        else:
            cluster, hart = cores.get(pid, (None, None))
            intervals.append({'source': source, 'task': info['name'],
                              'pid': pid, 'tid': tid, 'cluster': cluster, 'hart': hart,
                              'start': info['start'], 'end': info['end'],
                              'cycles': info['end'] - info['start']})
    return intervals, excluded


def mean_text(total, count):
    with localcontext() as ctx:
        ctx.prec = max(40, len(str(total)) + 10)
        return format(Decimal(total) / Decimal(count), '.3f')


def summarize(intervals, by_file=False):
    groups = {}
    for item in intervals:
        # Without core metadata, do not combine unrelated PIDs across files.
        core = ((item['cluster'], item['hart']) if item['cluster'] is not None
                else ('unknown', item['source'], item['pid']))
        key = item['task'], core, item['source'] if by_file else None
        if key not in groups:
            groups[key] = {'task': item['task'], 'cluster': item['cluster'],
                           'hart': item['hart'], 'pid': item['pid'],
                           'source': item['source'], 'count': 0, 'total': 0,
                           'min': item['cycles'], 'max': item['cycles']}
        group = groups[key]
        group['count'] += 1
        group['total'] += item['cycles']
        group['min'] = min(group['min'], item['cycles'])
        group['max'] = max(group['max'], item['cycles'])
    return list(groups.values())


def render(intervals, groups, excluded, by_file=False):
    quote = lambda text: json.dumps(text, ensure_ascii=False)
    lines = ['Task cycle report',
             'Scope: first RUNNING -> WAIT_EXIT, including intervening waits.',
             'Counters are device cycles; host time and PCU sample timestamps are not used.']
    for item in intervals:
        core = ('cluster=%s hart=%s' % (item['cluster'], item['hart'])
                if item['cluster'] is not None else 'core=unknown pid=%s' % item['pid'])
        lines.append('[TASK_PERF] file=%s task=%s %s tid=%s start=%s end=%s cycles=%s' %
                     (quote(item['source']), quote(item['task']), core, item['tid'],
                      item['start'], item['end'], item['cycles']))
    lines.extend(['', 'Performance Summary',
                  'Avg cycles = arithmetic mean of completed tasks in the supplied files.',
                  ('File | ' if by_file else '') +
                  'Task | Core | Count | Avg cycles | Min cycles | Max cycles'])
    for group in groups:
        core = ('cluster=%s hart=%s' % (group['cluster'], group['hart'])
                if group['cluster'] is not None else
                'unknown (%s pid=%s)' % (quote(group['source']), group['pid']))
        row = '%s | %s | %s | %s | %s | %s' % (
            quote(group['task']), core, group['count'],
            mean_text(group['total'], group['count']), group['min'], group['max'])
        lines.append((quote(group['source']) + ' | ' if by_file else '') + row)
    if not groups:
        lines.append('(no completed task intervals)')
    lines.append('Excluded: ' + ' '.join('%s=%s' % x for x in excluded.items()))
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, epilog=
        'Group inputs of the same shape together; task names do not encode shape. '
        'Start capture before the workload. Already-running tasks cannot be timed fully. '
        'Overlapping parent/child intervals must not be summed as device elapsed time.')
    parser.add_argument('traces', nargs='+', type=Path, help='dyperf trace JSON files')
    parser.add_argument('--task', help='report only this exact task name')
    parser.add_argument('--by-file', action='store_true', help='keep Summary groups separate per file')
    parser.add_argument('--csv', type=Path, help='also write Summary as UTF-8 CSV')
    parser.add_argument('--strict', action='store_true', help='return nonzero for excluded or empty results')
    opts = parser.parse_args(argv)
    intervals = []
    excluded = dict.fromkeys(['incomplete', 'missing_start', 'not_started', 'missing_name'], 0)
    seen = set()
    try:
        if opts.csv and opts.csv.resolve() in {p.resolve() for p in opts.traces}:
            raise ValueError('CSV output must not overwrite an input trace')
        for path in opts.traces:
            resolved = path.resolve()
            if resolved in seen:
                raise ValueError('duplicate input file: %s' % path)
            seen.add(resolved)
            with path.open(encoding='utf-8-sig') as f:
                data = json.load(f)
            try:
                rows, counts = analyze_trace(data, str(path), opts.task)
            except ValueError as error:
                raise ValueError('%s: %s' % (path, error)) from error
            intervals.extend(rows)
            for key in excluded:
                excluded[key] += counts[key]
        groups = summarize(intervals, opts.by_file)
        if opts.csv:
            fields = ['task', 'cluster', 'hart', 'source', 'pid', 'count', 'avg_cycles',
                      'min_cycles', 'max_cycles']
            with opts.csv.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for g in groups:
                    writer.writerow({'task':g['task'], 'cluster':g['cluster'], 'hart':g['hart'],
                        'source':g['source'] if opts.by_file or g['cluster'] is None else '',
                        'pid':g['pid'] if g['cluster'] is None else '', 'count':g['count'],
                        'avg_cycles':mean_text(g['total'],g['count']),
                        'min_cycles':g['min'], 'max_cycles':g['max']})
        sys.stdout.write(render(intervals, groups, excluded, opts.by_file))
    except (OSError, ValueError) as error:
        print('dyperf-report: %s' % error, file=sys.stderr)
        return 2
    return 1 if opts.strict and (not intervals or any(excluded.values())) else 0


if __name__ == '__main__':
    sys.exit(main())
