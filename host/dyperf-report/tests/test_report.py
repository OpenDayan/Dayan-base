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

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dyperf_report as report


def metadata(tid=6, name='softmax', pid=1):
    return {'ph':'M','name':'thread_name','pid':pid,'tid':tid,'args':{'name':name}}


def process(pid=1, cluster=0, hart=0):
    return {'ph':'M','name':'process_name','pid':pid,'tid':0,
            'args':{'name':'cluster %s hart %s' % (cluster,hart)}}


def state(counter, name, tid=6, pid=1, ph='B'):
    return {'ph':ph,'cat':'proc_state','name':name,'pid':pid,'tid':tid,
            'args':{'counter':counter}, 'ts':-999}


def trace(events):
    return {'traceEvents':events}


class ReportTests(unittest.TestCase):
    def test_waits_first_running_metadata_at_end_and_duplicate_exit(self):
        events=[state(90,'READY'),state(100,'RUNNING'),state(102,'WAIT_VEC'),
                state(107,'RUNNING'),state(110,'WAIT_EXIT'),state(111,'WAIT_EXIT'),
                metadata(),process()]
        rows,excluded=report.analyze_trace(trace(events),'test')
        self.assertEqual([r['cycles'] for r in rows],[10])
        self.assertFalse(any(excluded.values()))

    def test_mean_is_not_median_and_tid_reuse_is_separate(self):
        events=[process()]
        for tid,start,duration in [(6,0,10),(7,20,11),(8,40,15)]:
            events += [metadata(tid),state(start,'RUNNING',tid),state(start+duration,'WAIT_EXIT',tid)]
        rows,_=report.analyze_trace(trace(events),'test')
        group=report.summarize(rows)[0]
        self.assertEqual(group['count'],3)
        self.assertEqual(report.mean_text(group['total'],3),'12.000')
        self.assertEqual((group['min'],group['max']),(10,15))

    def test_wrap_while_another_track_is_running(self):
        events=[metadata(),metadata(7,'helper'),process(),
                state(report.PERIOD-5,'RUNNING'),state(report.PERIOD-2,'RUNNING',7),
                state(1,'RUNNING',7,ph='E'),state(4,'WAIT_EXIT'),state(5,'WAIT_EXIT',7)]
        rows,_=report.analyze_trace(trace(events),'wrap')
        self.assertEqual([r['cycles'] for r in rows],[9,7])

    def test_services_and_pcu_are_not_tasks_or_timing_inputs(self):
        events=[process(),metadata(),state(1,'RUNNING'),
                {'cat':'pcu','ph':'i','args':{'cycle':99999999}},state(11,'WAIT_EXIT')]
        for i,name in enumerate(sorted(report.SERVICES),20):
            events += [metadata(i,name),state(i*10,'RUNNING',i),state(i*10+1,'WAIT_EXIT',i)]
        rows,_=report.analyze_trace(trace(events),'test')
        self.assertEqual([r['cycles'] for r in rows],[10])

    def test_incomplete_missing_start_and_not_started(self):
        events=[process(),metadata(),metadata(7,'missing'),metadata(8,'queued'),
                state(1,'RUNNING'),state(2,'WAIT_EXIT',7),state(3,'READY',8)]
        rows,counts=report.analyze_trace(trace(events),'partial')
        self.assertFalse(rows)
        self.assertEqual(counts,dict(incomplete=1,missing_start=1,not_started=1,missing_name=0))

    def test_multicore_does_not_merge_and_unknown_cores_stay_per_file(self):
        rows=[]
        for pid,hart in [(1,0),(2,1)]:
            one,_=report.analyze_trace(trace([process(pid,0,hart),metadata(pid=pid),
                state(1,'RUNNING',pid=pid),state(11,'WAIT_EXIT',pid=pid)]),'a')
            rows+=one
        self.assertEqual(len(report.summarize(rows)),2)
        one,_=report.analyze_trace(trace([metadata(),state(1,'RUNNING'),state(11,'WAIT_EXIT')]),'a')
        two,_=report.analyze_trace(trace([metadata(),state(1,'RUNNING'),state(11,'WAIT_EXIT')]),'b')
        self.assertEqual(len(report.summarize(one+two)),2)

    def test_multiple_files_merge_or_stay_separate(self):
        data=trace([process(),metadata(),state(1,'RUNNING'),state(11,'WAIT_EXIT')])
        a,_=report.analyze_trace(data,'a')
        b,_=report.analyze_trace(data,'b')
        self.assertEqual(report.summarize(a+b)[0]['count'],2)
        self.assertEqual(len(report.summarize(a+b,True)),2)

    def test_reordered_and_invalid_counters_rejected(self):
        for value in [True,-1,1.5,report.PERIOD]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                report.analyze_trace(trace([metadata(),state(value,'RUNNING')]),'bad')
        with self.assertRaisesRegex(ValueError,'out of order'):
            report.analyze_trace(trace([metadata(),state(10,'RUNNING'),state(9,'WAIT_EXIT')]),'bad')

    def test_escaped_task_name_and_unknown_thread(self):
        rows,counts=report.analyze_trace(trace([process(),metadata(name='task\n"中文'),
            state(1,'RUNNING'),state(11,'WAIT_EXIT'),state(20,'RUNNING',7)]),'test')
        output=report.render(rows,report.summarize(rows),counts)
        self.assertIn('task="task\\n\\"中文"',output)
        self.assertEqual(counts['missing_name'],1)

    def test_empty_and_large_integer_mean(self):
        rows,counts=report.analyze_trace(trace([]),'empty')
        self.assertIn('no completed task intervals',report.render(rows,[],counts))
        self.assertEqual(report.mean_text(3*(1<<60)+1,3),'1152921504606846976.333')

    def test_cli_csv_strict_and_duplicate_or_overwrite_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.json'
            path.write_text(json.dumps(trace([process(),metadata(),state(1,'RUNNING'),
                                             state(11,'WAIT_EXIT')])))
            out=Path(tmp)/'summary.csv'
            with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(report.main([str(path),'--csv',str(out),'--strict']),0)
                self.assertEqual(report.main([str(path),str(path)]),2)
                self.assertEqual(report.main([str(path),'--csv',str(path)]),2)
                self.assertEqual(report.main([str(path),'--task','absent','--strict']),1)
            with out.open() as f:
                row=next(csv.DictReader(f))
            self.assertEqual(row['avg_cycles'],'10.000')
            self.assertEqual(row['count'],'1')
            self.assertIn('traceEvents',json.loads(path.read_text()))

    def test_invalid_trace_schema_and_json(self):
        for data in [[],{},trace([None]),trace([metadata(),dict(state(1,'RUNNING'),args=None)])]:
            with self.assertRaises(ValueError): report.analyze_trace(data,'bad')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'
            path.write_text('{')
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(report.main([str(path)]),2)


if __name__ == '__main__':
    unittest.main()
