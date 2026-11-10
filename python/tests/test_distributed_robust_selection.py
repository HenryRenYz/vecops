# SPDX-FileCopyrightText: 2026 HenryRenYz and vecops contributors
# SPDX-License-Identifier: MIT
"""Bounded route selection regressions, including retained measured outliers.
"""
import copy
import math
import statistics
import unittest

from vecops.distributed.tuning import (
    _select_route_with_confirmation as select, _rotated_route_indices as rotate)
# Existing actual initial samples; any confirmation below is explicitly synthetic.
REAL = {'one_mib': [[0.006623912, 0.006423574, 0.0064886695, 0.006757639, 0.0067169235], [0.007207391, 0.004168864, 0.0044873295, 0.0041125785, 0.004280346]], 'second_round': [[0.0063643745, 0.0065968385, 0.0063985355, 0.006556063, 0.0063356895], [0.0043299045, 0.011066605, 0.004201233, 0.0042701585, 0.004087166]]}


class ConfirmationRequested(Exception):pass


def run(records, confirmation=None, minimum=.05):
    values=copy.deepcopy(records);calls=[]
    def sample(round_index,indices):
        calls.append((round_index,indices))
        if confirmation is None:raise ConfirmationRequested()
        index=len(calls)-1
        values[0].append(confirmation[0][index]);values[indices[1]].append(confirmation[1][index])
    selected,report=select(values,minimum,sample)
    return selected,report,values,calls


class StatisticsTests(unittest.TestCase):
    def test_real_1m_outlier_requires_measurement_not_direct_selection(self):
        with self.assertRaises(ConfirmationRequested):run(REAL['one_mib'])

    def test_real_second_round_outlier_also_requires_measurement(self):
        with self.assertRaises(ConfirmationRequested):run(REAL['second_round'])

    def test_real_1m_plus_synthetic_stable_confirmation(self):
        rows=REAL['one_mib'];extra=([statistics.median(rows[0])]*5,[statistics.median(rows[1])]*5)
        chosen,report,values,calls=run(rows,extra)
        self.assertEqual(chosen,1);self.assertEqual(len(calls),5)
        self.assertEqual(report['combined_statistics']['wins'],9)
        self.assertEqual(values[1][:5],rows[1])
        self.assertEqual(report['candidate_round_counts'],[10,10])
        self.assertGreater(report['noise_fraction'],.5)

    def test_real_second_outlier_plus_synthetic_stable_confirmation(self):
        rows=REAL['second_round'];extra=([statistics.median(rows[0])]*5,[statistics.median(rows[1])]*5)
        chosen,report,values,calls=run(rows,extra)
        self.assertEqual(chosen,1);self.assertEqual(len(calls),5)
        self.assertEqual(report['combined_statistics']['wins'],9)
        self.assertEqual(values[1][1],rows[1][1])

    def test_bad_confirmation_keeps_flat(self):
        chosen,report,_,calls=run(REAL['one_mib'],([.0066]*5,[.0043,.0075,.0042,.0065,.0043]))
        self.assertEqual(chosen,0);self.assertEqual(len(calls),5)
        self.assertEqual(report['reason'],'confirmation_inconclusive_keep_baseline')

    def test_five_unanimous_clear_rounds_no_extra(self):
        chosen,report,_,calls=run([[10]*5,[6]*5])
        self.assertEqual(chosen,1);self.assertEqual(calls,[])
        self.assertEqual(report['confirmation_rounds'],0)

    def test_three_rounds_need_bounded_five(self):
        chosen,report,_,calls=run([[10]*3,[6]*3],([10]*5,[6]*5))
        self.assertEqual(chosen,1);self.assertEqual(len(calls),5)
        self.assertEqual([r for r,_ in calls],list(range(3,8)))
        self.assertEqual(report['combined_statistics']['rounds'],8)

    def test_near_threshold_below_does_not_select(self):
        chosen,_,_,calls=run([[10]*5,[9.6,9.55,9.58,9.59,9.54]])
        self.assertEqual(chosen,0);self.assertEqual(calls,[])

    def test_noisy_overlapping_does_not_select(self):
        chosen,_,_,calls=run([[10]*5,[7,12,9,11,8]])
        self.assertEqual(chosen,0);self.assertEqual(calls,[])

    def test_bimodal_three_fast_two_slow_madzero_rejected(self):
        chosen,report,_,calls=run([[10]*5,[6,6,6,14,14]])
        self.assertEqual(chosen,0);self.assertEqual(calls,[])
        self.assertEqual(report['initial_statistics']['paired_gain_mad'],0)

    def test_disastrous_tail_remains_in_mean_even_after_good_confirmation(self):
        chosen,report,values,calls=run([[10]*5,[6,6,6,6,100]],([10]*5,[6]*5))
        self.assertEqual(chosen,0);self.assertEqual(len(calls),5)
        self.assertLess(report['combined_statistics']['mean_time_gain'],0)
        self.assertIn(100,values[1])

    def test_common_round_drift_uses_paired_evidence(self):
        chosen,report,_,calls=run([[7,15,8,16,10],[4.2,9,4.8,9.6,6]])
        self.assertEqual(chosen,1);self.assertEqual(calls,[])
        self.assertGreater(report['noise_fraction'],.8)

    def test_multiple_candidates_fixed_best_no_runnerup_rescue(self):
        records=[[10]*5,[6,6,6,6,10.1],[7]*5]
        chosen,report,values,calls=run(records,([10]*5,[11]*5))
        self.assertEqual(chosen,0);self.assertEqual(report['fixed_best_index'],1)
        self.assertEqual(values[2],[7]*5)
        self.assertTrue(all(indices==(0,1)for _,indices in calls));self.assertEqual(len(calls),5)

    def test_baseline_best_no_confirmation(self):
        chosen,report,_,calls=run([[6]*5,[10]*5])
        self.assertEqual(chosen,0);self.assertEqual(calls,[])

    def test_minimum_gain_honored(self):
        chosen,_,_,calls=run([[10]*5,[8.5]*5],minimum=.2)
        self.assertEqual(chosen,0);self.assertEqual(calls,[])
        self.assertEqual(run([[10]*5,[8.5]*5],minimum=.1)[0],1)

    def test_invalid_timing_samples_fail_closed(self):
        for records in ([[1]*5,[1,1,0,1,1]],[[1]*5,[1,1,math.nan,1,1]],[[1]*5,[1]*4]):
            with self.subTest(records=records),self.assertRaises(ValueError):run(records)

    def test_sampler_cannot_omit_or_add_extra_samples(self):
        with self.assertRaises(ValueError):select(copy.deepcopy(REAL['one_mib']),.05,lambda *args:None)

    def test_rotation_preserves_full_and_pair_order(self):
        self.assertEqual(rotate((0,1,2),1),[1,2,0])
        self.assertEqual(rotate((0,2),5),[2,0])
        self.assertEqual(rotate((0,2),6),[0,2])


class LegacyRegressionTests(unittest.TestCase):
    def test_single_real_outlier_gets_confirmation_not_permanent_flat(self):
        rows=REAL['one_mib']
        chosen,_,_,calls=run(rows,([statistics.median(rows[0])]*5,[statistics.median(rows[1])]*5))
        self.assertEqual(len(calls),5)
        self.assertEqual(chosen,1)

    def test_second_round_real_outlier_gets_confirmation_not_permanent_flat(self):
        rows=REAL['second_round']
        chosen,_,_,calls=run(rows,([statistics.median(rows[0])]*5,[statistics.median(rows[1])]*5))
        self.assertEqual(len(calls),5)
        self.assertEqual(chosen,1)

    def test_round_drift_cannot_alone_veto_unanimous_paired_gain(self):
        self.assertEqual(run([[7,15,8,16,10],[4.2,9,4.8,9.6,6]])[0],1)

    def test_stable_clear_still_selected(self):
        self.assertEqual(run([[10]*5,[6]*5])[0],1)

    def test_overlap_noise_stays_baseline(self):
        self.assertEqual(run([[10]*5,[7,12,9,11,8]])[0],0)

    def test_catastrophic_tail_never_selected(self):
        self.assertEqual(run([[10]*5,[6,6,6,6,100]],([10]*5,[6]*5))[0],0)
