"""Fixed ten-trained / nine-eligible prevalence; no model or dataset access."""
import math
Z95 = 1.959963984540054

def wilson95(supports, total):
    if type(supports) is not int or type(total) is not int or total <= 0 or not 0 <= supports <= total:
        raise ValueError('Wilson requires integer counts 0 <= supports <= positive total')
    p = supports / total
    z2 = Z95 * Z95
    denominator = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denominator
    radius = Z95 * math.sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]

TRAINED = [291410,291411,291412,291413,291414,291415,291416]
ELIGIBLE = [291410,291411,291412,291413,291415,291416]
ORIGINAL = [291405,291406,291407]

def validate_rule(rule):
    counts={'original_supports':2,'original_total':3,'new_trained_total':7,'new_baseline_eligible':6,'new_baseline_ineligible':1,'fixed_total_trained_models':10,'baseline_eligible_total':9}
    for key,value in counts.items():
        if type(rule.get(key)) is not int or rule[key]!=value:raise ValueError('FROZEN_COUNT_CHANGED:'+key)
    if rule.get('new_trained_seed_ids')!=TRAINED or rule.get('mechanism_evaluation_seed_ids')!=ELIGIBLE or rule.get('baseline_ineligible_seed_ids')!=[291414]:raise ValueError('FIXED_SEED_COHORT_CHANGED')
    if rule.get('binary_success_threshold') is not None:raise ValueError('BINARY_THRESHOLD_FORBIDDEN')
    for key in ['replacement_allowed','additional_seed_allowed','health_threshold_relaxation_allowed','ineligible_mechanism_evaluation_allowed','outcome_dependent_stopping_allowed','new_primary_p_value_allowed']:
        if rule.get(key) is not False:raise ValueError('FROZEN_SCOPE_POLICY_CHANGED:'+key)

def validate_scope(rule,scope):
    validate_rule(rule)
    if scope.get('fixed_training_sample_seed_ids')!=ORIGINAL+TRAINED:raise ValueError('FIXED_TEN_TRAINING_DENOMINATOR_CHANGED')
    if scope.get('baseline_eligible_seed_ids')!=ORIGINAL+ELIGIBLE:raise ValueError('FIXED_NINE_ELIGIBLE_DENOMINATOR_CHANGED')
    if scope.get('mechanism_evaluation_seed_ids')!=ELIGIBLE:raise ValueError('SIX_ELIGIBLE_MECHANISM_COHORT_REQUIRED')
    retained=scope.get('baseline_ineligible_records')
    if not isinstance(retained,list) or len(retained)!=1:raise ValueError('INELIGIBLE_RECORD_MUST_BE_RETAINED')
    row=retained[0]
    if row.get('seed')!=291414 or row.get('status')!='BASELINE_INELIGIBLE':raise ValueError('INELIGIBLE_CLASSIFICATION_CHANGED')
    if row.get('mechanism_evaluation_allowed') is not False or row.get('mechanism_support') is not None or row.get('mechanism_status')!='NOT_EVALUATED_BASELINE_INELIGIBLE':raise ValueError('INELIGIBLE_MUST_NOT_BE_MECHANISM_CLASSIFIED')
    if row.get('end_to_end_support') is not False or row.get('included_in_training_denominator') is not True:raise ValueError('INELIGIBLE_DENOMINATOR_CHANGED')
    if scope.get('retired_seed_ids')!=[291408,291409]:raise ValueError('RETIRED_COHORT_CHANGED')
    return row

def compute_prevalence(s6,rule):
    validate_rule(rule)
    if type(s6) is not int or not 0<=s6<=6:raise ValueError('S6_MUST_BE_INTEGER_0_TO_6')
    numerator=2+s6
    return dict(S6=dict(support_count=s6,total=6,rate=s6/6,role='DESCRIPTIVE_NEW_ELIGIBLE_COHORT'),
        B10=dict(eligible_count=9,total_trained=10,rate=9/10),
        E10=dict(support_count=numerator,total_trained=10,rate=numerator/10,wilson95=wilson95(numerator,10),role='PRIMARY_EXPANDED_PREVALENCE',label='END_TO_END_SUPPORT_ACROSS_FIXED_TRAINING_SAMPLE'),
        M9=dict(support_count=numerator,total_eligible=9,rate=numerator/9,wilson95=wilson95(numerator,9),role='CONDITIONAL_MECHANISM_PREVALENCE',label='MECHANISM_SUPPORT_CONDITIONAL_ON_BASELINE_ELIGIBILITY'),
        binary_success_threshold=None,new_primary_p_value=None,qualitative_band=None,original_primary_support_count=2,original_primary_total=3,original_primary_reclassified=False)

def summarize_support(checkpoint_decisions,rule,scope):
    retained=validate_scope(rule,scope)
    if not isinstance(checkpoint_decisions,dict) or set(checkpoint_decisions)!={str(s) for s in ELIGIBLE} or len(checkpoint_decisions)!=6:raise ValueError('EXACT_SIX_ELIGIBLE_DECISIONS_REQUIRED')
    if any(type(v) is not bool for v in checkpoint_decisions.values()):raise ValueError('ACTUAL_BOOLEAN_SUPPORT_DECISIONS_REQUIRED')
    return dict(checkpoint_supports=dict(checkpoint_decisions),retained_baseline_ineligible=dict(retained),**compute_prevalence(sum(checkpoint_decisions.values()),rule))
