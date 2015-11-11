BENCHMARKS = ['time_cpu']

# square of mutual entropy correlation coefficients (for RxC matrices)
ENTROPY_METRICS = [
    'homogeneity', 'completeness', 'nmi_score',
]

CONTINGENCY_METRICS = [
    'adjusted_mutual_info_score', 'talburt_wang_index',
    'split_join_similarity', 'mirkin_match_coeff'
]

PAIRWISE_METRICS = [
    # correlation triples
    'adjusted_rand_score', 'kappa1', 'kappa0',
    'mi_corr', 'mi_corr1', 'mi_corr0',
    'matthews_corr', 'informedness', 'markedness',
    'fscore', 'precision', 'recall',

    # other
    'rand_index', 'accuracy',
    'dice_coeff', 'jaccard_coeff', 'ochiai_coeff', 'sokal_sneath_coeff',
]

INCIDENCE_METRICS = PAIRWISE_METRICS + CONTINGENCY_METRICS + ENTROPY_METRICS

ROC_METRICS = ['roc_max_info', 'roc_auc']
LIFT_METRICS = ['aul_score']

RANKING_METRICS = ROC_METRICS + LIFT_METRICS

METRICS = RANKING_METRICS + INCIDENCE_METRICS + BENCHMARKS


def serialize_args(args):
    namespace = dict(args.__dict__)
    fields_to_delete = ["input", "output", "func", "logging"]
    for field in fields_to_delete:
        try:
            del namespace[field]
        except KeyError:
            pass
    return namespace


