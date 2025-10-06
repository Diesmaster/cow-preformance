models = {
    'weight_test_adg':
    {
        'depended_attr': 'pred_adgLatest_average', 
        'indpended_attr': ['weight'] 
    },
    'weight_dmi_adg':
    {
        'depended_attr': 'pred_adgLatest_average', 
        'indpended_attr': ['weight', 'avg_real_dm_inake_per_weight_per_day_c','avg_real_dm_inake_per_weight_per_day_c2'] 
    },
    'naive_adg':
    {
        'depended_attr': 'pred_adgLatest_average', 
        'indpended_attr': ['originWeight', 'daysOnFeedNow', 'metabolic_weight', 'per_slobber_dm_dmi_c',  'avg_real_dm_inake_per_weight_per_day_c',  'day_diff']  
    },
}



#complex_models['first_order_models']['pred_adgLatest_average'] = []

