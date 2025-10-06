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
        'indpended_attr': ['originWeight', 'per_slobber_dm_log',  'avg_real_dm_inake_per_weight_per_day_log']  
    },
    'naive_wg':
        {
            'depended_attr': 'pred_weight_gain', 
            'indpended_attr': ['originWeight', 'per_slobber_dm_log',  'avg_real_dm_inake_per_weight_per_day_log']  
        },

}



#complex_models['first_order_models']['pred_adgLatest_average'] = []

