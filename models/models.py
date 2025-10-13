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
        'indpended_attr':  [ 'last_adg', 'per_slobber_dm_log',  'avg_real_dm_inake_per_weight_per_day_log'] 
    },
    'naive_wg':
        {
            'depended_attr': 'pred_weight_gain', 
            'indpended_attr': ['originWeight', 'metabolic_weight_Limousin', 'metabolic_weight_Simental', 'metabolic_weight_Other', 'per_slobber_dm_log',  'avg_real_dm_inake_per_weight_per_day_log']  
        },
    'book_adg':
    {
        'depended_attr': 'pred_adgLatest_average', 
        'indpended_attr':   [ 'metabolic_weight','tdn_silage_over_mw_dt', 'tdn_rumput_over_mw_dt', 'tdn_slobber_over_mw_dt', 'day_diff_2']
    },
    'book_adg_1':
    {
        'depended_attr': 'pred_adgLatest_average', 
        'indpended_attr':   [ 'metabolic_weight', 'tdn_silage_over_mw_dt', 'tdn_rumput_over_mw_dt', 'tdn_slobber_over_mw_dt',  'day_diff']
    },
    'simental_book_adg_1':
    {
        'depended_attr': 'pred_adgLatest_average', 
        'indpended_attr':   [  'metabolic_weight', 'total_tdn_greens_over_mw_dt', 'tdn_slobber_over_mw_dt',  'day_diff']
    },
'simental_book_fcr_1':
    {
        'depended_attr': 'pred_adgLatest_average', 
        'indpended_attr':   [  'metabolic_weight', 'total_tdn_greens_over_mw_dt', 'tdn_slobber_over_mw_dt',  'day_diff']
    },
    'book_wg_1':
        {
            'depended_attr': 'pred_weight_gain', 
            'indpended_attr':   [ 'metabolic_weight_Limousin', 'metabolic_weight_Simental', 'tdn_silage_over_mw', 'tdn_rumput_over_mw', 'tdn_slobber_over_mw',  'day_diff']
        },
    'book_wg_2':
        {
            'depended_attr': 'pred_weight_gain', 
            'indpended_attr':   [ 'metabolic_weight_Limousin', 'metabolic_weight_Simental',  'tdn_slobber_over_mw', 'total_tdn_mw_2',  'day_diff_2']
        },
  'adg_trail':
    {
        'depended_attr': 'pred_adgLatest_average', 
        'indpended_attr':   [ 'last_adg', 'mw_diff', 'tdn_silage_over_mw_dt_diff', 'tdn_rumput_over_mw_dt_diff', 'tdn_slobber_over_mw_dt_diff',  'day_diff_2']
    },
}



#complex_models['first_order_models']['pred_adgLatest_average'] = []

