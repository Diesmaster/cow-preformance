models = {
    'limousine_book_adg_1':
        {
            'pass':False,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'mw_dmi_dt',  'tdn_silage_dt', 'tdn_rumput_dt', 'tdn_slobber_dt', 'day_diff_dmi']
        },
    'limousine_book_fcr_1':
        {
            'pass':False,
            'depended_attr': 'pred_fcrLatest_average', 
            'indpended_attr':   [ 'metabolic_weight', 'tdn_silage_ddmi', 'tdn_rumput_ddmi', 'tdn_slobber_ddmi', 'day_diff_2' ]
        },
    'limousine_book_wg_1':
        {
            'pass':False,
            'depended_attr': 'pred_weight_gain', 
            'indpended_attr':   [ 'mw_dmi', 'tdn_silage', 'tdn_rumput', 'tdn_slobber', 'day_diff_2_dmi' ]
        },
    'simental_book_adg_1':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [  'metabolic_weight', 'total_tdn_greens_over_mw_dt', 'tdn_slobber_over_mw_dt',  'day_diff']
        },
    'simental_book_fcr_1':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [  'metabolic_weight', 'total_tdn_greens_over_mw_dt', 'tdn_slobber_over_mw_dt',  'day_diff']
        },
  }



#complex_models['first_order_models']['pred_adgLatest_average'] = []

