models = {
    'limousine_book_adg_1_final':
        {
            'pass':False,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_dt', 'tdn_rumput_dt', 'tdn_slobber_dt', 'day_diff_dmi']
        },
    'limousine_book_fcr_1_final':
        {
            'pass':False,
            'depended_attr': 'pred_fcrLatest_average', 
            'indpended_attr':   [ 'hasBEF', 'mw_ratio', 'tdn_silage_ddmi', 'tdn_rumput_ddmi', 'tdn_slobber_ddmi', 'day_diff_2' ]
        },
    'limousine_book_wg_1_final':
        {
            'pass':False,
            'depended_attr': 'pred_weight_gain', 
            'indpended_attr':   [ 'hasBEF_dmi', 'mw_ratio_dmi', 'tdn_silage', 'tdn_rumput', 'tdn_slobber', 'day_diff_2_dmi']
        },
    'limousine_book_adg_1_root_dmi':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_dt_r_dmi', 'tdn_rumput_dt_r_dmi', 'tdn_slobber_dt_r_dmi', 'day_diff_dmi']
        },
    'limousine_book_adg_1_final_mw':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_over_mw_dt', 'tdn_rumput_over_mw_dt', 'tdn_slobber_over_mw_dt', 'day_diff_dmi']
        },
    'limousine_book_adg_1_final_total_tdn_2_3':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_dt_2_3', 'tdn_rumput_dt_2_3', 'tdn_slobber_dt_2_3', 'day_diff_dmi']
        },
    'limousine_book_adg_1_adg_2':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average_2', 
            'indpended_attr':   [ 'hasBEF_dmi_dt_2', 'mw_dmi_dt_2', 'tdn_silage_dt_2', 'tdn_rumput_dt_2', 'tdn_slobber_dt_2', 'total_tdn_dt', 'mw_dmi_dt', 'day_diff_dmi']
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



OLS_models = {
        'limousine_book_adg_1_final':
            {
                'pass':False,
                'depended_attr': 'pred_adgLatest_average', 
                'indpended_attr':   [ 'startWeight', 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_dt', 'tdn_rumput_dt', 'tdn_slobber_dt', 'day_diff_dmi']
            },
        'limousine_book_fcr_1_final':
            {
                'pass':False,
                'depended_attr': 'pred_fcrLatest_average', 
                'indpended_attr':   [ 'startWeight', 'hasBEF', 'mw_ratio', 'tdn_silage_ddmi', 'tdn_rumput_ddmi', 'tdn_slobber_ddmi', 'day_diff_2' ]
            },
        'limousine_book_wg_1_final':
            {
                'pass':False,
                'depended_attr': 'pred_weight_gain', 
                'indpended_attr':   [ 'startWeight', 'hasBEF_dmi', 'mw_ratio_dmi', 'tdn_silage', 'tdn_rumput', 'tdn_slobber', 'day_diff_2_dmi']
            }
    } 
