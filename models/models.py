models = {
     'limousine_book_adg_1_dmi_analysis':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'weight', 'total_dmi', 'total_dmi_log', 'day_diff']
        },
      'limousine_book_fcr_1_dmi_analysis_linear':
        {
            'pass':True,
            'depended_attr': 'pred_fcrLatest_average', 
            'indpended_attr':   [ 'weight_ddmi', 'total_dmi']
        },       
       'limousine_book_fcr_1_dmi_analysis_derivative':
        {
            'pass':True,
            'depended_attr': 'pred_fcrLatest_average', 
            'indpended_attr':   [ 'weight_ddmi', '1_total_dmi']
        },        
        'limousine_book_adg_1_naive':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'weight', 'total_dmi', 'total_dmi_log', 'FeedRatio', 'day_diff']
        },   
        'limousine_book_adg_1_final':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'gotAppetiteBoost_dmi_dt', 'gotDewormed_dmi_dt', 'gotHNMVaccination_dmi_dt', 'mw_dmi_dt_ratio', 'tdn_silage_dt', 'tdn_rumput_dt', 'tdn_slobber_dt', 'tdn_SP2A_dt', 'day_diff_dmi']
        },
         'simental_book_adg_1_final':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [  'hasBEF_dmi_dt', 'gotAppetiteBoost_dmi_dt', 'gotDewormed_dmi_dt', 'gotHNMVaccination_dmi_dt', 'mw_dmi_dt_ratio','mw_dmi_dt_ratio_2', 'tdn_silage_dt', 'tdn_rumput_dt', 'tdn_slobber_dt', 'tdn_SP2A_dt', 'tdn_ricehay_dt', 'day_diff_dmi']
        },       
        'limousine_book_adg_1_final_log':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average_log', 
            'indpended_attr':   [ 'hasBEF_dmi_dt_log', 'mw_dmi_dt_ratio_log',  'tdn_silage_dt_log', 'tdn_rumput_dt_log', 'tdn_slobber_dt_log', 'total_tdn_dt_log', 'day_diff_dmi_log']
        },
    'limousine_book_fcr_1_final':
        {
            'pass':True,
            'depended_attr': 'pred_fcrLatest_average', 
            'indpended_attr':   [ 'hasBEF_dt', 'gotDewormed_dt',  'gotAppetiteBoost_dt', 'mw_ratio', 'tdn_silage_ddmi', 'tdn_rumput_ddmi', 'tdn_slobber_ddmi', 'FeedRatio_squared', 'day_diff_2' ]
        },
    'limousine_book_wg_1_final':
        {
            'pass':True,
            'depended_attr': 'pred_weight_gain', 
            'indpended_attr':   [ 'hasBEF_dmi', 'mw_ratio_dmi', 'tdn_silage', 'tdn_rumput', 'tdn_slobber', 'day_diff_2_dmi']
        },
    'limousine_book_adg_1_root_dmi':
        {
            'pass':False,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'gotAppetiteBoost_dmi_dt', 'gotDewormed_dmi_dt', 'gotHNMVaccination_dmi_dt', 'weight_ddmi', 'total_tdn_3_dt_dmi', 'daysOnFeedNow_r', 'mw_dmi_dt_ratio', 'tdn_silage_dt_r_dmi', 'tdn_concentrats_dt_r_dmi', 'tdn_rumput_log_r_dmi',  'day_diff_dmi']
        },
    'limousine_naive_2_adg_1':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'gotAppetiteBoost_dmi_dt', 'gotDewormed_dmi_dt', 'gotHNMVaccination_dmi_dt', 'total_tdn_1_dt_dmi','total_tdn_2_dt_dmi', 'total_tdn_3_dt_dmi', 'daysOnFeedNow_r', 'mw_dmi_dt_ratio', 'day_diff_dmi']
        },

    'limousine_book_adg_1_full_exp':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'gotAppetiteBoost_dmi_dt', 'gotDewormed_dmi_dt', 'gotHNMVaccination_dmi_dt', 'weight_ddmi', 'total_tdn_3_dt_dmi', 'daysOnFeedNow', 'mw_dmi_dt_ratio', 'tdn_silage_dt_r_dmi', 'tdn_slobber_dt_r_dmi', 'tdn_rumput_log', 'FeedRatio_squared', 'day_diff_dmi']
        },
    'limousine_book_adg_1_dmi_over_mw':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'gotAppetiteBoost_dmi_dt', 'gotDewormed_dmi_dt', 'gotHNMVaccination_dmi_dt', 'mw_dmi_dt_ratio', 'tdn_silage_dt_r_dmi_mw', 'tdn_slobber_dt_r_dmi_mw', 'tdn_SP2A_dt_r_dmi_mw', 'tdn_rumput_log', 'FeedRatio_squared', 'day_diff_dmi']
        },

    'simental_book_adg_1_root_dmi':
        {
            'pass':False,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_dt_r_dmi', 'tdn_rumput_dt_r_dmi_mw', 'tdn_slobber_dt_r_dmi_mw', 'tdn_SP2A_dt_r_dmi_mw', 'tdn_ricehay_dt_r_dmi_mw','day_diff_dmi']
        },
    'limousine_book_adg_1_final_mw':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_over_mw_dt', 'tdn_rumput_over_mw_dt', 'tdn_slobber_over_mw_dt', 'tdn_SP2A_over_mw_dt','tdn_ricehay_over_mw_dt','day_diff_dmi']
        },
    'simental_book_adg_1_final_mw':
        {
            'pass':True,
            'depended_attr': 'pred_adgLatest_average', 
            'indpended_attr':   [ 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_over_mw_dt', 'tdn_rumput_over_mw_dt', 'tdn_slobber_over_mw_dt', 'tdn_SP2A_over_mw_dt','tdn_ricehay_over_mw_dt','day_diff_dmi']
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
                'pass':True,
                'depended_attr': 'pred_adgLatest_average', 
                'indpended_attr':   [ 'startWeight', 'hasBEF_dmi_dt', 'mw_dmi_dt_ratio',  'tdn_silage_dt', 'tdn_rumput_dt', 'tdn_slobber_dt', 'day_diff_dmi']
            },
        'limousine_book_fcr_1_final':
            {
                'pass':True,
                'depended_attr': 'pred_fcrLatest_average', 
                'indpended_attr':   [ 'startWeight', 'hasBEF', 'mw_ratio', 'tdn_silage_ddmi', 'tdn_rumput_ddmi', 'tdn_slobber_ddmi', 'day_diff_2' ]
            },
        'limousine_book_wg_1_final':
            {
                'pass':True,
                'depended_attr': 'pred_weight_gain', 
                'indpended_attr':   [ 'startWeight', 'hasBEF_dmi', 'mw_ratio_dmi', 'tdn_silage', 'tdn_rumput', 'tdn_slobber', 'day_diff_2_dmi']
            }
    } 
