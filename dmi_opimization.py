from consts.consts import pk_table, tdn_table, costs_per_dm

coefficients = {
    "hasBEF_dmi_dt": -0.005683826095069094,
    "mw_dmi_dt_ratio": -0.0005528394139441515,
    "tdn_silage_dt": 0.36215976616719464,
    "tdn_rumput_dt": 0.562520841247301,
    "tdn_slobber_dt": 0.2588482670072208,
    "day_diff_dmi": -0.00018165571176710093
}


start_weight = 350
current_weight = 400
silage_percentage = 0
rumput_percentage = 0.4
slobber_percentage = 0.6
day_diff = 10


metabolic_weight = current_weight**0.75
increase_ratio = current_weight/start_weight

tdn_per_silage = silage_percentage*tdn_table['silage']  
tdn_per_rumput = rumput_percentage*tdn_table['grass']  
tdn_per_slobber =  slobber_percentage*tdn_table['slobber']  

mw_attr = metabolic_weight*increase_ratio*coefficients['mw_dmi_dt_ratio']
silage_attr = tdn_per_silage*coefficients['tdn_silage_dt']
rumput_attr = tdn_per_rumput*coefficients['tdn_rumput_dt']
slobber_attr = tdn_per_slobber*coefficients['tdn_slobber_dt']
day_diff_attr = day_diff*coefficients['day_diff_dmi']


slope = mw_attr + silage_attr + rumput_attr + slobber_attr + day_diff_attr

print(f"slope: {slope}")

price_per_kg_dmi = 4000

price_per_kg_cattle = 55000

print(f"price_dm: {price_per_kg_dmi}, price cattle per kilo: {price_per_kg_cattle}")

price_per_adg_increase = price_per_kg_dmi/slope

increase = price_per_adg_increase < price_per_kg_cattle
profit = ((price_per_kg_cattle/price_per_adg_increase)-1)*100

print(f"price to add 1 kg to adg: {price_per_adg_increase}")
print(f"profit on increase: {profit}")
print(f"increase? {increase}")

