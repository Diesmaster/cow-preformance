from datetime import datetime
from consts.consts import pk_table, tdn_table, costs_per_dm
import numpy as np

class FeedProcessor:
    """
    Processes feed and DMI (Dry Matter Intake) data for cattle.
    Calculates various feed-related metrics and nutritional values.
    """
    
    def __init__(self, feed_history, weight_history, start_index, n_weighing):
        """
        Initialize the ProcessFeed object.
        
        Args:
            feed_history: FeedHistoryData object
            weight_history: WeightHistoryData object
            start_index (int): Starting index in weight history
            n_weighing (int): Number of weighings to process
        """
        self.feed_history = feed_history
        self.weight_history = weight_history
        self.start_index = start_index
        self.n_weighing = n_weighing
        
        self.start_date = weight_history.data[start_index]['date']
        self.end_date = weight_history.data[start_index + n_weighing]['date']
        
        self.day_diff = (datetime.strptime(self.end_date, "%Y-%m-%d") - 
                        datetime.strptime(self.start_date, "%Y-%m-%d")).days
        
        # Calculate all metrics
        self._calculate_dmi_metrics()
        self._calculate_feed_composition()
    
    def _calculate_dmi_metrics(self):
        """Calculate DMI-related metrics."""
        self.total_dm_intake = self.feed_history.get_dry_matter_intake(
            self.start_date, self.end_date
        )
        
        avg_dm_intake_per_day = 0
        avg_real_dm_inake_per_weight_per_day = 0
        avg_real_dm_inake_per_mw_per_day = 0
        avg_weight = 0
        
        for i in range(self.n_weighing):
            index = self.start_index + i
            
            day_diff_segment = (
                datetime.strptime(self.weight_history.data[index + 1]['date'], "%Y-%m-%d") - 
                datetime.strptime(self.weight_history.data[index]['date'], "%Y-%m-%d")
            ).days
            
            day_diff_fraction = day_diff_segment / self.day_diff
            
            dm_intake = self.feed_history.get_dry_matter_intake(
                self.weight_history.data[index]['date'],
                self.weight_history.data[index + 1]['date']
            )
            
            weight_at_index = self.weight_history.data[index]['weight']
            dm_intake_per_weight = (dm_intake / weight_at_index) * 100
            dm_intake_per_mw = (dm_intake / (weight_at_index**0.75)) * 100
            
            avg_dm_intake_per_day += dm_intake / day_diff_segment
            avg_real_dm_inake_per_weight_per_day += (dm_intake_per_weight / day_diff_segment) * day_diff_fraction
            avg_real_dm_inake_per_mw_per_day += (dm_intake_per_mw / day_diff_segment) * day_diff_fraction
            avg_weight += weight_at_index * day_diff_fraction
        
        self.avg_dm_intake_per_day = avg_dm_intake_per_day
        self.avg_real_dm_inake_per_weight_per_day = avg_real_dm_inake_per_weight_per_day
        self.avg_real_dm_inake_per_mw_per_day = avg_real_dm_inake_per_mw_per_day
        self.avg_weight = avg_weight
        self.avg_mw = avg_weight**0.75
        
        # Calculate squared and log transformations for DMI metrics
        self.avg_dm_intake_per_day_squared = avg_dm_intake_per_day ** 2
        self.avg_real_dm_inake_per_weight_per_day_squared = avg_real_dm_inake_per_weight_per_day ** 2
        self.avg_real_dm_inake_per_mw_per_day_squared = avg_real_dm_inake_per_mw_per_day ** 2
        
        # Log transformations (handle zero/negative values)
        self.avg_dm_intake_per_day_log = np.log(avg_dm_intake_per_day) if avg_dm_intake_per_day > 0 else np.nan
        self.avg_real_dm_inake_per_weight_per_day_log = np.log(avg_real_dm_inake_per_weight_per_day) if avg_real_dm_inake_per_weight_per_day > 0 else np.nan
        self.avg_real_dm_inake_per_mw_per_day_log = np.log(avg_real_dm_inake_per_mw_per_day) if avg_real_dm_inake_per_mw_per_day > 0 else np.nan
    
    def _calculate_feed_composition(self):
        """Calculate feed composition metrics."""
        ration = self.feed_history.get_diet(self.start_date, self.end_date)
        
        # Check if we have the required feed types
        self.has_required_feeds = (
            ('Silase Jagung' in ration or 'Rumput' in ration or 'Pakchong Grass - Qurban') and 
            ('Slobber Mix' in ration or 'SP2B Mix' in ration or 'SP2A Mix' in ration)
        )
       
        if not self.has_required_feeds:
            return
       
        #Rice Hay

        # Calculate feed dry matter amounts
        silage_dm = ration.get('Silase Jagung', {}).get('asFedIntakePerCow', 0) * 0.3
        grass_dm = ration.get('Rumput', ration.get('Pakchong Grass - Qurban', {})).get('asFedIntakePerCow', 0) * 0.2


        if grass_dm < 0:
            print(f"{grass_dm=}")
            print(f"{ration=}")

        self.rice_hay_dm = 0 
        if 'Rice Hay' in ration:
            self.rice_hay_dm = ration['Rice Hay']['asFedIntakePerCow'] * 0.85

        self.slobber_dm = 0 
        if 'Slobber Mix' in ration:
            self.slobber_dm  = ration['Slobber Mix']['asFedIntakePerCow'] * 0.4506

        self.SP2B_mix_dm = 0
        if 'SP2B Mix' in ration:
            self.SP2B_mix_dm = ration['SP2B Mix']['asFedIntakePerCow'] * 0.8623 

        self.SP2A_mix_dm = 0   
        if 'SP2A Mix' in ration:
            self.SP2A_mix_dm = ration['SP2A Mix']['asFedIntakePerCow'] * 0.8623
        
        self.dm_total = silage_dm + grass_dm + self.slobber_dm + self.SP2B_mix_dm + self.SP2A_mix_dm
        self.dm_total_dt = self.dm_total/self.day_diff
        

        self.silage_dm = silage_dm
        self.grass_dm = grass_dm
        self.green_dm = silage_dm + grass_dm + self.rice_hay_dm
        
        # TDN calculations
        self.tdn_silage = silage_dm * tdn_table['silage']
        self.tdn_ricehay = self.rice_hay_dm * tdn_table['Rice Hay']
        self.tdn_rumput = grass_dm * tdn_table['grass']
        self.tdn_slobber = self.slobber_dm * tdn_table['slobber']
        self.tdn_SP2B_mix = self.SP2B_mix_dm * tdn_table['SP2B mix']
        self.tdn_SP2A_mix = self.SP2A_mix_dm * tdn_table['SP2A mix']
        self.tdn_new_concentrat = self.tdn_SP2B_mix + self.tdn_SP2A_mix 


        self.pk_silage = silage_dm *pk_table['silage']
        self.pk_rumput = grass_dm * pk_table['grass']
        self.pk_ricehay = self.rice_hay_dm * pk_table['Rice Hay']
        self.pk_slobber = self.slobber_dm *pk_table['slobber']
        self.pk_sp2b = self.SP2B_mix_dm *pk_table['SP2B mix']
        self.pk_sp2a = self.SP2A_mix_dm *pk_table['SP2A mix']


        self.pk_total = self.pk_silage + self.pk_rumput + self.pk_slobber + self.pk_sp2a + self.pk_sp2b + self.pk_ricehay

        self.pk_per = self.pk_total / self.dm_total

        self.total_tdn = self.tdn_SP2A_mix + self.tdn_SP2B_mix + self.tdn_slobber + self.tdn_silage + self.tdn_rumput + self.tdn_ricehay
       
        self.per_slobber_tdn = self.tdn_slobber/self.total_tdn 
        self.per_rumput_tdn = self.tdn_rumput/self.total_tdn 
        self.per_silage_tdn = self.tdn_silage/self.total_tdn 

        self.not_greens = self.slobber_dm + self.SP2A_mix_dm + self.SP2B_mix_dm

        # Feed ratios and percentages
        self.feed_ratio = self.green_dm / self.not_greens if self.not_greens > 0 else 0
        self.per_slobber_dm = self.not_greens / self.total_dm_intake if self.total_dm_intake > 0 else 0
        self.per_green_dm = self.green_dm / self.total_dm_intake if self.total_dm_intake > 0 else 0
        
        # Cost calculations
        if self.green_dm > 0:
            self.greens_cost_per_kg = (
                (silage_dm * costs_per_dm['silage'] + grass_dm * costs_per_dm['grass']) / 
                self.green_dm
            )
        else:
            self.greens_cost_per_kg = costs_per_dm['silage']
        
        self.greens_cost = silage_dm * costs_per_dm['silage'] + grass_dm * costs_per_dm['grass']
        self.slobber_cost = self.slobber_dm * costs_per_dm['slobber']
        self.feed_cost = self.greens_cost + self.slobber_cost
        self.feed_cost_per_dm = self.feed_cost / self.total_dm_intake if self.total_dm_intake > 0 else 0
        
        # TDN per day and metabolic weight
        self.total_tdn_dt = self.total_tdn / self.day_diff
        self.total_tdn_mw_dt = self.total_tdn / self.day_diff / self.avg_mw
        self.total_tdn_mw = self.total_tdn /  self.avg_mw
       
        self.tdn_silage_dt = self.tdn_silage / self.day_diff
        self.tdn_rumput_dt = self.tdn_rumput / self.day_diff
        self.tdn_slobber_dt = self.tdn_slobber / self.day_diff
        self.tdn_new_concentrat_dt = self.tdn_new_concentrat / self.day_diff
        self.tdn_sp2a_dt = self.tdn_SP2A_mix / self.day_diff
        self.tdn_ricehay_dt = self.tdn_ricehay / self.day_diff
        self.tdn_sp2b_dt = self.tdn_SP2B_mix / self.day_diff

        self.tdn_concentrats_dt = self.tdn_sp2a_dt + self.tdn_slobber_dt 

        self.r_tdn_silage_dt = (self.tdn_silage / self.day_diff)**(1/2)
        self.r_tdn_rumput_dt = (self.tdn_rumput / self.day_diff)**(1/2)
        self.r_tdn_slobber_dt = (self.tdn_slobber / self.day_diff)**(1/2)
        self.r_tdn_new_concentrat_dt = (self.tdn_new_concentrat / self.day_diff)**(1/2)
        self.r_tdn_sp2a_dt = (self.tdn_SP2A_mix / self.day_diff)**(1/2)
        self.r_tdn_ricehay_dt = (self.tdn_ricehay / self.day_diff)**(1/2)
        self.r_tdn_sp2b_dt = (self.tdn_SP2B_mix / self.day_diff)**(1/2)

        
        self.total_tdn_greens = self.tdn_silage + self.tdn_rumput

        self.total_tdn_greens_over_mw = (self.total_tdn_greens / self.avg_mw)
        self.total_tdn_greens_over_mw_dt = (self.total_tdn_greens / self.avg_mw) / self.day_diff


        self.tdn_silage_over_mw_dt = (self.tdn_silage / self.avg_mw) / self.day_diff
        self.tdn_rumput_over_mw_dt = (self.tdn_rumput / self.avg_mw) / self.day_diff
        self.tdn_slobber_over_mw_dt = (self.tdn_slobber / self.avg_mw) / self.day_diff
        self.tdn_sp2a_over_mw_dt = (self.tdn_SP2A_mix / self.avg_mw) / self.day_diff
        self.tdn_ricehay_over_mw_dt = (self.tdn_ricehay / self.avg_mw) / self.day_diff
        
        self.tdn_silage_over_mw = (self.tdn_silage / self.avg_mw)
        self.tdn_rumput_over_mw = (self.tdn_rumput / self.avg_mw) 
        self.tdn_slobber_over_mw = (self.tdn_slobber / self.avg_mw) 

        self.tdn_silage_dt = (self.tdn_silage ) / self.day_diff
        self.tdn_rumput_dt = (self.tdn_rumput ) / self.day_diff
        self.tdn_slobber_dt = (self.tdn_slobber ) / self.day_diff
        
        self.total_tdn_1_dt_dmi = (
            ((self.total_tdn_dt )) * 
            self.avg_real_dm_inake_per_weight_per_day
        )

        self.total_tdn_2_dt_dmi = (
            ((self.total_tdn_dt )**2) * 
            self.avg_real_dm_inake_per_weight_per_day
        )
        self.total_tdn_3_dt_dmi = (
            ((self.total_tdn_dt )**3) * 
            self.avg_real_dm_inake_per_weight_per_day
        )
       
        # Calculate squared transformations for key feed composition metrics
        self.silage_dm_squared = silage_dm ** 2
        self.grass_dm_squared = grass_dm ** 2
        self.slobber_dm_squared = self.slobber_dm ** 2
        self.green_dm_squared = self.green_dm ** 2
        self.total_tdn_squared = self.total_tdn ** 2
        self.feed_ratio_squared = self.feed_ratio ** 2
        self.per_slobber_dm_squared = self.per_slobber_dm ** 2
        self.per_green_dm_squared = self.per_green_dm ** 2
        self.feed_cost_squared = self.feed_cost ** 2
        self.feed_cost_per_dm_squared = self.feed_cost_per_dm ** 2
        self.total_tdn_dt_squared = self.total_tdn_dt ** 2
        self.total_tdn_3_dt = self.total_tdn_mw_dt ** 3
        self.total_tdn_2 = self.total_tdn_mw ** 2
        self.total_tdn_3 = self.total_tdn_mw ** 3
        self.total_tdn_mw_dt_squared = self.total_tdn_mw_dt ** 2
       
        # times dmi
        self.per_slobber_dm_dmi = (self.per_slobber_dm)*self.avg_real_dm_inake_per_weight_per_day

        # Calculate log transformations for key feed composition metrics
        self.silage_dm_log = np.log(silage_dm) if silage_dm > 0 else np.nan
        self.grass_dm_log = np.log(grass_dm) if grass_dm > 0 else np.nan
        self.slobber_dm_log = np.log(self.slobber_dm) if self.slobber_dm > 0 else np.nan
        self.green_dm_log = np.log(self.green_dm) if self.green_dm > 0 else np.nan
        self.total_tdn_log = np.log(self.total_tdn) if self.total_tdn > 0 else np.nan
        self.feed_ratio_log = np.log(self.feed_ratio) if self.feed_ratio > 0 else np.nan
        self.per_slobber_dm_log = np.log(self.per_slobber_dm) if self.per_slobber_dm > 0 else np.nan
        self.per_green_dm_log = np.log(self.per_green_dm) if self.per_green_dm > 0 else np.nan
        self.feed_cost_log = np.log(self.feed_cost) if self.feed_cost > 0 else np.nan
        self.feed_cost_per_dm_log = np.log(self.feed_cost_per_dm) if self.feed_cost_per_dm > 0 else np.nan
        self.total_tdn_dt_log = np.log(self.total_tdn_dt) if self.total_tdn_dt > 0 else np.nan
        self.total_tdn_mw_dt_log = np.log(self.total_tdn_mw_dt) if self.total_tdn_mw_dt > 0 else np.nan
        self.tdn_silage_log = np.log(self.tdn_silage) if self.tdn_silage > 0 else np.nan
        self.tdn_rumput_log = np.log(self.tdn_rumput) if self.tdn_rumput > 0 else 0 
        self.tdn_rumput_log = np.log(self.tdn_rumput) if self.tdn_rumput > 0 else 0 
        self.over_tdn_rumput = 1/(self.tdn_rumput) if self.tdn_rumput > 0 else 0 
        self.tdn_slobber_log = np.log(self.tdn_slobber) if self.tdn_slobber > 0 else np.nan
        self.tdn_silage_dt_log = np.log(self.tdn_silage_dt) if self.tdn_silage > 0 else 0 
        self.tdn_rumput_dt_log = np.log(self.tdn_rumput_dt) if self.tdn_rumput > 0 else 0 
        self.tdn_slobber_dt_log = np.log(self.tdn_slobber_dt) if self.tdn_slobber > 0 else 0  
    

    
    def get_dmi_features(self):
        """
        Get DMI-related features as a dictionary.
        
        Returns:
            dict: DMI features including squared and log transformations
        """
        return {
            'total_dm_intake': self.total_dm_intake,
            'avg_dm_intake_per_day': self.avg_dm_intake_per_day,
            'avg_dm_intake_per_day_squared': self.avg_dm_intake_per_day_squared,
            'avg_dm_intake_per_day_log': self.avg_dm_intake_per_day_log,
            'avg_real_dm_inake_per_weight_per_day': self.avg_real_dm_inake_per_weight_per_day,
            'avg_real_dm_inake_per_weight_per_day_squared': self.avg_real_dm_inake_per_weight_per_day_squared,
            'avg_real_dm_inake_per_weight_per_day_log': self.avg_real_dm_inake_per_weight_per_day_log,
            'avg_real_dm_inake_per_mw_per_day': self.avg_real_dm_inake_per_mw_per_day,
            'avg_real_dm_inake_per_mw_per_day_squared': self.avg_real_dm_inake_per_mw_per_day_squared,
            'avg_real_dm_inake_per_mw_per_day_log': self.avg_real_dm_inake_per_mw_per_day_log,
            'avg_weight': self.avg_weight,
            'avg_mw': self.avg_mw,
        }
    
    def get_feed_composition_features(self):
        """
        Get feed composition features as a dictionary.
        
        Returns:
            dict: Feed composition features including squared and log transformations, 
                  or empty dict if required feeds not present
        """
        if not self.has_required_feeds:
            return {}
        
        return {
            # Original features
            'silage_dm': self.silage_dm,
            'grass_dm': self.grass_dm,
            'slobber_dm': self.slobber_dm,
            'green_dm': self.green_dm,
            'tdn_silage': self.tdn_silage,
            'tdn_rumput': self.tdn_rumput,
            'tdn_slobber': self.tdn_slobber,
            'total_tdn': self.total_tdn,
            '1_total_tdn': 1/self.total_tdn,
            'total_tdn_2': self.total_tdn**2,
            'total_tdn_3': self.total_tdn**3,
            'pk_silage': self.pk_silage,
            'pk_rumput': self.pk_rumput,
            'pk_slobber': self.pk_slobber,
            'total_pk': self.pk_total,
            'per_pk': self.pk_per,
            'FeedRatio': self.feed_ratio,
            'per_slobber_dm': self.per_slobber_dm,
            'per_slobber_tdn': self.per_slobber_tdn,
            'per_rumput_tdn': self.per_rumput_tdn,
            'per_silage_tdn': self.per_silage_tdn,
            'per_green_dm': self.per_green_dm,
            'greens_cost_per_kg': self.greens_cost_per_kg,
            'greens_cost': self.greens_cost,
            'slobber_cost': self.slobber_cost,
            'feed_cost': self.feed_cost,
            'feed_cost_per_dm': self.feed_cost_per_dm,
            'total_tdn': self.total_tdn,
            'total_tdn_dt': self.total_tdn_dt,
            'total_tdn_dt_log': np.log(self.total_tdn_dt),
            'total_tdn_mw_dt': self.total_tdn_mw_dt,
            'total_tdn_mw': self.total_tdn_mw,
            'tdn_silage_dt': self.tdn_silage_dt,
            'tdn_silage_dt_log': self.tdn_silage_dt_log,
            'tdn_rumput_dt': self.tdn_rumput_dt,
            'tdn_rumput_dt_log': self.tdn_rumput_dt_log,
            'silage_x_rumput_tdn': self.tdn_rumput_over_mw_dt*self.tdn_slobber_over_mw_dt,
            'tdn_slobber_dt': self.tdn_slobber_dt,
            'tdn_new_concentrat_dt': self.tdn_new_concentrat_dt,
            'tdn_ricehay_dt': self.tdn_ricehay_dt,
            'tdn_SP2A_dt': self.tdn_sp2a_dt,
            'tdn_SP2B_dt': self.tdn_sp2b_dt,
            'tdn_slobber_dt_log': self.tdn_slobber_dt_log,
            'total_tdn_greens_over_mw': self.total_tdn_greens_over_mw,
            'total_tdn_greens_over_mw_dt': self.total_tdn_greens_over_mw_dt,
            'tdn_silage_over_mw_dt': self.tdn_silage_over_mw_dt,
            'tdn_rumput_over_mw_dt': self.tdn_rumput_over_mw_dt,
            'tdn_slobber_over_mw_dt': self.tdn_slobber_over_mw_dt,
            'tdn_SP2A_over_mw_dt': self.tdn_sp2a_over_mw_dt,
            'tdn_ricehay_over_mw_dt': self.tdn_ricehay_over_mw_dt,
            'tdn_silage_dt': self.tdn_silage_dt,
            'tdn_rumput_dt': self.tdn_rumput_dt,
            'tdn_slobber_dt': self.tdn_slobber_dt,
            'r_tdn_silage_dt': self.r_tdn_silage_dt,
            'r_tdn_rumput_dt': self.r_tdn_rumput_dt,
            'r_tdn_slobber_dt': self.r_tdn_slobber_dt,
            'r_tdn_ricehay_dt': self.r_tdn_ricehay_dt,
            'r_tdn_SP2A_dt': self.r_tdn_sp2a_dt,
 
            'tdn_silage_dt_2': self.tdn_silage_dt**2,
            'tdn_rumput_dt_2': self.tdn_rumput_dt**2,
            'tdn_slobber_dt_2': self.tdn_slobber_dt**2,
            'tdn_silage_over_mw': self.tdn_silage_over_mw,
            'tdn_rumput_over_mw': self.tdn_rumput_over_mw,
            'silage_rumput_mw': self.tdn_silage_over_mw*self.tdn_rumput_over_mw,
            'silage_rumput_dt_r_dmi': ((self.tdn_silage*self.tdn_rumput)/self.day_diff),
            'silage_rumput': ((self.tdn_silage*self.tdn_rumput)),
            'greens_slobber_mw': (
                self.total_tdn_greens_over_mw / self.tdn_slobber_over_mw
                if self.tdn_slobber_over_mw != 0
                else 0
            ),
            'greens_slobber_mw_2': (self.total_tdn_greens_over_mw/self.tdn_slobber_over_mw if self.tdn_slobber_over_mw != 0 else 0),
            'tdn_slobber_over_mw': self.tdn_slobber_over_mw,
            'tdn_slobber_over_mw_2': self.tdn_slobber_over_mw**2,
            'total_tdn_1_dt_dmi': self.total_tdn_1_dt_dmi,
            'total_tdn_2_dt_dmi': self.total_tdn_2_dt_dmi,
            'total_tdn_3_dt_dmi': self.total_tdn_3_dt_dmi,
            
            # Squared features
            'silage_dm_squared': self.silage_dm_squared,
            'grass_dm_squared': self.grass_dm_squared,
            'slobber_dm_squared': self.slobber_dm_squared,
            'green_dm_squared': self.green_dm_squared,
            'total_tdn_squared': self.total_tdn_squared,
            'total_tdn_squared_ddmi': self.total_tdn_squared/self.dm_total,
            'total_tdn_3_ddmi': (self.total_tdn**3)/self.dm_total,
            'FeedRatio_squared': self.feed_ratio_squared,
            'per_slobber_dm_squared': self.per_slobber_dm_squared,
            'per_green_dm_squared': self.per_green_dm_squared,
            'feed_cost_squared': self.feed_cost_squared,
            'feed_cost_per_dm_squared': self.feed_cost_per_dm_squared,
            'tdn_silage_over_mw_dt_2': self.tdn_silage_over_mw_dt**2,
            'tdn_rumput_over_mw_dt_2': self.tdn_rumput_over_mw_dt**2,
            'tdn_slobber_over_mw_dt_2': self.tdn_slobber_over_mw_dt**2,
            'tdn_silage_dt_2_3': self.tdn_silage_dt/((self.total_tdn)**2/3),
            'tdn_rumput_dt_2_3': self.tdn_rumput_dt/((self.total_tdn)**2/3),
            'tdn_slobber_dt_2_3': self.tdn_slobber_dt/((self.total_tdn)**2/3),
            'tdn_silage_dt_r_dmi': self.tdn_silage_dt*((self.dm_total)**1),
            'tdn_rumput_log_r_dmi': self.tdn_rumput_dt*((self.dm_total)**1),
            'tdn_rumput_dt_r_dmi': self.tdn_rumput_log*((self.dm_total)**1),
            'tdn_rumput_dt_r_dmi_log': np.log(self.tdn_rumput_dt*((self.dm_total)**1/2)) if self.tdn_rumput_dt > 0 else 0 ,
            'tdn_slobber_dt_r_dmi': self.tdn_slobber_dt*((self.dm_total)**1),
            'tdn_SP2A_dt_r_dmi': self.tdn_sp2a_dt*((self.dm_total)**1),
            'tdn_concentrats_dt_r_dmi': self.tdn_concentrats_dt*((self.dm_total)**1),
            'tdn_concentrats_dt_3_r_dmi': (self.tdn_concentrats_dt**3)*((self.dm_total)**1),
            'tdn_ricehay_dt_r_dmi': self.tdn_ricehay_dt*((self.dm_total)**1),
            'tdn_silage_dt_r_dmi_mw': (self.tdn_silage_dt*self.dm_total),
            'tdn_rumput_dt_r_dmi_mw': (self.tdn_rumput_dt*self.dm_total),
            'tdn_slobber_dt_r_dmi_mw': (self.tdn_slobber_dt*self.dm_total),
            'tdn_SP2A_dt_r_dmi_mw': (self.tdn_sp2a_dt*self.dm_total),
            'tdn_ricehay_dt_r_dmi_mw': (self.tdn_ricehay_dt*self.dm_total),
            'total_tdn_dt_2': self.total_tdn_dt_squared,
            'total_tdn_dt_3': self.total_tdn_dt**3,
            'total_tdn_mw_dt_3': self.total_tdn_3_dt,
            'total_tdn_mw_dt_2': self.total_tdn_mw_dt_squared,
            'total_tdn_mw_3': self.total_tdn_3,
            'total_tdn_mw_2': self.total_tdn_2,
           
            # dmi features
            'per_slobber_dm_dmi':self.per_slobber_dm_dmi,
            'tdn_silage_over_mw_dt_ddmi': self.tdn_silage_over_mw_dt/self.dm_total_dt,
            'tdn_rumput_over_mw_dt_ddmi': self.tdn_rumput_over_mw_dt/self.dm_total_dt,
            'tdn_slobber_over_mw_dt_ddmi': self.tdn_slobber_over_mw_dt/self.dm_total_dt,
            'tdn_silage_ddmi': (self.tdn_silage/self.dm_total),
            'tdn_rumput_ddmi': self.tdn_rumput/self.dm_total,
            'tdn_slobber_ddmi': self.tdn_slobber/self.dm_total,
            'total_tdn_2_ddmi': (self.total_tdn**2)/self.dm_total,
            'total_dmi': self.dm_total,
            'weight_ddmi': (self.dm_total/self.day_diff)/self.avg_weight,
            'day_diff_ddmi': (self.day_diff)/self.dm_total,
            'day_diff_2_ddmi': (self.day_diff**2)/self.dm_total,
            'total_dmi_2': self.dm_total**2,
            '1_total_dmi': 1/self.dm_total,
            'total_dmi_dt': self.dm_total/self.day_diff,
            'r_total_dmi_dt': (self.dm_total/self.day_diff)**(1/2),
            'total_dmi_dw': self.dm_total/self.avg_weight,
            'total_dmi_log': np.log(self.dm_total),
            'total_dmi_log_dmi': np.log(self.dm_total)*self.dm_total,
            'total_dmi_log_dt': np.log(self.dm_total/self.day_diff),
            'total_dmi_log_dw': np.log(self.dm_total/self.avg_weight),

            # Log features
            'silage_dm_log': self.silage_dm_log,
            'grass_dm_log': self.grass_dm_log,
            'slobber_dm_log': self.slobber_dm_log,
            'green_dm_log': self.green_dm_log,
            'total_tdn_log': self.total_tdn_log,
            'FeedRatio_log': self.feed_ratio_log,
            'per_slobber_dm_log': self.per_slobber_dm_log,
            'per_green_dm_log': self.per_green_dm_log,
            'feed_cost_log': self.feed_cost_log,
            'feed_cost_per_dm_log': self.feed_cost_per_dm_log,
            'tdn_slobber_over_mw_dt_log': np.log(self.tdn_slobber_over_mw_dt) if self.tdn_slobber_over_mw_dt > 0 else 0 ,
            'total_tdn_dt_log': self.total_tdn_dt_log,
            'total_tdn_mw_dt_log': self.total_tdn_mw_dt_log,
            'tdn_silage_log': self.tdn_silage_log,
            'tdn_rumput_log': self.tdn_rumput_log,
            '1_over_tdn_rumput': self.over_tdn_rumput,
            'tdn_slobber_log': self.tdn_slobber_log,
        }
    
    def get_all_features(self):
        """
        Get all features as a single dictionary.
        
        Returns:
            dict: All features, or None if required feeds not present
        """
        if not self.has_required_feeds:
            return None
        
        features = self.get_dmi_features()
        features.update(self.get_feed_composition_features())
        return features
