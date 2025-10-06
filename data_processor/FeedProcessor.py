from datetime import datetime
from consts.consts import tdn_table, costs_per_dm

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
    
    def _calculate_feed_composition(self):
        """Calculate feed composition metrics."""
        ration = self.feed_history.get_diet(self.start_date, self.end_date)
        
        # Check if we have the required feed types
        self.has_required_feeds = (
            ('Silase Jagung' in ration or 'Rumput' in ration) and 
            'Slobber Mix' in ration
        )
        
        if not self.has_required_feeds:
            return
        
        # Calculate feed dry matter amounts
        silage_dm = ration.get('Silase Jagung', {}).get('asFedIntakePerCow', 0) * 0.3
        grass_dm = ration.get('Rumput', {}).get('asFedIntakePerCow', 0) * 0.2
        slobber_dm = ration['Slobber Mix']['asFedIntakePerCow'] * 0.4506
        
        self.silage_dm = silage_dm
        self.grass_dm = grass_dm
        self.slobber_dm = slobber_dm
        self.green_dm = silage_dm + grass_dm
        
        # TDN calculations
        self.tdn_silage = silage_dm * tdn_table['silage']
        self.tdn_rumput = grass_dm * tdn_table['grass']
        self.tdn_slobber = slobber_dm * tdn_table['slobber']
        self.total_tdn = self.tdn_slobber + self.tdn_silage + self.tdn_rumput
        
        # Feed ratios and percentages
        self.feed_ratio = self.green_dm / self.slobber_dm if self.slobber_dm > 0 else 0
        self.per_slobber_dm = self.slobber_dm / self.total_dm_intake if self.total_dm_intake > 0 else 0
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
        self.slobber_cost = slobber_dm * costs_per_dm['slobber']
        self.feed_cost = self.greens_cost + self.slobber_cost
        self.feed_cost_per_dm = self.feed_cost / self.total_dm_intake if self.total_dm_intake > 0 else 0
        
        # TDN per day and metabolic weight
        self.total_tdn_dt = self.total_tdn / self.day_diff
        self.total_tdn_mw_dt = self.total_tdn / self.day_diff / self.avg_mw
        
        self.tdn_silage_over_mw_dt = (self.tdn_silage / self.avg_mw) / self.day_diff
        self.tdn_rumput_over_mw_dt = (self.tdn_rumput / self.avg_mw) / self.day_diff
        self.tdn_slobber_over_mw_dt = (self.tdn_slobber / self.avg_mw) / self.day_diff
        
        self.total_tdn_2_dt_dmi = (
            ((self.total_tdn_dt / self.avg_mw)**2) * 
            self.avg_real_dm_inake_per_weight_per_day
        )
        self.total_tdn_3_dt_dmi = (
            ((self.total_tdn_dt / self.avg_mw)**3) * 
            self.avg_real_dm_inake_per_weight_per_day
        )
    
    def get_dmi_features(self):
        """
        Get DMI-related features as a dictionary.
        
        Returns:
            dict: DMI features
        """
        return {
            'total_dm_intake': self.total_dm_intake,
            'avg_dm_intake_per_day': self.avg_dm_intake_per_day,
            'avg_real_dm_inake_per_weight_per_day': self.avg_real_dm_inake_per_weight_per_day,
            'avg_real_dm_inake_per_mw_per_day': self.avg_real_dm_inake_per_mw_per_day,
            'avg_weight': self.avg_weight,
            'avg_mw': self.avg_mw,
        }
    
    def get_feed_composition_features(self):
        """
        Get feed composition features as a dictionary.
        
        Returns:
            dict: Feed composition features, or empty dict if required feeds not present
        """
        if not self.has_required_feeds:
            return {}
        
        return {
            'silage_dm': self.silage_dm,
            'grass_dm': self.grass_dm,
            'slobber_dm': self.slobber_dm,
            'green_dm': self.green_dm,
            'tdn_silage': self.tdn_silage,
            'tdn_rumput': self.tdn_rumput,
            'tdn_slobber': self.tdn_slobber,
            'total_tdn': self.total_tdn,
            'FeedRatio': self.feed_ratio,
            'per_slobber_dm': self.per_slobber_dm,
            'per_green_dm': self.per_green_dm,
            'greens_cost_per_kg': self.greens_cost_per_kg,
            'greens_cost': self.greens_cost,
            'slobber_cost': self.slobber_cost,
            'feed_cost': self.feed_cost,
            'feed_cost_per_dm': self.feed_cost_per_dm,
            'total_tdn_dt': self.total_tdn_dt,
            'total_tdn_mw_dt': self.total_tdn_mw_dt,
            'tdn_silage_over_mw_dt': self.tdn_silage_over_mw_dt,
            'tdn_rumput_over_mw_dt': self.tdn_rumput_over_mw_dt,
            'tdn_slobber_over_mw_dt': self.tdn_slobber_over_mw_dt,
            'total_tdn_2_dt_dmi': self.total_tdn_2_dt_dmi,
            'total_tdn_3_dt_dmi': self.total_tdn_3_dt_dmi,
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
