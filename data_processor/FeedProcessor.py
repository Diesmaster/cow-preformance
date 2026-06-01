from datetime import datetime
from consts.consts import costs_per_dm
import numpy as np


class FeedProcessor:
    def __init__(self, feed_history, weight_history, start_index, n_weighing,
                 ingredient_nutrition: dict = None):
        self.feed_history = feed_history
        self.weight_history = weight_history
        self.start_index = start_index
        self.n_weighing = n_weighing
        self.ingredient_nutrition = ingredient_nutrition or {}

        self.start_date = weight_history.data[start_index]['date']
        self.end_date = weight_history.data[start_index + n_weighing]['date']

        self.day_diff = (datetime.strptime(self.end_date, "%Y-%m-%d") -
                         datetime.strptime(self.start_date, "%Y-%m-%d")).days

        self._calculate_dmi_metrics()
        self._calculate_feed_composition()

    # ------------------------------------------------------------------ #
    #  Nutrient lookup                                                     #
    # ------------------------------------------------------------------ #

    def _get_nutrient(self, feed_name: str, nutrient: str) -> float:
        # 1. flat ingredient nutrition lookup
        entry = self.ingredient_nutrition.get(feed_name, {})
        val = entry.get(nutrient)
        if val is not None:
            return val


        # 2. RecipeData object stored directly in ingredient_nutrition
        #    (caller can pre-populate this with loaded RecipeData objects)
        recipe_obj = self.ingredient_nutrition.get(f'__recipe_{feed_name}')
        if recipe_obj is not None:
            from data_objects.recipe_data import RecipeData as _RecipeData
            if isinstance(recipe_obj, _RecipeData):
                return recipe_obj.to_nutrition_dict().get(nutrient, 0.0)

        # 3. fallback: check ration dict for a loaded RecipeData ingredient
        ration = self.feed_history.get_diet(self.start_date, self.end_date)
        item = ration.get(feed_name)
        if isinstance(item, dict):
            ingredient_obj = item.get('ingredient')
            if ingredient_obj is not None:
                from data_objects.recipe_data import RecipeData as _RecipeData
                if isinstance(ingredient_obj, _RecipeData):
                    return ingredient_obj.to_nutrition_dict().get(nutrient, 0.0)

        return 0.0

    def _tdn(self, n):         return self._get_nutrient(n, 'tdn')
    def _cp(self, n):          return self._get_nutrient(n, 'cp')
    def _rdp(self, n):         return self._get_nutrient(n, 'rdp')
    def _cf(self, n):          return self._get_nutrient(n, 'cf')
    def _fat(self, n):         return self._get_nutrient(n, 'fat')
    def _betn(self, n):        return self._get_nutrient(n, 'betn')
    def _total_carbs(self, n): return self._cf(n) + self._betn(n)

    # ------------------------------------------------------------------ #
    #  DMI metrics                                                         #
    # ------------------------------------------------------------------ #

    def _calculate_dmi_metrics(self):
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
            dm_intake_per_mw = (dm_intake / (weight_at_index ** 0.75)) * 100

            avg_dm_intake_per_day += dm_intake / day_diff_segment
            avg_real_dm_inake_per_weight_per_day += (dm_intake_per_weight / day_diff_segment) * day_diff_fraction
            avg_real_dm_inake_per_mw_per_day += (dm_intake_per_mw / day_diff_segment) * day_diff_fraction
            avg_weight += weight_at_index * day_diff_fraction

        self.avg_dm_intake_per_day = avg_dm_intake_per_day
        self.avg_real_dm_inake_per_weight_per_day = avg_real_dm_inake_per_weight_per_day
        self.avg_real_dm_inake_per_mw_per_day = avg_real_dm_inake_per_mw_per_day
        self.avg_weight = avg_weight
        self.avg_mw = avg_weight ** 0.75

        self.avg_dm_intake_per_day_squared = avg_dm_intake_per_day ** 2
        self.avg_real_dm_inake_per_weight_per_day_squared = avg_real_dm_inake_per_weight_per_day ** 2
        self.avg_real_dm_inake_per_mw_per_day_squared = avg_real_dm_inake_per_mw_per_day ** 2

        self.avg_dm_intake_per_day_log = np.log(avg_dm_intake_per_day) if avg_dm_intake_per_day > 0 else np.nan
        self.avg_real_dm_inake_per_weight_per_day_log = np.log(avg_real_dm_inake_per_weight_per_day) if avg_real_dm_inake_per_weight_per_day > 0 else np.nan
        self.avg_real_dm_inake_per_mw_per_day_log = np.log(avg_real_dm_inake_per_mw_per_day) if avg_real_dm_inake_per_mw_per_day > 0 else np.nan

    # ------------------------------------------------------------------ #
    #  Feed composition                                                    #
    # ------------------------------------------------------------------ #

    def _calculate_feed_composition(self):
        ration = self.feed_history.get_diet(self.start_date, self.end_date)
        dm_total = self.feed_history.get_dry_matter_intake(self.start_date, self.end_date)

        self.has_required_feeds = (
            ('Silase Jagung' in ration or 'Rumput' in ration or 'Pakchong Grass - Qurban' in ration) and
            ('Slobber Mix' in ration or 'SP2B Mix' in ration or 'SP2A Mix' in ration)
        )

        if not self.has_required_feeds:
            return

        # ---- DM amounts ------------------------------------------------
        silage_dm = ration.get('Silase Jagung', {}).get('asFedIntakePerCow', 0) * 0.3
        grass_dm  = ration.get('Rumput', ration.get('Pakchong Grass - Qurban', {})).get('asFedIntakePerCow', 0) * 0.2

        self.rice_hay_dm = 0
        if 'Rice Hay' in ration:
            self.rice_hay_dm = ration['Rice Hay']['asFedIntakePerCow'] * 0.85

        self.slobber_dm = 0
        if 'Slobber Mix' in ration:
            self.slobber_dm = ration['Slobber Mix']['asFedIntakePerCow'] * 0.4506

        self.SP2B_mix_dm = 0
        if 'SP2B Mix' in ration:
            self.SP2B_mix_dm = ration['SP2B Mix']['asFedIntakePerCow'] * 0.8623

        self.SP2A_mix_dm = 0
        if 'SP2A Mix' in ration:
            self.SP2A_mix_dm += ration['SP2A Mix']['asFedIntakePerCow'] * 0.8623
        if 'Konsentrat' in ration:
            self.SP2A_mix_dm += ration['Konsentrat']['asFedIntakePerCow'] * 0.8623

        self.SMG_mix_dm = 0
        if 'SMG Mixfeed S14' in ration:
            self.SMG_mix_dm = ration['SMG Mixfeed S14']['asFedIntakePerCow'] * 0.8

        self.ampas_tahu_dm = 0
        if 'Ampas Tahu' in ration:
            self.ampas_tahu_dm = ration['Ampas Tahu']['asFedIntakePerCow'] * 0.16

        if (grass_dm + silage_dm + self.SP2B_mix_dm + self.SP2A_mix_dm) > dm_total + 5:
            print("##" * 30)
            print(f"{self.start_date=}")
            print(f"{(grass_dm + silage_dm + self.slobber_dm + self.SP2B_mix_dm + self.SP2A_mix_dm)}")
            print(f"{dm_total=}")
            print(f"{ration=}")
            print("##" * 30)

        self.dm_total    = dm_total
        self.dm_total_dt = dm_total / self.day_diff
        self.silage_dm   = silage_dm
        self.grass_dm    = grass_dm
        self.green_dm    = silage_dm + grass_dm + self.rice_hay_dm

        # ---- TDN -------------------------------------------------------
        self.tdn_silage     = silage_dm            * self._tdn('Silase Jagung')
        self.tdn_ricehay    = self.rice_hay_dm     * self._tdn('Rice Hay')
        self.tdn_rumput     = grass_dm             * self._tdn('Rumput')
        self.tdn_slobber    = self.slobber_dm      * self._tdn('Slobber Mix')
        self.tdn_SP2B_mix   = self.SP2B_mix_dm    * self._tdn('SP2B Mix')
        self.tdn_SP2A_mix   = self.SP2A_mix_dm    * self._tdn('SP2A Mix')
        self.tdn_SMG_mix    = self.SMG_mix_dm     * self._tdn('SMG Mixfeed S14')
        self.tdn_ampas_tahu = self.ampas_tahu_dm   * self._tdn('Ampas Tahu')

        self.tdn_greens    = self.tdn_silage + self.tdn_rumput + self.tdn_ricehay
        self.tdn_greens_dt = self.tdn_greens / self.day_diff

        self.total_tdn = (
            self.tdn_silage + self.tdn_rumput + self.tdn_ricehay +
            self.tdn_slobber + self.tdn_SP2A_mix + self.tdn_SP2B_mix +
            self.tdn_SMG_mix + self.tdn_ampas_tahu
        )
        self.total_tdn_ratio = self.total_tdn / self.total_dm_intake

        # ---- CP --------------------------------------------------------
        self.pk_silage    = silage_dm            * self._cp('Silase Jagung')
        self.pk_rumput    = grass_dm             * self._cp('Rumput')
        self.pk_ricehay   = self.rice_hay_dm     * self._cp('Rice Hay')
        self.pk_slobber   = self.slobber_dm      * self._cp('Slobber Mix')
        self.pk_sp2b      = self.SP2B_mix_dm    * self._cp('SP2B Mix')
        self.pk_sp2a      = self.SP2A_mix_dm    * self._cp('SP2A Mix')
        self.pk_smg       = self.SMG_mix_dm     * self._cp('SMG Mixfeed S14')
        self.pk_tahu      = self.ampas_tahu_dm   * self._cp('Ampas Tahu')


        self.pk_total = (
            self.pk_silage + self.pk_rumput + self.pk_ricehay +
            self.pk_slobber + self.pk_sp2a + self.pk_sp2b +
            self.pk_smg + self.pk_tahu
        )
        self.pk_per = self.pk_total / self.dm_total

        # ---- RDP -------------------------------------------------------
        self.rdp_silage  = silage_dm            * self._rdp('Silase Jagung')
        self.rdp_rumput  = grass_dm             * self._rdp('Rumput')
        self.rdp_ricehay = self.rice_hay_dm     * self._rdp('Rice Hay')
        self.rdp_slobber = self.slobber_dm      * self._rdp('Slobber Mix')
        self.rdp_sp2b    = self.SP2B_mix_dm    * self._rdp('SP2B Mix')
        self.rdp_sp2a    = self.SP2A_mix_dm    * self._rdp('SP2A Mix')
        self.rdp_smg     = self.SMG_mix_dm     * self._rdp('SMG Mixfeed S14')
        self.rdp_tahu    = self.ampas_tahu_dm   * self._rdp('Ampas Tahu')

        self.total_rdp = (
            self.rdp_silage + self.rdp_rumput + self.rdp_ricehay +
            self.rdp_slobber + self.rdp_sp2b + self.rdp_sp2a +
            self.rdp_smg + self.rdp_tahu
        )
        self.total_rdp_dt = self.total_rdp / self.day_diff
        self.rdp_per      = self.total_rdp / self.dm_total
        self.rdp_per_cp   = self.total_rdp / self.pk_total if self.pk_total > 0 else np.nan

        # ---- Crude Fibre -----------------------------------------------
        self.cf_silage  = silage_dm            * self._cf('Silase Jagung')
        self.cf_rumput  = grass_dm             * self._cf('Rumput')
        self.cf_ricehay = self.rice_hay_dm     * self._cf('Rice Hay')
        self.cf_slobber = self.slobber_dm      * self._cf('Slobber Mix')
        self.cf_sp2b    = self.SP2B_mix_dm    * self._cf('SP2B Mix')
        self.cf_sp2a    = self.SP2A_mix_dm    * self._cf('SP2A Mix')
        self.cf_smg     = self.SMG_mix_dm     * self._cf('SMG Mixfeed S14')
        self.cf_tahu    = self.ampas_tahu_dm   * self._cf('Ampas Tahu')

        self.total_cf = (
            self.cf_silage + self.cf_rumput + self.cf_ricehay +
            self.cf_slobber + self.cf_sp2b + self.cf_sp2a +
            self.cf_smg + self.cf_tahu
        )
        self.total_cf_dt = self.total_cf / self.day_diff
        self.cf_per      = self.total_cf / self.dm_total

        # ---- Crude Fat -------------------------------------------------
        self.fat_silage  = silage_dm            * self._fat('Silase Jagung')
        self.fat_rumput  = grass_dm             * self._fat('Rumput')
        self.fat_ricehay = self.rice_hay_dm     * self._fat('Rice Hay')
        self.fat_slobber = self.slobber_dm      * self._fat('Slobber Mix')
        self.fat_sp2b    = self.SP2B_mix_dm    * self._fat('SP2B Mix')
        self.fat_sp2a    = self.SP2A_mix_dm    * self._fat('SP2A Mix')
        self.fat_smg     = self.SMG_mix_dm     * self._fat('SMG Mixfeed S14')
        self.fat_tahu    = self.ampas_tahu_dm   * self._fat('Ampas Tahu')

        self.total_fat = (
            self.fat_silage + self.fat_rumput + self.fat_ricehay +
            self.fat_slobber + self.fat_sp2b + self.fat_sp2a +
            self.fat_smg + self.fat_tahu
        )
        self.total_fat_dt = self.total_fat / self.day_diff
        self.fat_per      = self.total_fat / self.dm_total

        # ---- BETN / NFE ------------------------------------------------
        self.betn_silage  = silage_dm            * self._betn('Silase Jagung')
        self.betn_rumput  = grass_dm             * self._betn('Rumput')
        self.betn_ricehay = self.rice_hay_dm     * self._betn('Rice Hay')
        self.betn_slobber = self.slobber_dm      * self._betn('Slobber Mix')
        self.betn_sp2b    = self.SP2B_mix_dm    * self._betn('SP2B Mix')
        self.betn_sp2a    = self.SP2A_mix_dm    * self._betn('SP2A Mix')
        self.betn_smg     = self.SMG_mix_dm     * self._betn('SMG Mixfeed S14')
        self.betn_tahu    = self.ampas_tahu_dm   * self._betn('Ampas Tahu')

        self.total_betn = (
            self.betn_silage + self.betn_rumput + self.betn_ricehay +
            self.betn_slobber + self.betn_sp2b + self.betn_sp2a +
            self.betn_smg + self.betn_tahu
        )
        self.total_betn_dt = self.total_betn / self.day_diff
        self.betn_per      = self.total_betn / self.dm_total

        # ---- Total Carbohydrates (CF + BETN) ---------------------------
        self.carbs_silage  = silage_dm            * self._total_carbs('Silase Jagung')
        self.carbs_rumput  = grass_dm             * self._total_carbs('Rumput')
        self.carbs_ricehay = self.rice_hay_dm     * self._total_carbs('Rice Hay')
        self.carbs_slobber = self.slobber_dm      * self._total_carbs('Slobber Mix')
        self.carbs_sp2b    = self.SP2B_mix_dm    * self._total_carbs('SP2B Mix')
        self.carbs_sp2a    = self.SP2A_mix_dm    * self._total_carbs('SP2A Mix')
        self.carbs_smg     = self.SMG_mix_dm     * self._total_carbs('SMG Mixfeed S14')
        self.carbs_tahu    = self.ampas_tahu_dm   * self._total_carbs('Ampas Tahu')

        self.total_carbs = (
            self.carbs_silage + self.carbs_rumput + self.carbs_ricehay +
            self.carbs_slobber + self.carbs_sp2b + self.carbs_sp2a +
            self.carbs_smg + self.carbs_tahu
        )
        self.total_carbs_dt = self.total_carbs / self.day_diff
        self.carbs_per      = self.total_carbs / self.dm_total

        # ---- Ratios ----------------------------------------------------
        self.dm_silage_ratio   = silage_dm           / self.total_dm_intake
        self.dm_ricehay_ratio  = self.rice_hay_dm    / self.total_dm_intake
        self.dm_rumput_ratio   = grass_dm            / self.total_dm_intake
        self.dm_slobber_ratio  = self.slobber_dm     / self.total_dm_intake
        self.dm_SP2B_mix_ratio = self.SP2B_mix_dm   / self.total_dm_intake
        self.dm_SP2A_mix_ratio = self.SP2A_mix_dm   / self.total_dm_intake
        self.dm_concentrats    = self.dm_slobber_ratio + self.dm_SP2A_mix_ratio + self.dm_SP2B_mix_ratio
        self.dm_concentrats_ratio = self.dm_concentrats / self.total_dm_intake

        self.tdn_new_concentrat    = self.tdn_SP2B_mix + self.tdn_SP2A_mix
        self.tdn_concentrats       = self.tdn_SP2A_mix + self.tdn_SP2B_mix + self.tdn_slobber
        self.tdn_concentrats_ratio = self.tdn_concentrats / self.dm_concentrats_ratio

        self.tdn_SP2A_mix_2 = (self.SP2A_mix_dm ** 2) * self._tdn('SP2A Mix')
        self.tdn_slobber_2  = (self.slobber_dm ** 2)   * self._tdn('Slobber Mix')

        self.per_slobber_tdn = self.tdn_slobber   / self.total_tdn
        self.per_rumput_tdn  = self.tdn_rumput    / self.total_tdn
        self.per_silage_tdn  = self.tdn_silage    / self.total_tdn

        self.not_greens     = self.slobber_dm + self.SP2A_mix_dm + self.SP2B_mix_dm
        self.feed_ratio     = self.green_dm / self.not_greens if self.not_greens > 0 else 0
        self.per_slobber_dm = self.not_greens / self.total_dm_intake if self.total_dm_intake > 0 else 0
        self.per_green_dm   = self.green_dm   / self.total_dm_intake if self.total_dm_intake > 0 else 0

        # ---- Costs -----------------------------------------------------
        if self.green_dm > 0:
            self.greens_cost_per_kg = (
                (silage_dm * costs_per_dm['silage'] + grass_dm * costs_per_dm['grass']) /
                self.green_dm
            )
        else:
            self.greens_cost_per_kg = costs_per_dm['silage']

        self.greens_cost      = silage_dm * costs_per_dm['silage'] + grass_dm * costs_per_dm['grass']
        self.slobber_cost     = self.slobber_dm * costs_per_dm['slobber']
        self.feed_cost        = self.greens_cost + self.slobber_cost
        self.feed_cost_per_dm = self.feed_cost / self.total_dm_intake if self.total_dm_intake > 0 else 0

        # ---- TDN per day / per metabolic weight ------------------------
        self.total_tdn_dt    = self.total_tdn / self.day_diff
        self.total_tdn_mw_dt = self.total_tdn / self.day_diff / self.avg_mw
        self.total_tdn_mw    = self.total_tdn / self.avg_mw

        self.tdn_silage_dt         = self.tdn_silage         / self.day_diff
        self.tdn_rumput_dt         = self.tdn_rumput         / self.day_diff
        self.tdn_ampas_tahu_dt     = self.tdn_ampas_tahu     / self.day_diff
        self.tdn_slobber_dt        = self.tdn_slobber        / self.day_diff
        self.tdn_new_concentrat_dt = self.tdn_new_concentrat / self.day_diff
        self.tdn_sp2a_dt           = self.tdn_SP2A_mix       / self.day_diff
        self.tdn_SMG_dt            = self.tdn_SMG_mix        / self.day_diff
        self.tdn_ricehay_dt        = self.tdn_ricehay        / self.day_diff
        self.tdn_sp2b_dt           = self.tdn_SP2B_mix       / self.day_diff
        self.tdn_tahu_dt           = self.tdn_ampas_tahu     / self.day_diff

        self.tdn_concentrats_2_dt = (self.tdn_sp2a_dt ** 2 + self.tdn_slobber_dt ** 2)
        self.tdn_concentrats_dt   = self.tdn_sp2a_dt + self.tdn_slobber_dt

        self.r_tdn_silage_dt         = (self.tdn_silage         / self.day_diff) ** 0.5
        self.r_tdn_rumput_dt         = (self.tdn_rumput         / self.day_diff) ** 0.5
        self.r_tdn_slobber_dt        = (self.tdn_slobber        / self.day_diff) ** 0.5
        self.r_tdn_new_concentrat_dt = (self.tdn_new_concentrat / self.day_diff) ** 0.5
        self.r_tdn_sp2a_dt           = (self.tdn_SP2A_mix       / self.day_diff) ** 0.5
        self.r_tdn_ricehay_dt        = (self.tdn_ricehay        / self.day_diff) ** 0.5
        self.r_tdn_sp2b_dt           = (self.tdn_SP2B_mix       / self.day_diff) ** 0.5

        self.total_tdn_greens            = self.tdn_silage + self.tdn_rumput
        self.total_tdn_greens_over_mw    = self.total_tdn_greens / self.avg_mw
        self.total_tdn_greens_over_mw_dt = self.total_tdn_greens_over_mw / self.day_diff

        self.tdn_silage_over_mw_dt  = (self.tdn_silage   / self.avg_mw) / self.day_diff
        self.tdn_rumput_over_mw_dt  = (self.tdn_rumput   / self.avg_mw) / self.day_diff
        self.tdn_slobber_over_mw_dt = (self.tdn_slobber  / self.avg_mw) / self.day_diff
        self.tdn_sp2a_over_mw_dt    = (self.tdn_SP2A_mix / self.avg_mw) / self.day_diff
        self.tdn_ricehay_over_mw_dt = (self.tdn_ricehay  / self.avg_mw) / self.day_diff

        self.tdn_silage_over_mw  = self.tdn_silage  / self.avg_mw
        self.tdn_rumput_over_mw  = self.tdn_rumput  / self.avg_mw
        self.tdn_slobber_over_mw = self.tdn_slobber / self.avg_mw

        self.tdn_ratio     = self.total_tdn / self.total_dm_intake
        self.tdn_ratio_r_3 = (self.total_tdn / self.total_dm_intake) ** (1 / 3)

        self.total_tdn_1_dt_dmi = self.total_tdn_dt        * self.avg_real_dm_inake_per_weight_per_day
        self.total_tdn_2_dt_dmi = (self.total_tdn_dt ** 2) * self.avg_real_dm_inake_per_weight_per_day
        self.total_tdn_3_dt_dmi = (self.total_tdn_dt ** 3) * self.avg_real_dm_inake_per_weight_per_day

        # ---- Squared / power transforms --------------------------------
        self.silage_dm_squared        = silage_dm ** 2
        self.grass_dm_squared         = grass_dm  ** 2
        self.slobber_dm_squared       = self.slobber_dm ** 2
        self.green_dm_squared         = self.green_dm   ** 2
        self.total_tdn_squared        = self.total_tdn  ** 2
        self.feed_ratio_squared       = self.feed_ratio ** 2
        self.per_slobber_dm_squared   = self.per_slobber_dm ** 2
        self.per_green_dm_squared     = self.per_green_dm  ** 2
        self.feed_cost_squared        = self.feed_cost     ** 2
        self.feed_cost_per_dm_squared = self.feed_cost_per_dm ** 2
        self.total_tdn_dt_squared     = self.total_tdn_dt ** 2
        self.total_tdn_3_dt           = self.total_tdn_mw_dt ** 3
        self.total_tdn_2              = self.total_tdn_mw ** 2
        self.total_tdn_3              = self.total_tdn_mw ** 3
        self.total_tdn_mw_dt_squared  = self.total_tdn_mw_dt ** 2

        # ---- Log transforms --------------------------------------------
        self.silage_dm_log        = np.log(silage_dm)             if silage_dm             > 0 else 0
        self.grass_dm_log         = np.log(grass_dm)              if grass_dm              > 0 else 0
        self.slobber_dm_log       = np.log(self.slobber_dm)       if self.slobber_dm       > 0 else np.nan
        self.green_dm_log         = np.log(self.green_dm)         if self.green_dm         > 0 else np.nan
        self.total_tdn_log        = np.log(self.total_tdn)        if self.total_tdn        > 0 else np.nan
        self.feed_ratio_log       = np.log(self.feed_ratio)       if self.feed_ratio       > 0 else np.nan
        self.per_slobber_dm_log   = np.log(self.per_slobber_dm)   if self.per_slobber_dm   > 0 else np.nan
        self.per_green_dm_log     = np.log(self.per_green_dm)     if self.per_green_dm     > 0 else np.nan
        self.feed_cost_log        = np.log(self.feed_cost)        if self.feed_cost        > 0 else np.nan
        self.feed_cost_per_dm_log = np.log(self.feed_cost_per_dm) if self.feed_cost_per_dm > 0 else np.nan
        self.total_tdn_dt_log     = np.log(self.total_tdn_dt)     if self.total_tdn_dt     > 0 else np.nan
        self.total_tdn_mw_dt_log  = np.log(self.total_tdn_mw_dt)  if self.total_tdn_mw_dt  > 0 else np.nan
        self.tdn_silage_log       = np.log(self.tdn_silage)       if self.tdn_silage       > 0 else 0
        self.tdn_rumput_log       = np.log(self.tdn_rumput)       if self.tdn_rumput       > 0 else 0
        self.over_tdn_rumput      = 1 / self.tdn_rumput           if self.tdn_rumput       > 0 else 0
        self.tdn_slobber_log      = np.log(self.tdn_slobber)      if self.tdn_slobber      > 0 else np.nan
        self.tdn_silage_dt_log    = np.log(self.tdn_silage_dt)    if self.tdn_silage       > 0 else 0
        self.tdn_rumput_dt_log    = np.log(self.tdn_rumput_dt)    if self.tdn_rumput       > 0 else 0
        self.tdn_slobber_dt_log   = np.log(self.tdn_slobber_dt)   if self.tdn_slobber      > 0 else 0

        self.per_slobber_dm_dmi = self.per_slobber_dm * self.avg_real_dm_inake_per_weight_per_day

    # ------------------------------------------------------------------ #
    #  Public feature getters                                              #
    # ------------------------------------------------------------------ #

    def get_dmi_features(self):
        return {
            'total_dm_intake':                              self.total_dm_intake,
            'avg_dm_intake_per_day':                        self.avg_dm_intake_per_day,
            'avg_dm_intake_per_day_squared':                self.avg_dm_intake_per_day_squared,
            'avg_dm_intake_per_day_log':                    self.avg_dm_intake_per_day_log,
            'avg_real_dm_inake_per_weight_per_day':         self.avg_real_dm_inake_per_weight_per_day,
            'one_over_dmi_dt':                              1 / self.avg_real_dm_inake_per_weight_per_day,
            'avg_real_dm_inake_per_weight_per_day_squared': self.avg_real_dm_inake_per_weight_per_day_squared,
            'avg_real_dm_inake_per_weight_per_day_log':     self.avg_real_dm_inake_per_weight_per_day_log,
            'avg_real_dm_inake_per_mw_per_day':             self.avg_real_dm_inake_per_mw_per_day,
            'avg_real_dm_inake_per_mw_per_day_squared':     self.avg_real_dm_inake_per_mw_per_day_squared,
            'avg_real_dm_inake_per_mw_per_day_log':         self.avg_real_dm_inake_per_mw_per_day_log,
            'avg_weight':                                   self.avg_weight,
            'avg_mw':                                       self.avg_mw,
        }

    def get_nutrition_summary(self):
        if not self.has_required_feeds:
            return {}

        return {
            'total_tdn_kg':   self.total_tdn,
            'total_cp_kg':    self.pk_total,
            'total_rdp_kg':   self.total_rdp,
            'total_cf_kg':    self.total_cf,
            'total_fat_kg':   self.total_fat,
            'total_betn_kg':  self.total_betn,
            'total_carbs_kg': self.total_carbs,

            'tdn_pct_dm':   self.total_tdn    / self.dm_total * 100,
            'cp_pct_dm':    self.pk_total     / self.dm_total * 100,
            'rdp_pct_dm':   self.rdp_per      * 100,
            'rdp_pct_cp':   self.rdp_per_cp   * 100 if not np.isnan(self.rdp_per_cp) else np.nan,
            'cf_pct_dm':    self.cf_per       * 100,
            'fat_pct_dm':   self.fat_per      * 100,
            'betn_pct_dm':  self.betn_per     * 100,
            'carbs_pct_dm': self.carbs_per    * 100,

            'total_tdn_dt':   self.total_tdn_dt,
            'total_cp_dt':    self.pk_total    / self.day_diff,
            'total_rdp_dt':   self.total_rdp_dt,
            'total_cf_dt':    self.total_cf_dt,
            'total_fat_dt':   self.total_fat_dt,
            'total_betn_dt':  self.total_betn_dt,
            'total_carbs_dt': self.total_carbs_dt,
        }

    def get_feed_composition_features(self):
        if not self.has_required_feeds:
            return {}

        return {
            'silage_dm':                    self.silage_dm,
            'grass_dm':                     self.grass_dm,
            'slobber_dm':                   self.slobber_dm,
            'green_dm':                     self.green_dm,
            'tdn_silage':                   self.tdn_silage,
            'tdn_rumput':                   self.tdn_rumput,
            'tdn_slobber':                  self.tdn_slobber,
            'total_tdn':                    self.total_tdn,
            '1_total_tdn':                  1 / self.total_tdn,
            'total_tdn_2':                  self.total_tdn ** 2,
            'total_tdn_3':                  self.total_tdn ** 3,
            'pk_silage':                    self.pk_silage,
            'pk_rumput':                    self.pk_rumput,
            'pk_slobber':                   self.pk_slobber,
            'total_pk':                     self.pk_total,
            'total_pk_log':                 np.log(self.pk_total) if self.pk_total > 0 else np.nan,
            'total_pk_2':                   self.pk_total ** 2,
            'per_pk':                       self.pk_per,
            'per_pk_2':                     self.pk_per ** 2,

            'total_rdp':                    self.total_rdp,
            'total_rdp_dt':                 self.total_rdp_dt,
            'rdp_pct_dm':                   self.rdp_per * 100,
            'rdp_pct_cp':                   self.rdp_per_cp * 100 if not np.isnan(self.rdp_per_cp) else np.nan,
            'total_cf':                     self.total_cf,
            'total_cf_dt':                  self.total_cf_dt,
            'cf_pct_dm':                    self.cf_per * 100,
            'total_fat':                    self.total_fat,
            'total_fat_dt':                 self.total_fat_dt,
            'fat_pct_dm':                   self.fat_per * 100,
            'total_betn':                   self.total_betn,
            'total_betn_dt':                self.total_betn_dt,
            'betn_pct_dm':                  self.betn_per * 100,

            'FeedRatio':                    self.feed_ratio,
            'per_slobber_dm':               self.per_slobber_dm,
            'per_slobber_tdn':              self.per_slobber_tdn,
            'per_rumput_tdn':               self.per_rumput_tdn,
            'per_silage_tdn':               self.per_silage_tdn,
            'per_green_dm':                 self.per_green_dm,
            'greens_cost_per_kg':           self.greens_cost_per_kg,
            'greens_cost':                  self.greens_cost,
            'slobber_cost':                 self.slobber_cost,
            'feed_cost':                    self.feed_cost,
            'feed_cost_per_dm':             self.feed_cost_per_dm,
            'total_tdn_dt':                 self.total_tdn_dt,
            'total_tdn_dt_2':               self.total_tdn_dt ** 2,
            'total_tdn_dt_2_sdmi':          (self.total_tdn_ratio ** 2) * ((self.total_dm_intake / self.day_diff) ** 3),
            'total_tdn_dt_3':               self.total_tdn_dt ** 3,
            'total_tdn_dt_log':             np.log(self.total_tdn_dt),
            'total_tdn_mw_dt':              self.total_tdn_mw_dt,
            'total_tdn_mw':                 self.total_tdn_mw,
            'tdn_silage_dt':                self.tdn_silage_dt,
            'tdn_silage_dt_log':            self.tdn_silage_dt_log,
            'tdn_rumput_dt':                self.tdn_rumput_dt,
            'tdn_rumput_dt_log':            self.tdn_rumput_dt_log,
            'silage_x_rumput_tdn':          self.tdn_rumput_over_mw_dt * self.tdn_slobber_over_mw_dt,
            'tdn_slobber_dt':               self.tdn_slobber_dt,
            'tdn_new_concentrat_dt':        self.tdn_new_concentrat_dt,
            'tdn_ricehay_dt':               self.tdn_ricehay_dt,
            'tdn_SP2A_dt':                  self.tdn_sp2a_dt,
            'tdn_SP2B_dt':                  self.tdn_sp2b_dt,
            'tdn_slobber_dt_log':           self.tdn_slobber_dt_log,
            'total_tdn_greens_over_mw':     self.total_tdn_greens_over_mw,
            'tdn_greens_dt':                self.tdn_greens_dt,
            'tdn_greens_dt_log':            np.log(self.tdn_greens_dt) if self.tdn_greens_dt > 0 else np.nan,
            'total_tdn_greens_over_mw_dt':  self.total_tdn_greens_over_mw_dt,
            'tdn_silage_over_mw_dt':        self.tdn_silage_over_mw_dt,
            'tdn_rumput_over_mw_dt':        self.tdn_rumput_over_mw_dt,
            'tdn_slobber_over_mw_dt':       self.tdn_slobber_over_mw_dt,
            'tdn_SP2A_over_mw_dt':          self.tdn_sp2a_over_mw_dt,
            'tdn_ricehay_over_mw_dt':       self.tdn_ricehay_over_mw_dt,
            'tdn_tahu_dt':                  self.tdn_tahu_dt,
            'tdn_SMG_dt':                   self.tdn_SMG_dt,
            'r_tdn_silage_dt':              self.r_tdn_silage_dt,
            'r_tdn_rumput_dt':              self.r_tdn_rumput_dt,
            'r_tdn_slobber_dt':             self.r_tdn_slobber_dt,
            'r_tdn_ricehay_dt':             self.r_tdn_ricehay_dt,
            'r_tdn_SP2A_dt':                self.r_tdn_sp2a_dt,
            'tdn_silage_dt_2':              self.tdn_silage_dt ** 2,
            'tdn_rumput_dt_2':              self.tdn_rumput_dt ** 2,
            'tdn_slobber_dt_2':             self.tdn_slobber_dt ** 2,
            'tdn_silage_over_mw':           self.tdn_silage_over_mw,
            'tdn_rumput_over_mw':           self.tdn_rumput_over_mw,
            'silage_rumput_mw':             self.tdn_silage_over_mw * self.tdn_rumput_over_mw,
            'silage_rumput_dt_r_dmi':       (self.tdn_silage * self.tdn_rumput) / self.day_diff,
            'silage_rumput':                self.tdn_silage * self.tdn_rumput,
            'greens_slobber_mw':            (self.total_tdn_greens_over_mw / self.tdn_slobber_over_mw
                                             if self.tdn_slobber_over_mw != 0 else 0),
            'greens_slobber_mw_2':          (self.total_tdn_greens_over_mw / self.tdn_slobber_over_mw
                                             if self.tdn_slobber_over_mw != 0 else 0),
            'tdn_slobber_over_mw':          self.tdn_slobber_over_mw,
            'tdn_slobber_over_mw_2':        self.tdn_slobber_over_mw ** 2,
            'total_tdn_1_dt_dmi':           self.total_tdn_1_dt_dmi,
            'total_tdn_2_dt_dmi':           self.total_tdn_2_dt_dmi,
            'total_tdn_3_dt_dmi':           self.total_tdn_3_dt_dmi,
            'total_tdn_3_dt':               self.total_tdn_3_dt,

            'silage_dm_squared':            self.silage_dm_squared,
            'grass_dm_squared':             self.grass_dm_squared,
            'slobber_dm_squared':           self.slobber_dm_squared,
            'green_dm_squared':             self.green_dm_squared,
            'total_tdn_squared':            self.total_tdn_squared,
            'total_tdn_squared_ddmi':       self.total_tdn_squared / self.dm_total,
            'total_tdn_3_ddmi':             (self.total_tdn ** 3) / self.dm_total,
            'FeedRatio_squared':            self.feed_ratio_squared,
            'per_slobber_dm_squared':       self.per_slobber_dm_squared,
            'per_green_dm_squared':         self.per_green_dm_squared,
            'feed_cost_squared':            self.feed_cost_squared,
            'feed_cost_per_dm_squared':     self.feed_cost_per_dm_squared,
            'tdn_silage_over_mw_dt_2':      self.tdn_silage_over_mw_dt ** 2,
            'tdn_rumput_over_mw_dt_2':      self.tdn_rumput_over_mw_dt ** 2,
            'tdn_slobber_over_mw_dt_2':     self.tdn_slobber_over_mw_dt ** 2,
            'tdn_silage_dt_2_3':            self.tdn_silage_dt  / ((self.total_tdn_dt) ** 2 / 3),
            'tdn_rumput_dt_2_3':            self.tdn_rumput_dt  / ((self.total_tdn_dt) ** 2 / 3),
            'tdn_slobber_dt_2_3':           self.tdn_slobber_dt / ((self.total_tdn_dt) ** 2 / 3),
            'tdn_concentrats_dt_2_3':       self.tdn_concentrats_dt / ((self.total_tdn_dt) ** 2 / 3),
            'per_tdn_silage':               self.tdn_silage      / self.total_tdn,
            'per_tdn_rumput':               self.tdn_rumput      / self.total_tdn,
            'per_tdn_slobber':              self.tdn_slobber     / self.total_tdn,
            'per_tdn_concentrates':         self.tdn_concentrats / self.total_tdn,
            'tdn_silage_dt_r_dmi':          (self.tdn_silage_dt  * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_silage_dt_sdmi':           (self._tdn('Silase Jagung')  * self.dm_silage_ratio  * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_ricehay_dt_sdmi':          (self._tdn('Rice Hay')       * self.dm_ricehay_ratio * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_rumput_log_r_dmi':         (self.tdn_rumput_dt          * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_rumput_log':               np.log(self.tdn_rumput_dt) if self.tdn_rumput_dt > 0 else 0,
            'tdn_rumput_dt_r_dmi':          (self.tdn_rumput_log         * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_rumput_dt_sdmi':           (self._tdn('Rumput')         * self.dm_rumput_ratio  * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_rumput_dt_r_dmi_log':      np.log(self.tdn_rumput_dt * (self.dm_total) * self.tdn_ratio_r_3) if self.tdn_rumput_dt > 0 else 0,
            'tdn_silage_dt_r_dmi_log':      np.log(self.tdn_silage_dt * self.dm_total) if self.tdn_silage_dt > 0 else 0,
            'tdn_slobber_dt_r_dmi':         (self.tdn_slobber_dt         * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_slobber_dt_sdmi':          (self._tdn('Slobber Mix')    * self.dm_slobber_ratio * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_SP2A_dt_r_dmi':            (self._tdn('SP2A Mix')       * self.dm_SP2A_mix_ratio* (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_concentrats_dt':           self.tdn_concentrats_dt,
            'tdn_concentrats_dt_r_dmi':     self.tdn_concentrats_dt,
            'tdn_concentrats_dt_sdmi':      (self.tdn_concentrats_ratio * self.dm_concentrats_ratio) * ((self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_concentrats_2_dt':         self.tdn_concentrats_2_dt,
            'tdn_concentrats':              self.tdn_concentrats,
            'tdn_concentrats_dt_3_r_dmi':   (self.tdn_concentrats_dt ** 3) * ((self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_ricehay_dt_r_dmi':         (self._tdn('Rice Hay')       * self.dm_ricehay_ratio * (self.dm_total) ** (4/3)) / self.tdn_ratio_r_3,
            'tdn_silage_dt_r_dmi_mw':       self.tdn_silage_dt    * self.dm_total,
            'tdn_rumput_dt_r_dmi_mw':       self.tdn_rumput_dt    * self.dm_total,
            'tdn_slobber_dt_r_dmi_mw':      self.tdn_slobber_dt   * self.dm_total,
            'tdn_tahu_dt_r_dmi_mw':         self.tdn_ampas_tahu_dt* self.dm_total,
            'tdn_SP2A_dt_r_dmi_mw':         self.tdn_sp2a_dt      * self.dm_total,
            'tdn_SMG_dt_r_dmi_mw':          self.tdn_SMG_dt       * self.dm_total,
            'tdn_SP2B_dt_r_dmi_mw':         self.tdn_sp2b_dt      * self.dm_total,
            'tdn_ricehay_dt_r_dmi_mw':      self.tdn_ricehay_dt   * self.dm_total,
            'tdn_greens_dt_r_dmi_mw':       self.tdn_greens_dt    * self.dm_total,
            'total_tdn_dt_2':               self.total_tdn_dt_squared,
            'total_tdn_dt_3':               self.total_tdn_dt ** 3,
            'total_tdn_mw_dt_3':            self.total_tdn_3_dt,
            'total_tdn_mw_dt_2':            self.total_tdn_mw_dt_squared,
            'total_tdn_mw_3':               self.total_tdn_3,
            'total_tdn_mw_2':               self.total_tdn_2,

            'per_slobber_dm_dmi':               self.per_slobber_dm_dmi,
            'tdn_silage_over_mw_dt_ddmi':       self.tdn_silage_over_mw_dt  / self.dm_total_dt,
            'tdn_rumput_over_mw_dt_ddmi':       self.tdn_rumput_over_mw_dt  / self.dm_total_dt,
            'tdn_slobber_over_mw_dt_ddmi':      self.tdn_slobber_over_mw_dt / self.dm_total_dt,
            'tdn_silage_ddmi':                  self.tdn_silage  / self.dm_total,
            'tdn_rumput_ddmi':                  self.tdn_rumput  / self.dm_total,
            'tdn_slobber_ddmi':                 self.tdn_slobber / self.dm_total,
            'total_tdn_2_ddmi':                 (self.total_tdn ** 2) / self.dm_total,
            'total_dmi':                        self.dm_total,
            'weight_ddmi':                      (self.dm_total / self.day_diff) / self.avg_weight,
            'weight_ddmi_ddmi':                 1 / (self.day_diff * self.avg_weight),
            'day_diff_ddmi':                    self.day_diff / self.dm_total,
            'day_diff_2_ddmi':                  (self.day_diff ** 2) / self.dm_total,
            'total_dmi_2':                      self.dm_total ** 2,
            '1_total_dmi':                      1 / self.dm_total if self.tdn_rumput != 0 else 0,
            '1_total_dmi_dt':                   1 / (self.dm_total / self.day_diff),
            'total_dmi_dt':                     self.dm_total / self.day_diff,
            'total_dmi_dt_2':                   (self.dm_total / self.day_diff) ** 2,
            'r_total_dmi_dt':                   (self.dm_total / self.day_diff) ** 0.5,
            'total_dmi_dw':                     self.dm_total / self.avg_weight,
            'total_dmi_log':                    np.log(self.dm_total),
            'total_dmi_log_dmi':                np.log(self.dm_total) * self.dm_total,
            'total_dmi_log_dt':                 np.log(self.dm_total / self.day_diff),
            'total_dmi_log_dw':                 np.log(self.dm_total / self.avg_weight),

            'silage_dm_log':                    self.silage_dm_log,
            'grass_dm_log':                     self.grass_dm_log,
            'slobber_dm_log':                   self.slobber_dm_log,
            'green_dm_log':                     self.green_dm_log,
            'total_tdn_log':                    self.total_tdn_log,
            'FeedRatio_log':                    self.feed_ratio_log,
            'per_slobber_dm_log':               self.per_slobber_dm_log,
            'per_green_dm_log':                 self.per_green_dm_log,
            'feed_cost_log':                    self.feed_cost_log,
            'feed_cost_per_dm_log':             self.feed_cost_per_dm_log,
            'tdn_slobber_over_mw_dt_log':       np.log(self.tdn_slobber_over_mw_dt) if self.tdn_slobber_over_mw_dt > 0 else 0,
            'total_tdn_dt_log':                 self.total_tdn_dt_log,
            'total_tdn_mw_dt_log':              self.total_tdn_mw_dt_log,
            'tdn_silage_log':                   self.tdn_silage_log,
            'tdn_rumput_log':                   self.tdn_rumput_log,
            '1_over_tdn_rumput':                self.over_tdn_rumput,
            'tdn_slobber_log':                  self.tdn_slobber_log,
        }

    def get_all_features(self):
        if not self.has_required_feeds:
            return None
        features = self.get_dmi_features()
        features.update(self.get_feed_composition_features())
        features.update(self.get_nutrition_summary())
        return features
