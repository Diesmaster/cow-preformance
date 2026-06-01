data {
  int<lower=1> N;
  int<lower=1> n_animals;
  array[N] int<lower=1>         animal_id;
  vector[N]                     days_gap;
  array[N] int<lower=0,upper=1> is_last;
  array[N] int<lower=1>         next_idx;
  vector[N] true_weight;
  vector[N] obs_total_cp;
  vector[N] obs_total_cf;
  vector[N] obs_total_fat;
  vector[N] obs_total_betn;
  vector[N] obs_total_dmi;
  vector[N] hasBEF_dt;
  vector[N] hormone_effect_dt;
  vector[N] gotDewormed_dt;
  vector[N] gotHNMVaccination_dt;
}

transformed data {
  vector[N] metabolic_weight;
  vector[N] cp_ratio;
  vector[N] cf_ratio;
  for (n in 1:N) {
    metabolic_weight[n] = pow(fmax(true_weight[n], 1.0), 0.75);
    real safe_dmi = fmax(obs_total_dmi[n], 1e-6);
    cf_ratio[n] = fmin(obs_total_cf[n] / safe_dmi, 2.0);
    cp_ratio[n] = fmin(obs_total_cp[n] / safe_dmi, 2.0);
  }
}

parameters {
  real<lower=0> mcal_per_kg_betn;
  real<lower=0> mcal_per_kg_cp;
  real<lower=0> mcal_per_kg_fat;

  real<lower=0> beta_cf_ratio;
  real<lower=0> const_DE_mul;

  real<lower=0, upper=1.0> beta_metabolic_weight;
  real<lower=0> beta_BEF;
  real<lower=0> beta_hormone;
  real<lower=0> beta_dewormed;
  real<lower=0> beta_vaccination;

  // --- cow-specific maintenance random effects ---
  vector[n_animals] u_animal;
  real<lower=1e-6>  sigma_u_animal;

  // --- cow-specific partition random effects ---
  vector[n_animals] alpha_animal;
  real<lower=1e-6>  sigma_animal;

  real          alpha_partition;
  real          gamma_hormone;
  real          gamma_cp_partition;
  real<lower=0> gamma_NEg;
  real<lower=1e-6> NEg_half;

  real<lower=1> energy_per_kg_muscle;
  real<lower=1> energy_per_kg_fat;

  real          day_diff_beta;

  real<lower=1e-6> sigma_adg;
}

transformed parameters {
  vector[N] Ntotal;
  for (n in 1:N) {
    Ntotal[n] = (mcal_per_kg_betn * obs_total_betn[n]
              + mcal_per_kg_cp   * obs_total_cp[n]
              + mcal_per_kg_fat  * obs_total_fat[n]);
  }

  vector[N] Nmaintenance;
  for (n in 1:N) {
    real event_effect =
          beta_BEF         * hasBEF_dt[n]
        + beta_hormone     * hormone_effect_dt[n]
        + beta_dewormed    * gotDewormed_dt[n]
        + beta_vaccination * gotHNMVaccination_dt[n];
    Nmaintenance[n] =
        beta_metabolic_weight * metabolic_weight[n]
      + u_animal[animal_id[n]]
      + event_effect;
  }

  vector[N] NE_balance;
  vector[N] NE_surplus;
  vector[N] NE_deficit;
  for (n in 1:N) {
    NE_balance[n] = Ntotal[n] - Nmaintenance[n];
    NE_surplus[n] = fmax(NE_balance[n],  0.0);
    NE_deficit[n] = fmax(-NE_balance[n], 0.0);
  }

  vector[N] muscle_fraction_gain;
  for (n in 1:N) {
    real denom   = NE_surplus[n] + NEg_half + 1e-8;
    real logit_p = alpha_partition
                 + alpha_animal[animal_id[n]]
                 + gamma_hormone      * hormone_effect_dt[n]
                 + gamma_cp_partition * cp_ratio[n]
                 - gamma_NEg          * NE_surplus[n] / denom;
    muscle_fraction_gain[n] = inv_logit(logit_p);
  }

  vector[N] muscle_gain_kg;
  vector[N] fat_gain_kg;
  vector[N] fat_loss_kg;
  for (n in 1:N) {
    muscle_gain_kg[n] = muscle_fraction_gain[n]       * NE_surplus[n] / energy_per_kg_muscle;
    fat_gain_kg[n]    = (1 - muscle_fraction_gain[n]) * NE_surplus[n] / energy_per_kg_fat;
    fat_loss_kg[n]    = NE_deficit[n] / energy_per_kg_fat;
  }

  vector[N] adg_predicted;
  for (n in 1:N) {
    adg_predicted[n] = muscle_gain_kg[n] + fat_gain_kg[n] - fat_loss_kg[n]
                     + day_diff_beta * days_gap[n];
  }
}

model {
  mcal_per_kg_betn ~ normal(2.6, 0.1);
  mcal_per_kg_cp   ~ normal(0.5, 0.5);
  mcal_per_kg_fat  ~ normal(7.5, 0.5);

  beta_cf_ratio ~ normal(0, 0.2);
  const_DE_mul ~ normal(0.80, 0.1);
  beta_metabolic_weight ~ normal(0.050, 0.10);
  beta_BEF              ~ normal(0,     0.2);
  beta_hormone          ~ normal(-0.1,  0.1);
  beta_dewormed         ~ normal(-0.1,  0.1);
  beta_vaccination      ~ normal(0.1,   0.1);

  u_animal       ~ normal(-5.5, sigma_u_animal);
  sigma_u_animal ~ exponential(2);

  alpha_animal ~ normal(0, sigma_animal);
  sigma_animal ~ exponential(2);

  alpha_partition    ~ normal(0.85, 0.3);
  gamma_hormone      ~ normal(0.3,  0.2);
  gamma_cp_partition ~ normal(0.3,  0.2);
  gamma_NEg          ~ normal(1.3,  0.1);
  NEg_half           ~ normal(4.0,  1.0);

  energy_per_kg_muscle ~ normal(1.5, 0.3);
  energy_per_kg_fat    ~ normal(8.0, 0.3);

  day_diff_beta ~ normal(0, 0.01);

  sigma_adg ~ exponential(1);

  for (n in 1:N) {
    if (is_last[n] == 0) {
      real obs_adg = (true_weight[next_idx[n]] - true_weight[n]) / days_gap[n];
      obs_adg ~ normal(adg_predicted[n], sigma_adg);
    }
  }
}

generated quantities {
  vector[N] adg;
  vector[N] adg_vs_predicted;
  vector[N] adg_predicted_save;
  vector[N] NEg;

  for (n in 1:N) {
    adg_predicted_save[n] = adg_predicted[n];
    NEg[n]                = NE_balance[n];
    if (is_last[n] == 1) {
      adg[n]              = not_a_number();
      adg_vs_predicted[n] = not_a_number();
    } else {
      adg[n]              = (true_weight[next_idx[n]] - true_weight[n]) / days_gap[n];
      adg_vs_predicted[n] = adg[n] - adg_predicted[n];
    }
  }
}
