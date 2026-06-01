data {
  int<lower=1> N;
  int<lower=1> n_animals;

  array[N] int<lower=1>         animal_id;
  vector[N]                     days_gap;
  vector[N]                     next_gap;
  array[N] int<lower=0,upper=1> is_first;
  array[N] int<lower=0,upper=1> is_last;
  array[N] int<lower=1>         next_idx;

  vector[N] true_weight;

  vector[N] obs_tdn_silage;
  vector[N] obs_tdn_tahu;
  vector[N] obs_tdn_SP2A;
  vector[N] obs_tdn_SP2B;
  vector[N] obs_tdn_SMG;
  vector[N] obs_tdn_rumput;

  vector[N] hasBEF_dt;
  vector[N] hormone_effect_dt;
  vector[N] gotDewormed_dt;
  vector[N] gotHNMVaccination_dt;
}

transformed data {
  vector[N] metabolic_weight;
  for (n in 1:N) {
    metabolic_weight[n] = pow(fmax(true_weight[n], 1.0), 0.75);
  }
}

parameters {
  // --- TDN to Mcal conversion factors ---
  real<lower=0> mcal_per_kg_silage;
  real<lower=0> mcal_per_kg_tahu;
  real<lower=0> mcal_per_kg_SP2A;
  real<lower=0> mcal_per_kg_SP2B;
  real<lower=0> mcal_per_kg_SMG;
  real<lower=0> mcal_per_kg_rumput;

  // --- feed-specific DMI interaction effects ---
  real gamma_dmi;

  // --- maintenance baseline + modifiers ---
  real<lower=0> beta_metabolic_weight;
  real<lower=0> beta_BEF;
  real<lower=0> beta_hormone;
  real<lower=0> beta_dewormed;
  real<lower=0> beta_vaccination;

  // --- animal-specific random effects (maintenance) ---        // <<< NEW
  vector[n_animals] u_animal;                                   // <<< NEW
  real<lower=1e-6>  sigma_u_animal;                             // <<< NEW

  // --- animal-specific random effects (partition) ---
  vector[n_animals] alpha_animal;
  real<lower=1e-6> sigma_animal;

  // --- positive-energy partitioning parameters ---
  real          alpha_partition;
  real          gamma_hormone;
  real<lower=0> gamma_NEg;
  real<lower=1e-6> NEg_half;

  // --- energy densities ---
  real<lower=1> energy_per_kg_muscle;
  real<lower=1> energy_per_kg_fat;

  // --- day difference effect ---
  real day_diff_beta;

  // --- residual ---
  real<lower=1e-6> sigma_adg;
}

transformed parameters {
  // --- total DMI calculation ---
  vector[N] total_dmi;
  for (n in 1:N) {
    total_dmi[n] = obs_tdn_silage[n]
                 + obs_tdn_tahu[n]
                 + obs_tdn_SP2A[n]
                 + obs_tdn_SP2B[n]
                 + obs_tdn_SMG[n]
                 + obs_tdn_rumput[n];
  }

  // --- total energy consumed with feed-DMI interactions (Mcal/day) ---
  vector[N] Ntotal;
  for (n in 1:N) {
    real dmi_total = total_dmi[n];

    Ntotal[n] = (mcal_per_kg_silage * obs_tdn_silage[n]
              + mcal_per_kg_tahu * obs_tdn_tahu[n]
              + mcal_per_kg_SP2A * obs_tdn_SP2A[n]
              + mcal_per_kg_SP2B * obs_tdn_SP2B[n]
              + mcal_per_kg_SMG * obs_tdn_SMG[n]
              + mcal_per_kg_rumput * obs_tdn_rumput[n])*dmi_total*gamma_dmi;
  }

  // --- maintenance energy (Mcal/day) ---
  vector[N] Nmaintenance;
  for (n in 1:N) {
    real event_effect =
          beta_BEF              * hasBEF_dt[n]
        + beta_hormone          * hormone_effect_dt[n]
        + beta_dewormed         * gotDewormed_dt[n]
        + beta_vaccination      * gotHNMVaccination_dt[n];

    Nmaintenance[n] =
        beta_metabolic_weight * metabolic_weight[n]
      + u_animal[animal_id[n]]                       
      + event_effect;
  }

  // --- retained energy balance after maintenance (can be positive or negative) ---
  vector[N] NE_balance;
  vector[N] NE_surplus;
  vector[N] NE_deficit;
  for (n in 1:N) {
    NE_balance[n] = Ntotal[n] - Nmaintenance[n];
    NE_surplus[n] = fmax(NE_balance[n], 0.0);
    NE_deficit[n] = fmax(-NE_balance[n], 0.0);
  }

  // --- positive regime: muscle fraction of gain ---
  vector[N] muscle_fraction_gain;
  for (n in 1:N) {
    real denom = NE_surplus[n] + NEg_half + 1e-8;
    real animal_effect = alpha_animal[animal_id[n]];
    real logit_p = alpha_partition
                 + animal_effect
                 + gamma_hormone * hormone_effect_dt[n]
                 - gamma_NEg * NE_surplus[n] / denom;

    muscle_fraction_gain[n] = inv_logit(logit_p);
  }

  // --- positive regime: deposited energy by tissue ---
  vector[N] NE_gain_muscle;
  vector[N] NE_gain_fat;
  for (n in 1:N) {
    NE_gain_muscle[n] = muscle_fraction_gain[n]       * NE_surplus[n];
    NE_gain_fat[n]    = (1 - muscle_fraction_gain[n]) * NE_surplus[n];
  }

  // --- convert gain energy to kg tissue ---
  vector[N] muscle_gain_kg;
  vector[N] fat_gain_kg;
  for (n in 1:N) {
    muscle_gain_kg[n] = NE_gain_muscle[n] / energy_per_kg_muscle;
    fat_gain_kg[n]    = NE_gain_fat[n]    / energy_per_kg_fat;
  }

  // --- negative regime: deficit covered by fat loss only ---
  vector[N] muscle_loss_kg;
  vector[N] fat_loss_kg;
  for (n in 1:N) {
    muscle_loss_kg[n] = 0.0;
    fat_loss_kg[n]    = NE_deficit[n] / energy_per_kg_fat;
  }

  // --- net predicted ADG ---
  vector[N] adg_predicted;
  for (n in 1:N) {
    adg_predicted[n] =
        muscle_gain_kg[n]
      + fat_gain_kg[n]
      - muscle_loss_kg[n]
      - fat_loss_kg[n]
      + day_diff_beta * days_gap[n];
  }
}

model {
  // --- TDN to Mcal conversion factor priors ---
  mcal_per_kg_silage ~ normal(1.5, 0.2);
  mcal_per_kg_tahu   ~ normal(1.6, 0.2);
  mcal_per_kg_SP2A   ~ normal(1.5, 0.2);
  mcal_per_kg_SP2B   ~ normal(1.5, 0.2);
  mcal_per_kg_SMG    ~ normal(1.4, 0.2);
  mcal_per_kg_rumput ~ normal(1.3, 0.2);

  // --- feed-DMI interaction priors ---
  gamma_dmi ~ normal(0, 0.1);

  // --- maintenance priors ---
  beta_metabolic_weight ~ normal(0.060, 0.05);
  beta_BEF              ~ normal(0,    0.2);
  beta_hormone          ~ normal(-0.1, 0.1);
  beta_dewormed         ~ normal(-0.1, 0.1);
  beta_vaccination      ~ normal(0.1,  0.1);

  // --- animal-specific maintenance random effects ---   // <<< NEW
  u_animal    ~ normal(-5, sigma_u_animal);               // <<< NEW
  sigma_u_animal ~ exponential(5);                       // <<< NEW/

  // --- animal random effects (partition) ---
  alpha_animal ~ normal(0, sigma_animal);
  sigma_animal ~ exponential(5);

  // --- positive-regime partition priors ---
  alpha_partition      ~ normal(0.85, 0.3);
  gamma_hormone        ~ normal(0.3,  0.2);
  gamma_NEg            ~ normal(1.0,  0.5);
  NEg_half             ~ normal(4.0,  1.0);

  // --- energy density priors ---
  energy_per_kg_muscle ~ normal(1.5, 0.3);
  energy_per_kg_fat    ~ normal(8.0, 0.3);

  // --- day difference effect prior ---
  day_diff_beta ~ normal(0, 0.01);

  // --- residual ---
  sigma_adg ~ exponential(1);

  // --- likelihood ---
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
  vector[N] muscle_kg;
  vector[N] fat_kg;
  vector[N] adg_predicted_save;
  vector[N] NEg;

  for (n in 1:N) {
    muscle_kg[n] = muscle_gain_kg[n] - muscle_loss_kg[n];
    fat_kg[n]    = fat_gain_kg[n]    - fat_loss_kg[n];

    NEg[n] = NE_balance[n];

    adg_predicted_save[n] = adg_predicted[n];

    if (is_last[n] == 1) {
      adg[n]              = not_a_number();
      adg_vs_predicted[n] = not_a_number();
    } else {
      adg[n] = (true_weight[next_idx[n]] - true_weight[n]) / days_gap[n];
      adg_vs_predicted[n] = adg[n] - adg_predicted[n];
    }
  }
}
