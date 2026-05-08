
// Kalman Filter Prototype Model — Single Subject, with Process Noise
// Key design:
//   log_r and log_q are the two parameters (both unconstrained).

//   r_value = exp(log_r): observation noise variance, how much variability is in a stimulus given its true category, 
//   i.e. how "spread out" each category is in feature space

//   q_value = exp(log_q): process noise / prototype drift rate, how fast the prototype drifts over time,
//    i.e. how much the agent expects categories to shift trial-to-trial

//   Kalman loop structure: Predict (add Q) → Decide → Update.
//   The full filter is in transformed parameters so log_lik in generated
//   quantities reads p[i] directly without re-running the filter.

data {
  int<lower=1> ntrials;
  int<lower=1> nfeatures; // I.e. eyes on stalks, spots, teeth etc (we have 5 in total)
  int<lower=2> ncategories; // Amount of categories , i.e. 4
  array[ntrials] int<lower=1, upper=4> cat_true; // true  category of the alien on trial i (1, 2, 3 or 4)
  array[ntrials] int<lower=1, upper=4> y; // agent's response on trial i, this is what we are trying to predict
  array[ntrials, nfeatures] real obs; // A matrix of size ntrials x nfeatures (that meaning obs[trial x] = [1,0,0,1,0] fx)

  array[ncategories] vector[nfeatures] initial_mu; // The start prototype for each category, an array of 4 vectors of length 5 (nfeatures) equaling to one value per feature
  real<lower=0> initial_sigma_diag; // initial uncertainty about each prototype, this will go into a diagonal covariance matrix below. Large values = very uncertain

// Hyperparams for the two free parameters we are estimating described above 
  real prior_logr_mean;
  real<lower=0> prior_logr_sd;
  real prior_logq_mean;
  real<lower=0> prior_logq_sd;
}

// Unconstrained log-scale parameters, Stan samples these values freely
parameters {
  real log_r;
  real log_q; // A high q_value means the participant thinks categories are non-stationary and updates aggressively. 
              // A low q_value means they think categories are stable and update slowly.
}

// Its in the transformed parameters the kalman filter is applied 
transformed parameters {
  //start with transforming the unconstrained params into positive numbers only:
  real<lower=0> r_value = exp(log_r); 
  real<lower=0> q_value = exp(log_q);

  // Create an aray for storing the predicted repsonse probability p for each trial. 
  // p[i] is a probability vector over 4 categories (simplex ensures they sum to 1)
  array[ntrials] simplex[ncategories] p;
  
  // Local variables are declared below inside the {}:
  {
    array[ncategories] vector[nfeatures] mu; //create an array of 4 vectors of length 5 (nfeatures), to store mu in 
    array[ncategories] matrix[nfeatures, nfeatures] sigma; //create an array of 4 5x5 matrices to store sigma in
    
    // Initialize all prototypes and covariances
    for (c in 1:ncategories) {
      
      // Begin with defining the same intital prototype means for each category as passed in the data chunk, these will be updated trial by trial
      mu[c]    = initial_mu[c]; 
      
      // Create prototype uncertainty covariance matrix, one for each category, using the initial uncertainty value passed in the data chunk
      // The diag_matrix() function creates a diagonal matrix (initial_sigma_diag on the diagonal and 0s everywhere else), which assumes indepence between features 
      sigma[c] = diag_matrix(rep_vector(initial_sigma_diag, nfeatures));
    }
      
    // Create a diagonal matrix, this time with the r_value
    // This means all features have the same observation noise, and noise is independent across features.
    matrix[nfeatures, nfeatures] r_matrix = diag_matrix(rep_vector(r_value, nfeatures));
    
    // Create another diagonal matrix, this time with the q_value, same implications as described above
    matrix[nfeatures, nfeatures] q_matrix = diag_matrix(rep_vector(q_value, nfeatures));
    
    // Lastly create a diagonal identity matrix, needed later in the covariance update formula
    matrix[nfeatures, nfeatures] I_mat = diag_matrix(rep_vector(1.0, nfeatures));

// Now we loop through each trial
    for (i in 1:ntrials) {
      vector[nfeatures] x = to_vector(obs[i]); //Convert the row in obs[i] into a Stan vector x, needed for matrix maths apparently 

      // ── Prediction step: add process noise to all categories, this increases uncertainty before seeing the stimulus
      for (c in 1:ncategories)
        sigma[c] = sigma[c] + q_matrix;
        
      // ── Decision ──────────────────────────────────────────────────────
      vector[ncategories] log_probs; // Collect log-likelihood under each of the 4 categories
      for (c in 1:ncategories) {
        matrix[nfeatures, nfeatures] cov_c = sigma[c] + r_matrix; // Add prototype uncertainty (sigma) and observation noise (r_matrix) together to create a total covariance matrix for each category.
        log_probs[c] = multi_normal_lpdf(x | mu[c], cov_c); // log-likelihood of the stimulus under each category's Gaussian distribution. Ie. how probable is the stimulus given the category's prototype and spread?
      }
      // Softmax over 4 categories instead of 2
      p[i] = softmax(log_probs); // Softmax gives a 4-vector summing to 1
  
      // ── Update (only the true category) ───────────────────────────────
      int c_true = cat_true[i]; // Index directly into the true category array, then that prototype is updated
      vector[nfeatures] innov                    = x - mu[c_true]; //prediciton error (x-mu_prev in course notes)
      matrix[nfeatures, nfeatures] S             = sigma[c_true] + r_matrix; //combine prototype uncertainty (sigma) and observation noise (r_matrix) to create a total uncertainty matrix of the predicted observation
      matrix[nfeatures, nfeatures] K             = mdivide_right_spd(sigma[c_true], S); //Calculate the Kalmain Gain (solves for K = sigma_cat1 · S⁻¹), K~1 = trust the new observation a lot, update strongly, and K~0 = trust your prior more, update weakly
      matrix[nfeatures, nfeatures] IK            = I_mat - K; //For simplifaction purposes, used in the Joseph form below where (I-K) is used multiple times
      mu[c_true]    = mu[c_true] + K * innov; // Update the prototype mean, equalling to "updating your guess" from course notes
      sigma[c_true] = IK * sigma[c_true] * IK' + K * r_matrix * K'; //Update the prototype uncertainty using the Joseph form, called "Updating Your Confidence" in course notes
      sigma[c_true] = 0.5 * (sigma[c_true] + sigma[c_true]'); //Force symmetry by averaging the matrix with its own transpose.
      }
    }
  } 

model {
  // Add log prior probabilities for log_r and log_q, these are gaussian distributions generated from the values passed in the data chunk
  target += normal_lpdf(log_r | prior_logr_mean, prior_logr_sd);
  target += normal_lpdf(log_q | prior_logq_mean, prior_logq_sd);
  
  for (i in 1:ntrials)
    target += categorical_lpmf(y[i] | p[i]); // Stan will push log_r and log_q toward values that make p[i] close to y[i] across trials
}



generated quantities {
  vector[ntrials] log_lik; //create a vector of per-trial log likelihoods
  real lprior; // create a scalar for the total log prior.
  
  // For leave-one-out cross validation (LOO):
  for (i in 1:ntrials)
    log_lik[i] = categorical_lpmf(y[i] | p[i]); // For each trial, compute the log probability of the agent's actual response given the model's predicted probability p[i].
  
  // For prior/posterior plots, saves the total log prior probability of the sampled parameter values
  lprior = normal_lpdf(log_r | prior_logr_mean, prior_logr_sd) +
           normal_lpdf(log_q | prior_logq_mean, prior_logq_sd);
           
  real logr_prior = normal_rng(prior_logr_mean, prior_logr_sd);
  real logq_prior = normal_rng(prior_logq_mean, prior_logq_sd);
}

