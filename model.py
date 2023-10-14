import pyro
import pyro.distributions as dist
import torch

# Define the model
def ultimate_loss_model(w,
                        d,
                        premium,
                        reported_loss,
                        paid_loss,
                        reported_counts,
                        closed_counts):
    w_, d_ = w.shape[0], d.shape[0]
    
    # Prior ultimates
    prior_log_loss_ratio = pyro.sample('prior_log_loss_ratio',
                                       dist.Normal(torch.zeros(w_), torch.ones(w_)))
    prior_log_claim_frequency = pyro.sample('prior_log_claim_frequency',
                                            dist.Normal(torch.zeros(w_), torch.ones(w_)))
    
    # log(Prior ultimate loss) = prior log loss ratio + log(premium)
    log_prior_ultimate_loss = prior_log_loss_ratio + torch.log(torch.tensor(premium))
    prior_ultimate_loss = log_prior_ultimate_loss.exp()
    prior_ultimate_loss_sd = prior_ultimate_loss.sqrt()

    log_prior_ultimate_counts = prior_log_claim_frequency + torch.log(torch.tensor(premium))
    prior_ultimate_counts = log_prior_ultimate_counts.exp()
    prior_ultimate_counts_sd = prior_ultimate_counts.sqrt()

    # # Tweedie power parameters - range between 1 and 3
    # power_ult_loss = pyro.sample('power_ult_loss',
    #                              dist.Uniform(1, 3))
    # power_ult_counts = pyro.sample('power_ult_counts',
    #                                dist.Uniform(1, 3))
    # power_ult_severity = pyro.sample('power_ult_severity',
    #                                  dist.Uniform(1, 3))

    # Priors
    ultimate_loss = pyro.sample('ultimate_loss',
                                dist.Gamma(torch.power(prior_ultimate_loss, 2) / torch.power(prior_ultimate_loss_sd, 2),
                                            prior_ultimate_loss / torch.power(prior_ultimate_loss_sd, 2)))
    ultimate_counts = pyro.sample('ultimate_counts',
                                  dist.Gamma(torch.power(prior_ultimate_counts, 2) / torch.power(prior_ultimate_counts_sd, 2),
                                            prior_ultimate_counts / torch.power(prior_ultimate_counts_sd, 2)))
    
    # Severity ultimate - ultimate loss / ultimate counts
    ultimate_severity = pyro.sample('ultimate_severity',
                                    dist.LogGamma(torch.power(ultimate_loss, 2) / torch.power(ultimate_counts, 2),
                                                    ultimate_loss / torch.power(ultimate_counts, 2)))

    # Development patterns - usually range between 0 and 1, but can be greater than 1
    # if claims adjusters overestimate the ultimate loss and then reduce the estimate
    # over time
    dev_reported_loss = pyro.sample('dev_reported_loss',
                                    dist.Beta(2 * torch.ones(d_), 2 * torch.ones(d_)))
    dev_paid_loss = pyro.sample('dev_paid_loss',
                                dist.Beta(2 * torch.ones(d_), 2 * torch.ones(d_)))
    dev_reported_counts = pyro.sample('dev_reported_counts',
                                      dist.Beta(2 * torch.ones(d_), 2 * torch.ones(d_)))
    dev_closed_counts = pyro.sample('dev_closed_counts',
                                    dist.Beta(2 * torch.ones(d_), 2 * torch.ones(d_)))
    dev_paid_reported_loss = pyro.sample('dev_paid_reported_loss',
                                         dist.Beta(2 * torch.ones(d_), 2 * torch.ones(d_)))
    dev_closed_reported_counts = pyro.sample('dev_closed_reported_counts',
                                             dist.Beta(2 * torch.ones(d_), 2 * torch.ones(d_)))

    # Calculate derived means for quantities
    reported_loss_mean = ultimate_loss * dev_reported_loss
    paid_loss_mean = ultimate_loss * dev_paid_loss
    reported_counts_mean = ultimate_counts * dev_reported_counts
    closed_counts_mean = ultimate_counts * dev_closed_counts
    paid_reported_loss_mean = 1 * dev_paid_reported_loss
    closed_reported_counts_mean = 1 * dev_closed_reported_counts

    # Calculate derived standard deviations for quantities
    reported_loss_std = torch.sqrt(reported_loss_mean)
    paid_loss_std = torch.sqrt(paid_loss_mean)
    reported_counts_std = torch.sqrt(reported_counts_mean)
    closed_counts_std = torch.sqrt(closed_counts_mean)
    paid_reported_loss_std = torch.sqrt(paid_reported_loss_mean)
    closed_reported_counts_std = torch.sqrt(closed_reported_counts_mean)

    # Likelihoods
    obs_reported_loss = pyro.sample('obs_reported_loss',
                                    dist.Gamma(torch.power(reported_loss_mean, 2) / torch.power(reported_loss_std, 2),
                                                  reported_loss_mean / torch.power(reported_loss_std, 2)),
                                    obs=reported_loss)
    obs_paid_loss = pyro.sample('obs_paid_loss',
                                dist.Gamma(torch.power(paid_loss_mean, 2) / torch.power(paid_loss_std, 2),
                                              paid_loss_mean / torch.power(paid_loss_std, 2)),
                                obs=paid_loss)
    obs_reported_counts = pyro.sample('obs_reported_counts',
                                      dist.Gamma(torch.power(reported_counts_mean, 2) / torch.power(reported_counts_std, 2),
                                                    reported_counts_mean / torch.power(reported_counts_std, 2)),
                                      obs=reported_counts)
    obs_closed_counts = pyro.sample('obs_closed_counts',
                                    dist.Gamma(torch.power(closed_counts_mean, 2) / torch.power(closed_counts_std, 2),
                                                    closed_counts_mean / torch.power(closed_counts_std, 2)),
                                    obs=closed_counts)

# TODO: Add inference and validation steps

def run_model(model=None, num_samples=1000, warmup_steps=500):
    from pyro.infer import MCMC, NUTS
    # Initialize NUTS sampler
    nuts_kernel = NUTS(model)

    # Run MCMC
    mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc.run()

    # Show summary
    mcmc.summary()

    # Return the MCMC samples
    return mcmc