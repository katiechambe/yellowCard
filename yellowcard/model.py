# Example: https://github.com/adrn/joaquin/blob/main/joaquin/joaquin.py

# TODO: imports!

# TODO: implement a function to evaluate the log-Normal probability, given a
# value x, mean, and stddev or variance
def ln_normal():
    pass


class TimingArgumentModel:

    def __init__(self, ):
        # TODO: allow passing in the data (distance, proper motion, RV to M31)
        # and uncertainties on the observed position/velocity of M31
        # TODO: define self._param_info dictionary to store names of parameters
        # (a, e, eta, ...etc)
        pass

    def ln_likelihood(self, par_dict):
        # TODO: call observables class to compute x, y, vx, vy given the
        # Keplerian elements / parameters
        # TODO: construct a LocalGroupHalocentric instance with x, y, vx, vy
        # (z=vz=0) and transform to ICRS
        # TODO: compute log-Normal probability for each model predicted quantity
        # (distance, proper motion, rv to M31)
        pass

    def ln_prior(self, par_dict):
        # TODO: need to discuss
        return 0.

    def ln_posterior(self, par_dict):
        # TODO: call ln_likelihood and ln_prior and add the values
        pass

    def __call__(self, par_arr):
        par_dict = self.unpack_pars(par_arr)
        return self.ln_posterior(par_dict)


# Defining __call__ makes this possible:
# model = TimingArgumentModel()
# model([1., 5., 0., 1.])
