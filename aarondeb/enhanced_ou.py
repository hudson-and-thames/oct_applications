import numpy as np
import pandas as pd
from mlfinlab.optimal_mean_reversion import OrnsteinUhlenbeck


class EnhancedOU(OrnsteinUhlenbeck):
    """
    Extension of the OrnsteinUhlenbeck class from the mlfinlab library. The new methods added
    consist of calculating the half life based of the selected time delta and calculating the
    count of mean crossings that happpen in a specific time delta.
    """

    def __init__(self):
        super().__init__()

    def actual_halflife(self) -> float:
        """
        Returns the half-life of the fitted OU process taking into consideration the specified
        delta_t variable.

        :return: (float) Half-life of the fitted OU process
        """
        return self.half_life() / self.delta_t

    def get_mean_crosses_by_timedelta(self, data, data_index: pd.Index, time_delta: str = 'Y') -> pd.DataFrame:
        """
        Returns a DataFrame with mean crossover counts that occur in the time_delta specified.

        :param data: (pd.DataFrame) Price data to construct a portfolio from
        :param data_index: (pd.DataTimeIndex) The time based index that the counts will be generated from
        :param time_delta: (str) Time delta that is going to be used in the resampling process
        :return: (pd.DataFrame) Counts of mean crosses by time_delta(in default case 'yearly') chosen 
        """

        if len(data.shape) == 1:  # If using portfolio prices
            portfolio = data
        elif data.shape[1] == 2:  # If using two assets prices
            # Checking the data type before preprocessing
            if isinstance(data, pd.DataFrame):
                # Transforming pd.Dataframe into a numpy array
                data = data.to_numpy().transpose()
            else:
                data = data.transpose()
            # Creating a portfolio with previously found optimal ratio
            portfolio = self.portfolio_from_prices(
                prices=data, b_variable=self.B_value)
        else:
            raise Exception("The number of dimensions for input data is incorrect. "
                            "Please provide a 1 or 2-dimensional array or dataframe.")

        centered_series = portfolio - self.theta

        cross_over_indices = np.where(np.diff(np.sign(centered_series)))[0]

        cross_overs_dates = data_index[cross_over_indices]

        cross_overs_by_delta = cross_overs_dates.to_frame().resample(time_delta).count()

        cross_overs_by_delta.columns = ['counts']

        return cross_overs_by_delta
