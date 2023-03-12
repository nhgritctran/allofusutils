from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import os
import polars as pl


class Profiling:

    def __init__(self):
        pass

    def create_param_profile(self, participant_dx_period, param, threshold, profile_type="ratio"):
        """
        :param participant_dx_period:
        :param param:
        :param threshold:
        :param profile_type:
        :return:
        """

        required_cols = ["person_id", "zip3", "start_date", "end_date"]
        if not set(required_cols).issubset(participant_dx_period.columns):
            print(f"Input dataframe must have these columns: {required_cols}")
            return

        if profile_type == "ratio":
            profile_function = self.get_param_ratio

        jobs = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            for i in tqdm(range(len(participant_dx_period))):
                jobs.append(executor.submit(profile_function,
                                            participant_dx_period,
                                            param,
                                            "date",
                                            threshold,
                                            participant_dx_period[i, "start_date"],
                                            participant_dx_period[i, "end_date"],
                                            participant_dx_period[i, "zip3"],
                                            participant_dx_period[i, "person_id"]))
        result_dicts = [job.result() for job in jobs]

        param_ratio_df = pl.from_dicts(result_dicts, schema={"person_id": pl.Utf8,
                                                             f"{param}_ratio": pl.Float64})

        final_df = participant_dx_period.join(param_ratio_df, how="inner", on="person_id")

        return final_df

    @staticmethod
    def get_param_ratio(param_df, param_col, date_col, param_threshold,
                        start_date, end_date, zip3, person_id=None):
        """
        :param param_df: polars df contains data for param of interest
        :param param_col: column name of param
        :param date_col: column name of date
        :param param_threshold: threshold which is used for ratio of above threshold:total
        :param start_date: start date of param data
        :param end_date: end date of param data
        :param zip3: zip3 of site measured param
        :param person_id: defaults to None; person id of interest
        :return: param_ratio of count above threshold:total count if no person_id provide;
                 else {"person_id": person_id, "<param_col>_ratio": param_ratio}
        """

        param_by_zip3 = param_df.filter(pl.col("zip3") == zip3)

        if len(param_by_zip3) > 0:
            param_by_zip3_and_date = param_by_zip3.filter((pl.col(date_col) >= start_date) &
                                                          (pl.col(date_col) <= end_date))
            above_threshold_count = param_by_zip3_and_date.filter(pl.col(param_col) > param_threshold)
            if len(param_by_zip3_and_date) > 0:
                param_ratio = len(above_threshold_count) / len(param_by_zip3_and_date)
            else:
                param_ratio = np.nan
        else:
            param_ratio = np.nan

        if person_id:
            return {"person_id": person_id, f"{param_col}_ratio": param_ratio}
        else:
            return param_ratio
