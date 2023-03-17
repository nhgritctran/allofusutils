from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np
import os
import polars as pl


class Profiling:

    def __init__(self):
        pass

    def create_epa_param_profile(self, participant_dx_period, epa_data, param_name,
                                 aqi_threshold, profile_type="aqi_ratio"):
        """
        :param participant_dx_period: participant df containing diagnosis period
        :param epa_data: EPA df of interest
        :param param_name: environmental parameter of interest
        :param aqi_threshold: aqi threshold
        :param profile_type: accepts "aqi_ratio"
        :return: original df with param aqi ratio added
        """

        required_cols = ["person_id", "zip3", "start_date", "end_date"]
        if not set(required_cols).issubset(participant_dx_period.columns):
            print(f"Input dataframe must have these columns: {required_cols}")
            return

        if profile_type == "aqi_ratio":
            profile_function = self.get_aqi

        jobs = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            for i in tqdm(range(len(participant_dx_period))):
                jobs.append(executor.submit(profile_function,
                                            epa_data,
                                            param_name,
                                            "date",
                                            aqi_threshold,
                                            participant_dx_period[i, "start_date"],
                                            participant_dx_period[i, "end_date"],
                                            participant_dx_period[i, "zip3"],
                                            participant_dx_period[i, "person_id"]))
        result_dicts = [job.result() for job in jobs]

        param_ratio_df = pl.from_dicts(result_dicts, schema={f"{param_name}_mean_aqi": pl.Float64,
                                                             f"{param_name}_aqi_sub25_days": pl.Float64,
                                                             f"{param_name}_aqi_sub50_days": pl.Float64,
                                                             f"{param_name}_aqi_sub75_days": pl.Float64,
                                                             f"{param_name}_aqi_sub100_days": pl.Float64,
                                                             f"{param_name}_aqi_sub150_days": pl.Float64,
                                                             f"{param_name}_total_measured_days": pl.Float64,
                                                             f"{param_name}_data_coverage": pl.Float64,
                                                             "person_id": pl.Utf8})

        final_df = participant_dx_period.join(param_ratio_df, how="inner", on="person_id")

        return final_df

    @staticmethod
    def get_aqi(param_df, param_name, date_col,
                start_date, end_date, zip3, person_id=None):
        """
        :param param_df: polars df contains data for param of interest
        :param param_name: name of param
        :param date_col: column name of date
        :param start_date: start date of param data
        :param end_date: end date of param data
        :param zip3: zip3 of site measured param
        :param person_id: defaults to None; person id of interest
        :return: new columns with AQI related data
        """

        param_by_zip3 = param_df.filter(pl.col("zip3") == zip3)

        aqi_dict = {f"{param_name}_mean_aqi": np.nan,
                    f"{param_name}_aqi_sub25_days": np.nan,
                    f"{param_name}_aqi_sub50_days": np.nan,
                    f"{param_name}_aqi_sub75_days": np.nan,
                    f"{param_name}_aqi_sub100_days": np.nan,
                    f"{param_name}_aqi_sub150_days": np.nan,
                    f"{param_name}_total_measured_days": np.nan,
                    f"{param_name}_data_coverage": np.nan}

        if len(param_by_zip3) > 0:
            param_by_zip3_and_date = param_by_zip3.filter((pl.col(date_col) >= start_date) &
                                                          (pl.col(date_col) <= end_date))
            # group by zip3 & date and get mean value
            param_by_zip3_and_date = param_by_zip3_and_date.groupby([date_col, "zip3"]).mean()
            # get rows where param value above threshold
            sub25days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 25))
            sub50days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 50))
            sub75days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 75))
            sub100days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 100))
            sub150days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 150))
            mean_aqi = param_by_zip3_and_date.groupby("zip3").mean()["aqi"][0, 0]

            if len(param_by_zip3_and_date) > 0:
                total_measured_days = len(param_by_zip3_and_date)
                dx_days = (end_date - start_date).days + 1
                data_coverage = total_measured_days / dx_days
                aqi_dict = {f"{param_name}_mean_aqi": mean_aqi,
                            f"{param_name}_aqi_sub25_days": sub25days,
                            f"{param_name}_aqi_sub50_days": sub50days,
                            f"{param_name}_aqi_sub75_days": sub75days,
                            f"{param_name}_aqi_sub100_days": sub100days,
                            f"{param_name}_aqi_sub150_days": sub150days,
                            f"{param_name}_total_measured_days": len(param_by_zip3_and_date),
                            f"{param_name}_data_coverage": data_coverage}

        if person_id:
            aqi_dict["person_id"] = person_id

        return aqi_dict
