from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import datetime
import numpy as np
import os
import polars as pl


class Profiling:

    def __init__(self):
        pass

    def create_epa_param_profile(self, participant_dx_period, epa_data, param_name, profile_type="aqi"):
        """
        :param participant_dx_period: participant df containing diagnosis period
        :param epa_data: EPA df of interest
        :param param_name: environmental parameter of interest
        :param profile_type: accepts "aqi"
        :return: original df with param aqi ratio added
        """

        required_cols = ["person_id", "zip3", "start_date", "end_date"]
        if not set(required_cols).issubset(participant_dx_period.columns):
            print(f"Input dataframe must have these columns: {required_cols}")
            return

        if profile_type == "aqi":
            profile_function = self.get_aqi

        jobs = []
        with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
            for i in tqdm(range(len(participant_dx_period))):
                jobs.append(executor.submit(profile_function,
                                            epa_data,
                                            param_name,
                                            "date",
                                            participant_dx_period[i, "start_date"],
                                            participant_dx_period[i, "end_date"],
                                            participant_dx_period[i, "zip3"],
                                            participant_dx_period[i, "person_id"]))
        result_dicts = [job.result() for job in jobs]

        param_ratio_df = pl.from_dicts(result_dicts, schema={f"{param_name}_all_time_mean_raw_value": pl.Float64,
                                                             f"{param_name}_all_time_mean_aqi": pl.Float64,
                                                             f"{param_name}_mean_raw_value": pl.Float64,
                                                             f"{param_name}_mean_aqi": pl.Float64,
                                                             f"{param_name}_aqi_0to25_days": pl.Float64,
                                                             f"{param_name}_aqi_26to50_days": pl.Float64,
                                                             f"{param_name}_aqi_51to75_days": pl.Float64,
                                                             f"{param_name}_aqi_76to100_days": pl.Float64,
                                                             f"{param_name}_aqi_101to150_days": pl.Float64,
                                                             f"{param_name}_aqi_151plus_days": pl.Float64,
                                                             f"{param_name}_measured_days_before_dx": pl.Float64,
                                                             f"{param_name}_total_measured_days": pl.Float64,
                                                             f"{param_name}_total_dx_days": pl.Float64,
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

        aqi_dict = {f"{param_name}_all_time_mean_raw_value": np.nan,
                    f"{param_name}_all_time_mean_aqi": np.nan,
                    f"{param_name}_mean_value": np.nan,
                    f"{param_name}_mean_aqi": np.nan,
                    f"{param_name}_aqi_0to25_days": np.nan,
                    f"{param_name}_aqi_26to50_days": np.nan,
                    f"{param_name}_aqi_51to75_days": np.nan,
                    f"{param_name}_aqi_76to100_days": np.nan,
                    f"{param_name}_aqi_101to150_days": np.nan,
                    f"{param_name}_aqi_151plus_days": np.nan,
                    f"{param_name}_measured_days_before_dx": np.nan,
                    f"{param_name}_total_measured_days": np.nan,
                    f"{param_name}_total_dx_days": np.nan,
                    f"{param_name}_data_coverage": np.nan}

        if len(param_by_zip3) > 0:

            # all time mean values
            all_time_mean_raw_value = np.nan
            if param_name != "aqi":
                all_time_mean_raw_value = param_by_zip3.groupby("zip3").mean()["arithmetic_mean"][0]
            all_time_mean_aqi = param_by_zip3.groupby("zip3").mean()["aqi"][0]

            # move start_date back 365 days
            # this to ensure measurement starts 1 year ahead
            # in case dx period is short, e.g., few days, there should still be enough measurement data
            dx_start_date = start_date
            start_date = start_date - datetime.timedelta(365)

            # filter data
            param_by_zip3_and_date = param_by_zip3.filter((pl.col(date_col) >= start_date) &
                                                          (pl.col(date_col) <= end_date))

            # group by zip3 & date and get mean value
            param_by_zip3_and_date = param_by_zip3_and_date.groupby([date_col, "zip3"]).mean()

            if len(param_by_zip3_and_date) > 0:

                # days by thresholds
                sub25days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 25))
                sub50days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 50))
                sub75days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 75))
                sub100days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 100))
                sub150days = len(param_by_zip3_and_date.filter(pl.col("aqi") <= 150))

                # days by bins
                total_measured_days = len(param_by_zip3_and_date)
                aqi26to50days = sub50days - sub25days
                aqi51to75days = sub75days - sub50days
                aqi76to100days = sub100days - sub75days
                aqi101to150days = sub150days - sub100days
                above150days = total_measured_days - sub150days

                # other stats
                mean_raw_value = np.nan
                if param_name != "aqi":
                    mean_raw_value = param_by_zip3_and_date.groupby("zip3").mean()["arithmetic_mean"][0]
                mean_aqi = param_by_zip3_and_date.groupby("zip3").mean()["aqi"][0]
                days_before_dx = len(param_by_zip3_and_date.filter(pl.col(date_col) <= dx_start_date))
                dx_days = (end_date - dx_start_date).days + 1
                data_coverage = total_measured_days / dx_days

                # put all together
                aqi_dict = {f"{param_name}_all_time_mean_raw_value": all_time_mean_raw_value,
                            f"{param_name}_all_time_mean_aqi": all_time_mean_aqi,
                            f"{param_name}_mean_raw_value": mean_raw_value,
                            f"{param_name}_mean_aqi": mean_aqi,
                            f"{param_name}_aqi_0to25_days": sub25days,
                            f"{param_name}_aqi_26to50_days": aqi26to50days,
                            f"{param_name}_aqi_51to75_days": aqi51to75days,
                            f"{param_name}_aqi_76to100_days": aqi76to100days,
                            f"{param_name}_aqi_101to150_days": aqi101to150days,
                            f"{param_name}_aqi_151plus_days": above150days,
                            f"{param_name}_measured_days_before_dx": days_before_dx,
                            f"{param_name}_total_measured_days": total_measured_days,
                            f"{param_name}_total_dx_days": dx_days,
                            f"{param_name}_data_coverage": data_coverage}

        if person_id:
            aqi_dict["person_id"] = person_id

        return aqi_dict
