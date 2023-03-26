from google.cloud import bigquery

import polars as pl


class SocioEconomicStatus:

    def __init__(self, cdr, question_id_dict=None):
        self.cdr = cdr

        self.aou_ses = self.polar_gbq(f"SELECT * FROM {self.cdr}.ds_zip_code_socioeconomic")

        if not question_id_dict:
            self.question_id_dict = {"own_or_rent": 1585370,
                                     "education": 1585940,
                                     "employment_status": 1585952,
                                     "annual_household_income": 1585375}
        self.question_ids = tuple(self.question_id_dict.values())
        self.survey_query = f"SELECT * FROM {self.cdr}.ds_survey WHERE question_concept_id IN {self.question_ids}"
        self.survey_data = self.polar_gbq(self.survey_query)

        self.income_dict = {"Annual Income: less 10k": 1,
                            "Annual Income: 10k 25k": 2,
                            "Annual Income: 25k 35k": 3,
                            "Annual Income: 35k 50k": 4,
                            "Annual Income: 50k 75k": 5,
                            "Annual Income: 75k 100k": 6,
                            "Annual Income: 100k 150k": 7,
                            "Annual Income: 150k 200k": 8,
                            "Annual Income: more 200k": 9}
        self.income_brackets = {"Annual Income: less 10k": [0.00, 9999.99],
                                "Annual Income: 10k 25k": [10000.00, 24999.99],
                                "Annual Income: 25k 35k": [25000.00, 34999.99],
                                "Annual Income: 35k 50k": [35000.00, 49999.99],
                                "Annual Income: 50k 75k": [50000.00, 74999.99],
                                "Annual Income: 75k 100k": [75000.00, 99999.99],
                                "Annual Income: 100k 150k": [100000.00, 149999.99],
                                "Annual Income: 150k 200k": [150000.00, 199999.99],
                                "Annual Income: more 200k": [200000.00, 999999.99]}
        self.edu_dict = {"Highest Grade: Never Attended": 1,
                         "Highest Grade: One Through Four": 2,
                         "Highest Grade: Five Through Eight": 3,
                         "Highest Grade: Nine Through Eleven": 4,
                         "Highest Grade: Twelve Or GED": 5,
                         "Highest Grade: College One to Three": 6,
                         "Highest Grade: College Graduate": 7,
                         "Highest Grade: Advanced Degree": 8}
        self.home_dict = {"Current Home Own: Own": "home_own",
                          "Current Home Own: Rent": "home_rent"}
        # "Current Home Own: Other Arrangement" are those with zero in both above categories
        self.employment_dict = {"Employment Status: Employed For Wages": "employed",
                                "Employment Status: Homemaker": "homemaker",
                                "Employment Status: Out Of Work Less Than One": "unemployed_less_1yr",
                                "Employment Status: Out Of Work One Or More": "unemployed_more_1yr",
                                "Employment Status: Retired": "retired",
                                "Employment Status: Self Employed": "self_employed",
                                "Employment Status: Student": "student"}
        # "Employment Status: Unable To Work" are those with zero in all other categories

    @staticmethod
    def polar_gbq(query):
        """
        :param query: BigQuery query
        :return: polars dataframe
        """
        client = bigquery.Client()
        query_job = client.query(query)
        rows = query_job.result()
        df = pl.from_arrow(rows.to_arrow())

        return df

    @staticmethod
    def dummy_coding(data, col_name, lookup_dict):
        """
        create dummy variables for a categorical variable
        :param data: polars dataframe
        :param col_name: variable of interest
        :param lookup_dict: dict to map dummy variables
        :return: polars dataframe with new dummy columns
        """
        for k, v in lookup_dict.items():
            data = data.with_columns(pl.when(pl.col(col_name) == k)
                                     .then(1)
                                     .otherwise(0)
                                     .alias(v))

        return data

    def compare_with_median_income(self, data):
        """

        :param data:
        :return:
        """
        ses_data = self.aou_ses[["PERSON_ID", "MEDIAN_INCOME"]]

        # 2-step mapping
        for k, v in self.income_brackets.items():
            ses_data = ses_data.with_columns(pl.when((pl.col("MEDIAN_INCOME") >= min(v)) &
                                                     (pl.col("MEDIAN_INCOME") <= max(v)))
                                             .then(k)
                                             .alias("TEMP_COL"))
        for k, v in self.income_dict.items():
            ses_data = ses_data.with_columns(pl.when(pl.col("TEMP_COL") == k)
                                             .then(v)
                                             .alias("MEDIAN_INCOME_BRACKET"))
        ses_data = ses_data.drop("TEMP_COL").rename({"PERSON_ID": "person_id",
                                                     "MEDIAN_INCOME": "median_income",
                                                     "MEDIAN_INCOME_BRACKET": "median_income_bracket"})

        # compare income and generate
        data = data.join(ses_data, how="inner", on="person_id")
        data = data.with_columns(pl.when(pl.col("income") < pl.col("median_income"))
                                 .then(-1)
                                 .when(pl.col("income") > pl.col("median_income"))
                                 .then(1)
                                 .otherwise(0)
                                 .alias("compare_with_median_income"))
        data = data.drop("median_income_bracket")

        return data

    def parse_survey_data(self):
        """

        :return:
        """
        # filter out people without survey answer, e.g., skip, don't know, prefer not to answer
        no_answer_ids = self.survey_data.filter(pl.col("answer").str.contains("PMI"))["person_id"].unique().to_list()
        survey_data = self.survey_data.filter(~pl.col("person_id").is_in(no_answer_ids))

        # split survey data into separate data by question
        question_list = self.survey_data["question"].unique().to_list()
        survey_dict = {}
        for question in question_list:
            key_name = question.split(":")[0].split(" ")[0]
            survey_dict[key_name] = survey_data.filter(pl.col("question") == question)
            survey_dict[key_name] = survey_dict[key_name][["person_id", "answer"]]
            survey_dict[key_name] = survey_dict[key_name].rename({"answer": f"{key_name.lower()}_answer"})

        # code income data
        survey_dict["Income"] = survey_dict["Income"].with_columns(pl.col("income_answer").alias("income"))
        survey_dict["Income"] = survey_dict["Income"].with_columns(pl.col("income")
                                                                   .map_dict(self.income_dict, default=pl.first())
                                                                   .cast(pl.Int64))
        survey_dict["Income"] = self.compare_with_median_income(survey_dict["Income"])

        # code education data
        survey_dict["Education"] = survey_dict["Education"].with_columns(pl.col("education_answer").alias("education"))
        survey_dict["Education"] = survey_dict["Education"].with_columns(pl.col("education")
                                                                         .map_dict(self.edu_dict, default=pl.first())
                                                                         .cast(pl.Int64))

        # code home own data
        survey_dict["Home"] = self.dummy_coding(data=survey_dict["Home"],
                                                col_name="home_answer",
                                                lookup_dict=self.home_dict)

        # code employment data
        survey_dict["Employment"] = self.dummy_coding(data=survey_dict["Employment"],
                                                      col_name="employment_answer",
                                                      lookup_dict=self.employment_dict)

        # merge data
        data = survey_dict["Income"].join(survey_dict["Education"], how="inner", on="person_id")
        data = data.join(survey_dict["Home"], how="inner", on="person_id")
        data = data.join(survey_dict["Employment"], how="inner", on="person_id")

        return data
