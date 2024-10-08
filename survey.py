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

        self.income_dict = {"Annual Income: less 10k": 1,
                            "Annual Income: 10k 25k": 2,
                            "Annual Income: 25k 35k": 3,
                            "Annual Income: 35k 50k": 4,
                            "Annual Income: 50k 75k": 5,
                            "Annual Income: 75k 100k": 6,
                            "Annual Income: 100k 150k": 7,
                            "Annual Income: 150k 200k": 8,
                            "Annual Income: more 200k": 9}
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
        self.smoking_dict = {"Smoke Frequency: Every Day": "smoking_every_day",
                             "Smoke Frequency: Some Days": "smoking_some_days"}
        # "Not At All" are those with zero in all other categories

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
        convert area median income to equivalent income bracket and then compare with participant's income bracket
        :param data:
        :return:
        """
        ses_data = self.aou_ses[["PERSON_ID", "MEDIAN_INCOME"]]

        # mapping median income to income brackets
        ses_data = ses_data.with_columns(pl.when((pl.col("MEDIAN_INCOME") >= 0.00) &
                                                 (pl.col("MEDIAN_INCOME") <= 9999.99))
                                         .then(1)
                                         .when((pl.col("MEDIAN_INCOME") >= 10000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 24999.99))
                                         .then(2)
                                         .when((pl.col("MEDIAN_INCOME") >= 25000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 34999.99))
                                         .then(3)
                                         .when((pl.col("MEDIAN_INCOME") >= 35000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 49999.99))
                                         .then(4)
                                         .when((pl.col("MEDIAN_INCOME") >= 50000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 74999.99))
                                         .then(5)
                                         .when((pl.col("MEDIAN_INCOME") >= 75000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 99999.99))
                                         .then(6)
                                         .when((pl.col("MEDIAN_INCOME") >= 100000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 149999.99))
                                         .then(7)
                                         .when((pl.col("MEDIAN_INCOME") >= 150000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 199999.99))
                                         .then(8)
                                         .when((pl.col("MEDIAN_INCOME") >= 200000.00) &
                                               (pl.col("MEDIAN_INCOME") <= 999999.99))
                                         .then(9)
                                         .alias("MEDIAN_INCOME_BRACKET"))
        ses_data = ses_data.rename({"PERSON_ID": "person_id",
                                    "MEDIAN_INCOME": "median_income",
                                    "MEDIAN_INCOME_BRACKET": "median_income_bracket"})

        # compare income and generate
        data = data.join(ses_data, how="inner", on="person_id")
        data = data.with_columns((pl.col("income_bracket") - pl.col("median_income_bracket"))
                                 .alias("compare_to_median_income"))
        # data = data.drop("median_income_bracket")

        return data

    def split_string(self, df, col, split_by, item_index):
        
        df = df.with_columns((pl.col(col).str.split(split_by).list[item_index]).alias(col))
        
        return df

    def parse_survey_data(self, smoking=False):
        """
        get survey data of certain questions
        :param smoking: defaults to False; if true, data on smoking frequency is added
        :return: polars dataframe with coded answers
        """
        if smoking:
            self.question_id_dict["smoking_frequency"] = 1585860
        question_ids = tuple(self.question_id_dict.values())

        survey_query = f"SELECT * FROM {self.cdr}.ds_survey WHERE question_concept_id IN {question_ids}"
        survey_data = self.polar_gbq(survey_query)

        # filter out people without survey answer, e.g., skip, don't know, prefer not to answer
        no_answer_ids = survey_data.filter(pl.col("answer").str.contains("PMI"))["person_id"].unique().to_list()
        survey_data = survey_data.filter(~pl.col("person_id").is_in(no_answer_ids))

        # split survey data into separate data by question
        question_list = survey_data["question"].unique().to_list()
        survey_dict = {}
        for question in question_list:
            key_name = question.split(":")[0].split(" ")[0]
            survey_dict[key_name] = survey_data.filter(pl.col("question") == question)
            survey_dict[key_name] = survey_dict[key_name][["person_id", "answer"]]
            survey_dict[key_name] = survey_dict[key_name].rename({"answer": f"{key_name.lower()}_answer"})

        # code income data
        survey_dict["Income"] = survey_dict["Income"].with_columns(pl.col("income_answer").alias("income_bracket"))
        survey_dict["Income"] = survey_dict["Income"].with_columns(pl.col("income_bracket")
                                                                   .replace(self.income_dict, default=pl.first())
                                                                   .cast(pl.Int64))
        survey_dict["Income"] = self.compare_with_median_income(survey_dict["Income"])

        # code education data
        survey_dict["Education"] = survey_dict["Education"].with_columns(pl.col("education_answer").alias("education_bracket"))
        survey_dict["Education"] = survey_dict["Education"].with_columns(pl.col("education_bracket")
                                                                         .replace(self.edu_dict, default=pl.first())
                                                                         .cast(pl.Int64))

        # code home own data
        survey_dict["Home"] = self.dummy_coding(data=survey_dict["Home"],
                                                col_name="home_answer",
                                                lookup_dict=self.home_dict)

        # code employment data
        survey_dict["Employment"] = self.dummy_coding(data=survey_dict["Employment"],
                                                      col_name="employment_answer",
                                                      lookup_dict=self.employment_dict)

        # code smoking data
        if smoking:
            survey_dict["Smoking"] = self.dummy_coding(data=survey_dict["Smoking"],
                                                       col_name="smoking_status",
                                                       lookup_dict=self.smoking_dict)

        # merge data
        data = survey_dict["Income"].join(survey_dict["Education"], how="inner", on="person_id")
        data = data.join(survey_dict["Home"], how="inner", on="person_id")
        data = data.join(survey_dict["Employment"], how="inner", on="person_id")
        if smoking:
            data = data.join(survey_dict["Smoking"], how="inner", on="person_id")

        data = self.split_string(df=data, col="income_answer", split_by=": ", item_index=1)
        data = self.split_string(df=data, col="education_answer", split_by=": ", item_index=1)
        data = self.split_string(df=data, col="home_answer", split_by=": ", item_index=1)
        data = self.split_string(df=data, col="employment_answer", split_by=": ", item_index=1)

        data = data.rename(
            {
                "income_answer": "annual income",
                "education_answer": "highest degree",
                "home_answer": "homeownership",
                "employment_answer": "employment status"
            }
        )
        
        return data
