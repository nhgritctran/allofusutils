from google.cloud import bigquery

import os
import polars as pl


class SocioEconomicStatus:
    def __init__(self, cdr, question_id_dict):
        self.cdr = cdr
        self.question_id_dict = question_id_dict

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

    def income_survey(self):
        pass
