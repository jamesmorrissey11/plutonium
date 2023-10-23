import json
from typing import Dict, List

import streamlit as st
from snowflake.snowpark import Session
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSessionException


class SnowparkConnector:
    def __init__(
        self,
        username: str,
        role: str,
        warehouse: str,
        account: str,
        datbase: str,
        authenticator: str = "externalbrowser",
    ):
        self.connection_params = {
            "account": account,
            "user": username,
            "authenticator": authenticator,
            "role": role,
            "warehouse": warehouse,
            "database": datbase,
        }
        self.snowpark_session = self.create_snowpark_session()
        self.user_id = self.get_metadata("current_user")
        self.session_id = self.get_metadata("current_session")

    def create_snowpark_session(self):
        try:
            sess = get_active_session()
            return sess

        except SnowparkSessionException:
            snowpark_session = Session.builder.configs(self.connection_params).create()
            return snowpark_session

    def query(self, snowpark_session, query: str):
        return snowpark_session.sql(query).to_pandas().iloc[0, 0]

    def write_to_snowflake(
        self, column_names, row: List[str], table_name: str, snowpark_session
    ):
        dataframe = snowpark_session.create_dataframe([row], schema=column_names)
        dataframe.write.mode("append").save_as_table(table_name)


def log_row(connector, row: Dict[str, str], log_table: str):
    connector.write_to_snowflake(
        column_names=list(row.keys()),
        row=list(row.values()),
        table_name=log_table,
        snowpark_session=connector.snowpark_session,
    )
    st.success("Feedback submitted successfully!")
