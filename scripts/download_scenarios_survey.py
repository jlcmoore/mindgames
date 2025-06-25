"""
Downloads the Google Sheet with the scenarios in it. 
Run with the following so long as you have Google credentials downloaded in 
`gcp_credentials.json`

E.g. see:
https://developers.google.com/workspace/guides/create-credentials#service-account

`python scripts/download_scenarios_survey.py`
"""

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Define the scope
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# Add credentials to the account
creds = ServiceAccountCredentials.from_json_keyfile_name("gcp_credentials.json", scope)

# Authorize the clientsheet
client = gspread.authorize(creds)


# Function to clean DataFrame based on a column list
def clean_dataframe(df, columns_to_check):
    df_cleaned = df.dropna(subset=columns_to_check)
    df_cleaned = df_cleaned[(df_cleaned[columns_to_check] != "").all(axis=1)]
    return df_cleaned


# Function to process the 'attributes' columns into lists
def process_attributes_column(df, column_name):
    df[column_name] = df[column_name].apply(
        lambda x: [attr.strip() for attr in x.split(",")] if x else []
    )
    return df


# Get the instance of the Spreadsheet
sheet = client.open("scenarios_vague")

# Process scenarios sheet
scenarios_worksheet = sheet.get_worksheet(0)
scenarios_data = scenarios_worksheet.get_all_values()
scenarios_df = pd.DataFrame(scenarios_data[1:], columns=scenarios_data[0])

scenarios_columns = ["id", "cover_story", "persuader_role", "target_role", "attributes"]
scenarios_df_cleaned = clean_dataframe(scenarios_df, scenarios_columns)

scenarios_df_cleaned = process_attributes_column(scenarios_df_cleaned, "attributes")
scenarios_df_cleaned = scenarios_df_cleaned[
    scenarios_df_cleaned["attributes"].apply(len) >= 3
]

scenarios_df_cleaned[scenarios_columns].to_json(
    "src/data/scenarios.jsonl", orient="records", lines=True
)

# Process survey questions sheet
sheet = client.open("survey_questions")
survey_questions_worksheet = sheet.get_worksheet(0)
survey_data = survey_questions_worksheet.get_all_values()
survey_df = pd.DataFrame(survey_data[1:], columns=survey_data[0])

survey_columns = ["id", "statement", "supporting_attributes", "opposing_attributes"]
survey_df_cleaned = clean_dataframe(survey_df, survey_columns)

survey_df_cleaned = process_attributes_column(
    survey_df_cleaned, "supporting_attributes"
)
survey_df_cleaned = process_attributes_column(survey_df_cleaned, "opposing_attributes")


# Validate that all supporting and opposing attributes are in the valid attributes set
def validate_attributes(attributes, valid_attributes):
    for attr in attributes:
        if attr not in valid_attributes:
            raise ValueError(f"Invalid attribute found: {attr}")


# Extract unique attributes from scenarios
valid_attributes = set(
    attr for sublist in scenarios_df_cleaned["attributes"] for attr in sublist
)

survey_df_cleaned["supporting_attributes"].apply(
    lambda attrs: validate_attributes(attrs, valid_attributes)
)
survey_df_cleaned["opposing_attributes"].apply(
    lambda attrs: validate_attributes(attrs, valid_attributes)
)

survey_df_cleaned[survey_columns].to_json(
    "src/data/survey.jsonl", orient="records", lines=True
)
