import pandas as pd

S3_PREFIX = "s3://chaturai-bucket/rwf_contracts/"

# load contracts data ########################################
contracts = pd.read_excel(
    S3_PREFIX + "Candidates_10_April_2025_11_31_PM.xlsx",
    storage_options=dict(profile="chaturai"),
)

contracts.naps_date.min(), contracts.naps_date.max()

contracts.describe()

"""
- Candidate marks = 0.0 for everyone
- Candidate trade = empty for everyone
- is_active = 1.0 for everyone
- disability = 0.0

Anomalies to check

no_of_opportunities_applied == 4629 ?
"""


pii_columns = [
    "name",
    "parent_name",
    "parent_relation",
    "guardian_name",
    "guardian_relation",
    "aadhar_no",
    "mobile",
]
zero_info_columns = [
    "candidate_marks",
    "candidate_trade",
    "isactive",
    "disability",
    "created_on",
    "created_user",
]
existing_columns = contracts.columns.tolist()
columns_to_drop = set(pii_columns + zero_info_columns).intersection(
    set(existing_columns)
)
contracts = contracts.drop(columns=columns_to_drop)

contracts.to_csv("./data/contracts_tmp.csv", index=False)
contracts = pd.read_csv("./data/contracts_tmp.csv")
# ["naps_code", "naps_date", "gender", "disability", "category", "address", "no_of_opportunity_invites_received", "no_of_opportunities_applied", "district_id", "district_name", "statename", "candidate_trade"]


# Load students data ############################################
# Note: Confirmed that the students data is the same as contracts data
# and that the contracts data is more up to date than the students data

students = pd.read_excel(
    S3_PREFIX + "Candidates_10_April_2025_11_31_PM.xlsx",
    storage_options=dict(profile="chaturai"),
)
students.info()

students = students.drop(columns=pii_columns)
zero_info_columns = [
    "candidate_marks",
    "candidate_trade",
    "isactive",
    "disability",
    "created_on",
    "created_user",
]
students = students.drop(columns=zero_info_columns)

students.shape, contracts.shape
set(contracts.columns) - set(students.columns)
# This is an empty set !!!

(
    contracts.sort_values("naps_code") == students.sort_values("naps_code")
).sum() / students.shape[0]

students.to_csv("./data/students_tmp.csv", index=False)
students = pd.read_csv("./data/students_tmp.csv")

contracts.naps_date.unique()
contracts.apprenticeship_status.unique()

# Useful columns
# Filter for apprenticeship_status == \
# ["Approved", "approved", "cand_signed", "Cand_signed", \
# "completed", "Verified", "Novated"]
# Potentially useful status for rejected candidates == \
# ["rejected", "Termination_approved", "termination_approved", \
# "Cancelled", "Rejected", "Expired", "Cand_Rejected"]
# Gender, category, address, no_of_opportunity_invites_received,
# no_of_opportunities_applied, district_id, district_name,
# statename, institute_of_vocational_training,

# Load opportunities data ############################################
opportunities = pd.read_excel(
    S3_PREFIX + "Opportunities List.xlsx",
    storage_options=dict(profile="chaturai"),
)

opportunities.info()
opportunities.merge(
    students, left_on="Sr. No.", right_on="naps_code", how="inner"
).shape

opportunities_ids = opportunities["Vacancy ID"].tolist()

# Join contracts with opportunities
opp_id_list_str = ", ".join([f"'{str(i)}'" for i in opportunities_ids])
# No column to join on?

opportunities["Registration Type"].value_counts()
opportunities["Establishment Type"].value_counts()
opportunities["Industry Type"].value_counts()
opportunities["Trade Type (Designated/Optional)"].value_counts()
