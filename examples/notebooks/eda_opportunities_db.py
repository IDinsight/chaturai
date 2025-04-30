from sqlalchemy import create_engine, URL
from sqlalchemy import text
import pandas as pd
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to make backend imports work
backend_path = str(Path(__file__).parent.parent)
if backend_path not in sys.path:
    sys.path.append(backend_path)

from src.recommendation.engine import (
    Gender,
    Qualification,
    AcademicQualificationOrdinal,
    QualificationType,
)

db_url = URL.create(
    database="naukriwaala",
    drivername="postgresql+" + "psycopg2",
    host="localhost",
    password=os.getenv("POSTGRES_PASSWORD"),
    port=5432,
    username="postgres",
)
db_engine = create_engine(db_url)

sample_query = text("""SELECT * FROM opportunities LIMIT 10000""")

sample_data = pd.read_sql_query(sample_query, con=db_engine)

sample_data.info()
sample_data.gender_type.value_counts()
sample_data.course_data.iloc[0]
sample_data.course_data.iloc[0]["sector"]
sample_data.course_data.iloc[0]["minimum_qualification"]
sample_data.locations_data.iloc[12]

sample_data.locations_data.apply(len).value_counts()

"""
Data to collect from students

Hard criteria
1. gender (categorical) -> Filter
4. location (categorical first)
    - On WhatsApp we can ask if they want to work in their current \
location or somewhere else
    - Nearby, specific district(s), within state, other states
5. Qualification (categorical) ->

Soft criteria
3. trade (chips) -> opportunities.name (filter if present?)
0. stipend (float) -> hard or soft?
2. interest (free text) -> opportunities.name, \
 opportunities.short_description (check for duplicates), establishment info
6. Registration type (categorical) -> establishment.registration_type
      2	17
    huf	6
    ts	144
    la	16
    16	13
    boi	6
    soc	111
    other	372
    1	11
    psi	1198
    9	2
    4	9
    12	1
        97
    7	63
    partnership	764
    15	4
    5	28
<establishment related info like govenrment, size, ...>


Rank by
1. Stipend amount
2. cosine similarity of interests and naukri type

Note
- We'll exclude working_days since vast majority of opportunities are 6 days
"""

# Locations data collection with LLM ##########################################

S3_PREFIX = "s3://naukriwaala-bucket/rwf_contracts/"
locations = pd.read_csv(S3_PREFIX + "locations.csv")
locations.groupby("state_name").nunique()


def capitalize_first_letter(s: str) -> str:
    """
    Capitalize the first letter of a string
    """
    return s[0].upper() + s[1:]


def capitalize(s: str) -> str:
    """
    Capitalize the first letter of each word in a string
    """
    return " ".join([capitalize_first_letter(word) for word in s.split()])


locations_list_str = ""
for state_name, districts in locations.groupby("state_name"):
    locations_list_str += capitalize(state_name) + "\n"
    for district_name in districts["district_name"]:
        locations_list_str += f"- {capitalize(district_name)}\n"
    locations_list_str += "\n"

LOCATION_PROMPT = f"""
You are a location expert. Your task is to identify the location preference \
of a student based on their input. \
Your output should be a json following this pydantic model:

    class StateLocationPreference(BaseModel):
        state: str
        districts: list[str] | None  # must belong to the state

    class LocationPreference(BaseModel):
        state_preferences: list[StateLocationPreference]
        is_final: bool
        next_question: str | None

In the following cases, set is_final to False and provide a clarifying \
question in next_question in the language of the user: If the student provides
- a location name not in the following list of states and districts in India
- no location
- a district but no state
- a state but no district (this is okay, but double check that the student \
can work anywhere in the state)
Otherwise, set `is_final` to True and `next_question` to None.

The state and district names should come from the following list of states \
and districts in India:

{locations_list_str}
"""

from pydantic import BaseModel


class StateLocationPreference(BaseModel):
    state: str
    districts: list[str] | None  # must belong to the state


class LocationPreference(BaseModel):
    state_preferences: list[
        StateLocationPreference
    ]  # at least one state should be chosen
    is_final: bool
    next_question: str | None


from litellm import completion, embedding

# Rememeber to set LLM API key env var
messages = [
    {"role": "system", "content": LOCATION_PROMPT},
    {"role": "user", "content": "chatrapati sambhajnagar aur thane bhi theek ho jaega"},
]

result = (
    completion(
        model="gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        n=1,
        response_format={"type": "json_object"},
    )
    .choices[0]
    .message.content
)

print(result)

messages.append({"role": "assistant", "content": result})
messages.append(
    {
        "role": "user",
        "content": "??? sirf chhatrapati sambhajnagar aur thane me kaam karna",
    }
)

"""
Quirks of data
- Sub-district information would have to be extracted from the address field. \
    For the first version, let's just use district-level.
- In the DB, district is "chhatrapati sambhajinagar" but in SHRUG it corresponds \
    to "aurangabad"
"""


# Define filtering functions #############################################


def filter_gender(student_gender: Gender | None, opp_gender_type: str | None) -> bool:
    """
    Filter opportunities based on gender
    """

    if student_gender is None:
        return True
    else:
        return opp_gender_type is None or student_gender.value in opp_gender_type


def filter_location(
    student_location: LocationPreference,
    opp_locations: list[dict] | None,
) -> bool:
    """
    Filter opportunities based on location
    """
    for loc in opp_locations:
        district_name = loc["address"]["district_name"].lower()
        city = loc["address"]["city"].lower()
        state = loc["address"]["state_name"].lower()
        address = (
            loc["address"]["address_1"].lower()
            + " "
            + loc["address"]["address_2"].lower()
        )

        for pref in student_location.state_preferences:
            if pref.state.lower() == state:
                if pref.districts is None:
                    return True
                else:
                    for district in pref.districts:
                        if (
                            district.lower() == district_name
                            or district.lower() == city
                            or district.lower() in address
                        ):
                            return True
    return False


def filter_qualification(
    student_qualification: Qualification,
    opp_min_qualification: dict,
) -> bool:
    """
    Filter opportunities based on qualification
    """

    qualification_type = opp_min_qualification["minimum_qualification"]["type"].lower()
    qualification_value = opp_min_qualification["minimum_qualification"][
        "qualification_type"
    ]["title"].lower()

    if qualification_type != student_qualification.qualification_type.value:
        return False
    elif qualification_type == QualificationType.academic:
        return AcademicQualificationOrdinal.meets_minimum_requirement(
            student_qual=student_qualification.qualification,
            required_qual=qualification_value,
        )
    elif qualification_type == QualificationType.trained_under_scheme:
        return student_qualification.qualification == qualification_value
    else:
        return False


def filter_stipend(
    student_min_stipend: float | None,
    opp_min_stipend: float | None,
) -> bool:
    """
    Filter opportunities based on stipend
    """
    if student_min_stipend is None:
        return True
    elif opp_min_stipend is None:
        return False
    else:
        return student_min_stipend <= opp_min_stipend


result = embedding(
    model="text-embedding-ada-002",
    input="electrican",
).data[
    0
]["embedding"]

opp_trade = embedding(
    model="text-embedding-ada-002",
    input="Industrial Electrician",
).data[0]["embedding"]

# get cosine similarity
import numpy as np


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


cosine_similarity(result, opp_trade)


def filter_trade(
    student_trade: str | None,
    opp_trade: str | None,
) -> bool:
    """
    Filter opportunities based on trade
    """
    if student_trade is None:
        return True
    elif opp_trade is None:
        return False
    elif student_trade.lower() in opp_trade.lower():
        return True
    else:
        student_trade_embedding = embedding(
            model="text-embedding-ada-002",
            input=student_trade,
        ).data[0]["embedding"]

        opp_trade_embedding = embedding(
            model="text-embedding-ada-002",
            input=opp_trade,
        ).data[0]["embedding"]

        cosine_sim = cosine_similarity(student_trade_embedding, opp_trade_embedding)
        if cosine_sim > 0.8:
            return True
        else:
            return False


# TODO: Define ranking functions #############################################
