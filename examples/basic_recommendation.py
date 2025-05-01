"""This example showcases the basic recommendation engine functionality.

The script creates two sample student profiles, one with standard attributes and one
with additional attributes, initializes the basic recommendation engine, and prints the
recommendations for both profiles, including score components.

From the **backend** directory of this project, this example can be executed from the
command line via:

python ../examples/basic_recommendation.py
"""

# Standard Library
import os
import sys
from pathlib import Path

# Append the framework path. NB: This is required if this entry point is invoked from
# the command line. However, it is not necessary if it is imported from a pip install.
if __name__ == "__main__":
    PACKAGE_PATH_ROOT = str(Path(__file__).resolve())
    PACKAGE_PATH_SPLIT = PACKAGE_PATH_ROOT.split(os.path.join("examples"))
    PACKAGE_PATH = Path(PACKAGE_PATH_SPLIT[0]) / "backend" / "src"
    if PACKAGE_PATH not in sys.path:
        print(f"Appending '{PACKAGE_PATH}' to system path...")
        sys.path.append(str(PACKAGE_PATH))

# Package Library
from naukriwaala.recommendation.basic_recommendation import \
    BasicRecommendationEngine
from naukriwaala.recommendation.schemas import Gender, StudentProfile
from naukriwaala.utils.logging_ import initialize_logger

logger = initialize_logger()


def main() -> None:
    """Example script demonstrating the recommendation engine functionality."""

    # Create a sample student profile with standard attributes.
    student = StudentProfile(
        education_level="12th",
        gender=Gender.Male.value,
        minimum_stipend=10000.0,
        preferred_locations=["Bangalore", "Mumbai", "Delhi"],
        preferred_sectors=["Electronics", "IT"],
        specialization="Science",
    )

    # Create a student profile with additional attributes.
    student_dict = {
        "education_level": "ITI",
        "gender": Gender.Female.value,
        "minimum_stipend": 15000.0,
        "preferred_locations": ["Chennai", "Hyderabad"],
        "preferred_sectors": ["Automotive", "Manufacturing"],
        "specialization": "Electronics",
        # Additional attributes
        "languages": ["Tamil", "Telugu", "English"],
        "preferred_training_mode": "Hybrid",
        "skills": ["Python", "AutoCAD", "PCB Design"],
        "work_experience_months": 6,
    }
    student2 = StudentProfile(**student_dict)

    # Initialize the basic recommendation engine.
    engine = BasicRecommendationEngine()

    try:
        # Get recommendations with score components.
        recommendations = engine.get_recommendations(
            include_score_components=True, limit=5, student=student
        )

        # Print recommendations.
        logger.info("\nTop 5 Recommendations:")
        logger.info("=====================")

        for i, rec in enumerate(recommendations, 1):
            logger.info(f"\n{i}. {rec.opportunity_details['name']}")
            logger.info(f"   Establishment: {rec.opportunity_details['establishment']}")
            logger.info(
                f"   Stipend Range: ₹{rec.opportunity_details['stipend_from']}-{rec.opportunity_details['stipend_upto']}"
            )
            logger.info(f"   Overall Match Score: {rec.score:.2f}")

            if rec.score_components:
                logger.info("\n   Score Components:")
                for component, score in rec.score_components.items():
                    logger.info(f"   - {component.title()}: {score:.2f}")

        # Example with student 2 (with additional attributes).
        logger.info("\n\nRecommendations for Student 2:")
        logger.info("=============================")
        logger.info("Student Profile:")
        logger.info(f"- Education: {student2.education_level}")
        logger.info(f"- Specialization: {student2.specialization}")
        logger.info("- Additional Attributes:")
        for key, value in student2.additional_attributes.items():
            logger.info(f"  * {key}: {value}")

        recommendations2 = engine.get_recommendations(
            include_score_components=True, limit=3, student=student2
        )

        for i, rec in enumerate(recommendations2, 1):
            logger.info(f"\n{i}. {rec.opportunity_details['name']}")
            logger.info(f"   Match Score: {rec.score:.2f}")

            if rec.score_components:
                logger.info("   Score Components:")
                for component, score in rec.score_components.items():
                    logger.info(f"   - {component.title()}: {score:.2f}")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
