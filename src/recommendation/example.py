from .engine import StudentProfile
from .implementations.basic_engine import BasicRecommendationEngine


def main():
    """Example script demonstrating the recommendation engine functionality.

    Creates two sample student profiles (one with standard attributes and one with additional attributes),
    initializes the basic recommendation engine, and prints recommendations for both profiles.
    """
    # Create a sample student profile with standard attributes
    student = StudentProfile(
        education_level="12th",
        specialization="Science",
        preferred_locations=["Bangalore", "Mumbai", "Delhi"],
        gender="Male",
        minimum_stipend=10000.0,
        preferred_sectors=["Electronics", "IT"],
    )

    # Create a student profile with additional attributes
    student_dict = {
        "education_level": "ITI",
        "specialization": "Electronics",
        "preferred_locations": ["Chennai", "Hyderabad"],
        "gender": "Female",
        "minimum_stipend": 15000.0,
        "preferred_sectors": ["Automotive", "Manufacturing"],
        # Additional attributes
        "languages": ["Tamil", "Telugu", "English"],
        "skills": ["Python", "AutoCAD", "PCB Design"],
        "work_experience_months": 6,
        "preferred_training_mode": "Hybrid",
    }
    student2 = StudentProfile.from_dict(student_dict)

    # Initialize basic recommendation engine
    engine = BasicRecommendationEngine()

    try:
        # Get recommendations with score components
        recommendations = engine.get_recommendations(
            student=student, limit=5, include_score_components=True
        )

        # Print recommendations
        print("\nTop 5 Recommendations:")
        print("=====================")

        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec.opportunity_details['name']}")
            print(f"   Establishment: {rec.opportunity_details['establishment']}")
            print(
                f"   Stipend Range: ₹{rec.opportunity_details['stipend_from']}-{rec.opportunity_details['stipend_upto']}"
            )
            print(f"   Overall Match Score: {rec.score:.2f}")

            if rec.score_components:
                print("\n   Score Components:")
                for component, score in rec.score_components.items():
                    print(f"   - {component.title()}: {score:.2f}")

        # Example with student2 (with additional attributes)
        print("\n\nRecommendations for Student 2:")
        print("=============================")
        print("Student Profile:")
        print(f"- Education: {student2.education_level}")
        print(f"- Specialization: {student2.specialization}")
        print("- Additional Attributes:")
        for key, value in student2.additional_attributes.items():
            print(f"  * {key}: {value}")

        recommendations2 = engine.get_recommendations(
            student=student2, limit=3, include_score_components=True
        )

        for i, rec in enumerate(recommendations2, 1):
            print(f"\n{i}. {rec.opportunity_details['name']}")
            print(f"   Match Score: {rec.score:.2f}")

            if rec.score_components:
                print("   Score Components:")
                for component, score in rec.score_components.items():
                    print(f"   - {component.title()}: {score:.2f}")

    finally:
        engine.close()


if __name__ == "__main__":
    main()
