from typing import Dict, Optional, Any


class PromptsEN:
    DIAGNOSIS = """
    Patient: 29-year-old female, Polish descent.
    Chief Complaint: Migratory joint pain (wrists and knees), persistent low-grade fever (38.2°C), and worsening "brain fog" over the last 3 weeks.
    Physical Exam: Malar rash (butterfly shape) across cheeks; 2+ pitting edema in both ankles; no signs of active infection.
    """

    LAB_RESULTS = """
    Category	Parameter	Result	Reference Range
    Immunology	ANA (HEp-2)	1:1280 (Speckled)	< 1:80 (Entry Criterion Met)
    Immunology	Anti-dsDNA	85 IU/mL	< 30 IU/mL
    Immunology	Complement C3 / C4	Low (42 / 4 mg/dL)	80-160 / 16-48 mg/dL
    Hematology	White Blood Cell Count	3.2 x 10³/µL	4.5–11.0 x 10³/µL (Leukopenia)
    Renal	Urine Protein/Creatinine	0.8 g/day	< 0.2 g/day (Proteinuria)
    """

