from openai import OpenAI

client = OpenAI()

selected_model = "o4-mini"  # reasoning model

def analyze_case_multispecialist(description: str):
    """
    Analiza edukacyjna: model generuje ogólne obserwacje,
    potencjalne obszary wymagające dalszej oceny
    oraz perspektywę kilku specjalistów.
    Nie zawiera diagnoz ani planów leczenia.
    """

    specialists = [
        "Nephrologist", "Hematologist",
        "Vascular Surgeon", "Endovascular Interventionist",
    ]

    prompt = f"""
    You are an AI assistant generating **educational, medical-related** analysis.
    The user provided a medical case description.

    Your task:
    - provide diagnosis,
    - provide treatment plans,
    - recommend medications, doses, procedures, or interventions,
    - imply certainty or clinical authority,
    - provide most precise and case-oriented, high-level educational insights.

    Case description:
    {description}

    Generate:
    1. A neutral summary of the case.
    2. A list of general areas that may require further evaluation.
    3. A section with perspectives from these specialists:
       {", ".join(specialists)}
       Each specialist should describe:
       - what aspects they might pay attention to,
       - what general categories of tests or symptoms are relevant in their field,
       - what treatment plan including medications, procedures, or interventions should be implemented,
       - what uncertainties remain.
    """

    response = client.responses.create(
        model=selected_model,
        input=prompt,
        max_output_tokens=2048
    )

    # return response.output_text
    return response


# Przykład użycia:
case_description = """
A 41-year-old male with type-2 diabetes, hypertension, and IHD presented to the emergency department one week after ureteroscopy and laser lithotripsy for a right-sided mid-ureteric stone. 
    The patient had complaints of severe back pain, nausea, and vomiting. The pain was radiating to both lower limbs and was associated with significant weakness in both legs. 
    The patient reported no history of trauma, fever, dysuria, or hematuria. 
    The patient’s vital signs were as follows: heart rate 130/min, blood pressure 150/90 mmHg, O2 saturation 98%, respiratory rate 20/min, and temperature 36 C. 
    Both lower limbs were dusky in color; bilateral peripheral pulses of the femoral, popliteal, and dorsalis pedis arteries were absent. 
    An echocardiogram (ECG) showed sinus  tachycardia  with  evidence  of  previous  anterior ischemic changes. 
    At this time, lower limb ischemia was strongly suspected, and an urgent computed tomography angiography (CTA) of the chest, abdomen, and lower limbs was performed. 
    Laboratory results showed a normal CBC of Hb 15.3 g/dL (ref. value: 12–17.5 g/dL), WBC 4.6 x109/L (ref. value: 4-10 x109/L) and platelet count 287 x109/L (ref. value: 150-410  x109/L). 
    The  chemistry  panel  showed  normal electrolytes with mildly elevated creatinine 109 umol/L (ref. value: 44-115 umol/L). 
    Creatine phosphokinase (CPK) and lactate dehydrogenase (LDH) enzymes were markedly elevated. 
    CPK level was 58,155 IU/L (ref. value: 22-200 IU/L) while LDH level was 1,793 IU/L (ref. value 140-280 IU/L). 
    CTA showed that the distal part of the abdominal aorta was totally occluded, with a massive intramural thrombus that extended distally to both iliac arteries. 
    Further thrombosis was detected in both the renal and splenic arteries, resulting in a subtotal infarction of the right kidney, a segmental infarction of the left renal lower pole, and a wedge-shaped infarction of the mid-part of the spleen. 
    Notably, the scan did not reveal any evidence of dissection in any aortic division.
"""

analysis = analyze_case_multispecialist(case_description)
print(analysis)
print("-"*80)
print(analysis.output_text)
