# Scenariusz Konsylium (Workflow)
# 
# Inicjacja: Przedstawienie danych pacjentki (Raport Radiologiczny i Patomorfologiczny).
# Runda 1 (Diagnostyka): Patomorfolog i Radiolog oceniają kompletność danych.
# Runda 2 (Strategia): Onkolog Kliniczny i Chirurg spierają się o kolejność (Chemia vs Operacja).
# Runda 3 (Dopełnienie): Radioterapeuta określa zakres leczenia regionalnego.
# Konkluzja: Wyznaczenie koordynatora i planu w karcie DiLO.
#
# TODO:
# - zaimplementowac model Doctor response (pydantic) do strukturyzacji odpowiedzi - diagnoza, rekomendacja, braki, adresat
# - niestety struktura xPath rozni sie w zaleznosci od modelu, trzeba bedzie to ujednolicic aby latwo zaciagac odpowiedzi
# - przetlumaczyc prompty oraz scenariusz z konsulium_1 na angielski


import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from prompts.prompts import Prompts

import os
from dotenv import load_dotenv

# Roles
ROLES = [
    "Radiolog",
    "Patomorfolog",
    "Chirurg Onkolog",
    "Onkolog Kliniczny",
    "Radioterapeuta"
]

# Read case data
def read_case_data(file_path) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# Initialize Selenium
def init_selenium() -> webdriver.Firefox:
    options = webdriver.FirefoxOptions()
    # options.add_argument("--headless")  # Uncomment for headless
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()), options=options)
    return driver

# Login to GitHub (placeholder, user needs to log in manually or provide credentials)
def login_github(driver: webdriver.Firefox) -> None:
    driver.get("https://github.com/login")
    
    # try to read credentials from environment variables
    load_dotenv()
    user = os.getenv("GITHUB_USER")
    pwd = os.getenv("GITHUB_PASSWORD")
    if user and pwd:
        # locate fields and submit
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, "//*[@id='login_field']")))
        driver.find_element(By.XPATH, "//*[@id='login_field']").send_keys(user)
        driver.find_element(By.XPATH, "//*[@id='password']").send_keys(pwd)
        driver.find_element(By.XPATH, "/html/body/div[1]/div[4]/main/div/div[2]/form/div[3]/input").click()
        # optionally wait for successful login by checking profile icon
        #WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "summary[aria-label=\"View profile and more\"]")))
    else:
        # fallback to manual login
        input("Please log in to GitHub and press Enter...")


# Scrape models from marketplace
def scrape_models() -> list:
    # Placeholder: hardcoded models for testing
    models = [
        {"name": "gpt-4-1-mini", "link": "https://github.com/marketplace/models/azure-openai/gpt-4-1-mini/playground"},#"https://github.com/marketplace/models/azure-openai/gpt-5/playground"},
        {"name": "MAI-DS-R1", "link": "https://github.com/marketplace/models/azureml/MAI-DS-R1/playground"},
        {"name": "Grok 3", "link": "https://github.com/marketplace/models/azureml-xai/grok-3/playground"},
        {"name": "Llama 3.1 405B", "link": "https://github.com/marketplace/models/azureml-meta/Meta-Llama-3-1-405B-Instruct/playground"},
        {"name": "Mistral Medium", "link": "https://github.com/marketplace/models/azureml-mistral/mistral-medium-2505/playground"}
    ]
    return models


def get_role_models(models: list) -> dict:
    selected = models
    role_model = dict(zip(ROLES, selected))
    return role_model

# Construct prompt for a role
def construct_prompt(role: str, case_data: str, previous_responses: dict) -> str:
    prompt = f"Jesteś światowej klasy {role.lower()} specjalizującym się w zaawansowanych rakach piersi. Uczestniczysz w konsylium medycznym.\n\n"
    prompt += f"Przypadek pacjenta:\n{case_data}\n\n"
    if previous_responses:
        prompt += "Poprzednie wypowiedzi członków konsylium:\n"
        for r, resp in previous_responses.items():
            prompt += f"{r}: {resp}\n"
    prompt += "\nPodziel się swoją opinią na temat przypadku, uwzględniając pytania/problemy omawiane w konsylium."
    return prompt

# Call model via Selenium (assuming playground interface)
def call_model(driver, model_link, prompt) -> str:
    driver.get(model_link)
    # Assume there's an input field and submit button
    input_field = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".PlaygroundChatInput-module__textarea__utifr")))  # Adjust
    input_field.send_keys(prompt)
    input_field.send_keys(Keys.RETURN)
    # Wait for response
    time.sleep(15)  # Adjust based on expected response time
    response_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.CSS_SELECTOR, ".tmp-py-3 > div:nth-child(1) > div:nth-child(2) > div:nth-child(2) > div:nth-child(1) > div:nth-child(1) > div:nth-child(1)")))  # Adjust
    response = response_element.text
    return response

# Main simulation with scenario rounds
def simulate_consortium(driver: webdriver.Firefox, role_model: dict, case_data: str) -> tuple[dict, list]:
    responses = {}
    logs = []

    prompts = Prompts()
    # define the workflow rounds
    rounds = [
        # initiation already covered implicitly by including case data in each prompt
        ["Patomorfolog", "Radiolog"],               # Runda 1 (Diagnostyka): Patomorfolog i Radiolog oceniają kompletność danych.
        ["Onkolog Kliniczny", "Chirurg Onkolog"],   # Runda 2 (Strategia): Onkolog Kliniczny i Chirurg spierają się o kolejność (Chemia vs Operacja).
        ["Radioterapeuta"]                          # Runda 3 (Dopełnienie): Radioterapeuta określa zakres leczenia regionalnego.
    ]

    for idx, round_roles in enumerate(rounds, start=1):
        print(f"--- Runda {idx}: {' & '.join(round_roles)} ---")
        for role in round_roles:
            model = role_model[role]
            prompt = prompts.get_prompt_based_on_role_x(role=role, additional_data=prompts.RADIOLOGY_REPORT)
            print(f"Role: {role}, Model: {model['name']}, Link: {model['link']}")
            print(f"Prompt: {prompt}")
            response = call_model(driver, model["link"], prompt)
            responses[role] = response
            logs.append({
                "role": role,
                "model": model["name"],
                "prompt": prompt,
                "response": response
            })
    # conclusion is just summarizing, we can optionally add a final note
    return responses, logs

# Generate report
def generate_report(responses: dict, logs: list) -> str:
    report = "Raport z Konsorcjum Medycznego\n\n"
    report += "Dyskusja:\n"
    for role, resp in responses.items():
        report += f"{role}: {resp}\n\n"
    report += "Logi szczegółowe:\n"
    for log in logs:
        report += f"Rola: {log['role']}, Model: {log['model']}\nPrompt: {log['prompt']}\nResponse: {log['response']}\n\n"
    return report

# Main function
def main() -> None:
    case_file = "konsulium_1"
    case_data = read_case_data(case_file)
    
    driver = init_selenium()
    try:
        login_github(driver) 
        models = scrape_models()  # Now hardcoded
        role_model = get_role_models(models)
        responses, logs = simulate_consortium(driver, role_model, case_data)
        report = generate_report(responses, logs)
        with open("report.txt", "w", encoding="utf-8") as f:
            f.write(report)
        print("Raport zapisany w report.txt")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()