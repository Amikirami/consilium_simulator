from typing import Dict, Optional, Any


class Prompts:
    PATHOMORPHOLOG_COMMENT = ""
    RADIOLOG_COMMENT = ""
    SPECJALISATION = ""

    SUBJECT = """
    Temat: Wielodyscyplinarna kwalifikacja do leczenia raka piersi (zaawansowany rak piersi)

    Omawiany Przypadek: Pacjentka: Kobieta, 55 lat, z niedawno zdiagnozowanym rakiem piersi (wynik histopatologiczny potwierdzający raka przewodowego).
    Problem: Guz jest duży, zlokalizowany blisko ściany klatki piersiowej, a badania obrazowe (MRI/PET-CT) sugerują potencjalne zajęcie węzłów chłonnych w okolicy nadobojczykowej.
    Cel konsylium: Ustalenie, czy najpierw przeprowadzić operację (chirurg), czy rozpocząć od chemioterapii w celu zmniejszenia guza (onkolog kliniczny), a następnie radioterapii. 
    """

    RADIOLOGY_REPORT = """
    Pacjentka: 55 lat. Lokalizacja: Lewa pierś, kwadrant górny zewnętrzny (UQO).

    Opis ogniska pierwotnego: Masa guzowata o wymiarach 52 x 48 x 45 mm. 
    Guz wykazuje cechy naciekania powięzi mięśnia piersiowego większego. 
    W badaniu MRI widoczna silna, niejednorodna kinetyka wzmocnienia kontrastowego typu wash-out.

    Węzły chłonne: Pakiet węzłów chłonnych pachy po stronie lewej (wymiar największego węzła 22 mm, z zatartą wnęką). 
    Dodatkowo w badaniu PET-CT ujawniono ognisko o podwyższonym metabolizmie glukozy (SUVmax 8.4) w okolicy nadobojczykowej lewej, co nasuwa podejrzenie zajęcia węzłów chłonnych grupy nadobojczykowej.
    """

    PATHOMORPHOLOGICAL_REPORT = """
    Pacjentka: 55 lat. Materiał: 1. Biopsja gruboigłowa guza piersi lewej (4 bioptaty).
    2. Biopsja cienkoigłowa (BAC) węzła chłonnego nadobojczykowego lewego.

    Rozpoznanie mikroskopowe:
        Typ: Rak przewodowy naciekający (NST), G3 (wysoki stopień złośliwości).
        Inwazja naczyń: Widoczna inwazja naczyń limfatycznych i krwionośnych w podścielisku (LVI+).
        Komponent in situ: Obecny komponent raka wewnątrzprzewodowego (DCIS) typu comedo (ok. 20% objętości).

    Profil Immunohistochemiczny (IHC):
        Receptory Estrogenowe (ER): Wyrazista reakcja jądrowa w 10% komórek nowotworowych (Słabo dodatni / Low positive).
        Receptory Progesteronowe (PR): 0% (Ujemny).
        HER2: Wynik 2+ (Niejednoznaczny).
        Ki-67 (Indeks proliferacyjny): 75% (Bardzo wysoki).

    Wynik BAC węzła nadobojczykowego:
    W rozmazie obecne nieliczne, rozproszone komórki o cechach atypii, podejrzane o przerzut raka piersi, jednak mała liczebność materiału nie pozwala na jednoznaczne potwierdzenie (kategoria C4 wg skali Bethesda).
    """

    PATOMORFOLOG = """
    Jesteś lekarzem patomorfologiem w zespole MDT (Multidisciplinary Team). 
    Twoim zadaniem jest analiza profilu biologicznego nowotworu na podstawie dostarczonych wyników biopsji i IHC.

    Skup się na: stopniu złośliwości (G), indeksie proliferacji (Ki-67) oraz statusie receptorów.

    Twoja rola: Wskaż, czy dane są wystarczające do podjęcia decyzji o leczeniu celowanym. Jeśli wyniki są niejednoznaczne (np. HER2 2+), domagaj się badań pogłębionych (FISH/ISH). Oceń ryzyko nawrotu na podstawie inwazji naczyń (LVI).

    Referencja: Analizuj dostarczony "Raport Patomorfologiczny".
    '{PATHOMORPHOLOGICAL_REPORT}'
    """

    RADIOLOG = """
    Jesteś lekarzem radiologiem specjalizującym się w diagnostyce obrazowej piersi. Twoim zadaniem jest ocena zaawansowania lokalnego i regionalnego (staging TNM).
    Skup się na: wielkości guza, jego relacji do ściany klatki piersiowej/mięśni oraz ocenie węzłów chłonnych (pachowych, nadobojczykowych, wewnątrzpiersiowych).
    Twoja rola: Określ, czy guz jest technicznie operacyjny w momencie diagnozy. Zinterpretuj wyniki PET-CT w korelacji z MRI.

    Referencja: Analizuj dostarczony "Raport Radiologiczny".
    '{RADIOLOGY_REPORT}'
    """

    ONKOLOG = """
    Jesteś onkologiem klinicznym. Odpowiadasz za leczenie systemowe (chemioterapia, hormonoterapia, leczenie celowane).

    Skup się na: wrażliwości nowotworu na leki w zależności od podtypu molekularnego.

    Twoja rola: Zaproponuj sekwencję leczenia. Czy pacjentka odniesie większą korzyść z chemioterapii neoadjuwantowej (przedoperacyjnej)? Jakie schematy lekowe zastosujesz (np. antracykliny, taksany, anty-HER2)?

    Referencja: Głównym punktem odniesienia jest dla Ciebie profil IHC z raportu patomorfologa.
    {PATHOMORPHOLOG_COMMENT}
    """

    CHIRURG = """
    Jesteś chirurgiem onkologiem. Twoim celem jest uzyskanie czystości onkologicznej (marginesy R0) przy zachowaniu jak najlepszej jakości życia pacjentki.

    Skup się na: technicznej możliwości wykonania zabiegu. Rozważ, czy lepsza będzie mastektomia, czy leczenie oszczędzające (BCT).

    Twoja rola: Oceń, czy potrzebujesz "zmniejszenia" guza przed operacją (downstaging). Określ zakres operacji węzłów chłonnych (biopsja węzła wartowniczego vs limfadenektomia).

    Referencja: Odnieś się do wymiarów guza z raportu radiologa 
    '{RADIOLOG_COMMENT}'

    i agresywności z raportu patomorfologa
    '{PATHOMORPHOLOG_COMMENT}'
    """

    RADIOTERAPEUTA = """
    Cel: Zaplanowanie leczenia uzupełniającego.

    Działaj jako radioterapeuta. Twój głos jest kluczowy w kontekście węzłów nadobojczykowych. 
    Jeśli to przerzut, pole naświetlania musi być znacznie szersze. 
    Wyjaśnij zespołowi, że niezależnie od wyniku operacji, pacjentka będzie wymagała radioterapii na ścianę klatki piersiowej i obszary węzłowe ze względu na cechę T3 i zajęcie węzłów (N+). 
    Jakie ryzyko powikłań widzisz przy tak rozległym naświetlaniu (np. obrzęk limfatyczny)?
    """

    RESPONSE_RULES = """
    Odpowiadaj wyłącznie jako {SPECJALISATION}. Twoja wypowiedź musi być krótka, techniczna i pozbawiona uprzejmości.

    Zasady odpowiedzi:

    Diagnoza: Max 2 konkretne wnioski z wyników.
    Rekomendacja: Jedno kluczowe działanie (co robimy?).
    Braki: Czego brakuje w dokumentacji (jeśli dotyczy)?
    Do kogo kierujesz swoje rekomendacje/pytanie (jeśli dotyczy)? (Patomorfolog, Radiolog, Radioterapeuta, Chirurg, Onkolog)

    Limit: Maksymalnie 150 słów. Używaj punktów zamiast pełnych zdań, jeśli to możliwe. 
    Nie powtarzaj faktów ogólnych, skup się na anomalii w badaniach (52 mm, G3, HER2 2+, N3?).
    """

    def get_prompt_based_on_role_x(self, role, additional_data=None):
        if role == "Patomorfolog":
            return self.PATOMORFOLOG.format(PATHOMORPHOLOG_COMMENT=additional_data) + self.RESPONSE_RULES.format(SPECJALISATION = "patomorfolog")
        elif role == "Radiolog":
            return self.RADIOLOG.format(RADIOLOGY_REPORT=additional_data) + self.RESPONSE_RULES.format(SPECJALISATION = "radiolog")
        elif role == "Onkolog Kliniczny":
            return self.ONKOLOG + self.RESPONSE_RULES.format(SPECJALISATION = "onkolog kliniczny")
        elif role == "Chirurg Onkolog":
            return self.CHIRURG + self.RESPONSE_RULES.format(SPECJALISATION = "chirurg onkolog")
        elif role == "Radioterapeuta":
            return self.RADIOTERAPEUTA + self.RESPONSE_RULES.format(SPECJALISATION = "radioterapeuta")
        else:
            raise ValueError(f"Nieznana rola: {role}")


    def get_prompt_based_on_role(self, role, add_data: Optional[Dict[str, Any]] = None) -> str:
        if add_data is None:
            add_data = {}
        if role == "Patomorfolog":
            print("get_prompt_based_on_role  ::", self.PATOMORFOLOG, "\n####"*4,
                  self.PATOMORFOLOG.format(PATHOMORPHOLOG_COMMENT=add_data.get("PATHOMORPHOLOG_COMMENT", ""),
                                           # PATHOMORPHOLOGICAL_REPORT=self.PATHOMORPHOLOGICAL_REPORT)
                                           PATHOMORPHOLOGICAL_REPORT=add_data.get("PATHOMORPHOLOGICAL_REPORT", ""))
                  )
            return self.PATOMORFOLOG.format(
                PATHOMORPHOLOG_COMMENT=add_data.get("PATHOMORPHOLOG_COMMENT", ""),
                PATHOMORPHOLOGICAL_REPORT=add_data.get("PATHOMORPHOLOGICAL_REPORT", "")
            ) + self.RESPONSE_RULES.format(SPECJALISATION = "patomorfolog")
        elif role == "Radiolog":
            return self.RADIOLOG.format(
                RADIOLOGY_REPORT=add_data.get("RADIOLOGY_REPORT", "")
            ) + self.RESPONSE_RULES.format(SPECJALISATION = "radiolog")
        elif role == "Onkolog Kliniczny":
            return self.ONKOLOG.format(
                PATHOMORPHOLOG_COMMENT=add_data.get("PATHOMORPHOLOG_COMMENT", "")
            ) + self.RESPONSE_RULES.format(
                SPECJALISATION = "onkolog kliniczny"
            )
        elif role == "Chirurg Onkolog":
            return self.CHIRURG.format(
                RADIOLOG_COMMENT=add_data.get("RADIOLOG_COMMENT", ""),
                PATHOMORPHOLOG_COMMENT=add_data.get("PATHOMORPHOLOG_COMMENT", "")
            ) + self.RESPONSE_RULES.format(
                SPECJALISATION = "chirurg onkolog",
                )
        elif role == "Radioterapeuta":
            return self.RADIOTERAPEUTA + self.RESPONSE_RULES.format(SPECJALISATION = "radioterapeuta")
        else:
            raise ValueError(f"Nieznana rola: {role}")
