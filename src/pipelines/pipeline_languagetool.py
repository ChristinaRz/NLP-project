"""
Παραδοτέο 1Β
Pipeline 1 — Grammar-based reconstruction με LanguageTool.

Το script:
1) Φορτώνει τα δύο κείμενα (TEXT1, TEXT2) από το src.data.texts
2) Κάνει βασικη κανονικοποίηση
3) Ελέγχει το κείμενο με LanguageTool
4) Εκτυπώνει:
   - πόσα "matches" βρέθηκαν
   - μερικά rule_ids + μηνύματα (debug)
   - το τελικό διορθωμένο κείμενο

"""

import re
import language_tool_python
from src.data.texts import TEXT1, TEXT2


# αρχικοποίηση LanguageTool στα αγγλικά
tool = language_tool_python.LanguageTool("en-US")

#κανονικοποίηση κειμένου
def normalize(text: str) -> str:

    #εν΄ωνουμε new lines και περιττά κενά
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def reconstruct(text: str) -> str:

    text = normalize(text)
    #προτεινόμενες διορθώσεις
    matches = tool.check(text)
    #εφαρμογή
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected


def debug_check(text: str, label: str) -> None:
    text_n = normalize(text)
    matches = tool.check(text_n)

    print(f"\n--- {label} ---")
    print(f"Matches found: {len(matches)}")

    # εκτυπώνουμε μόνο τα πρώτα 12
    for m in matches[:12]:
        # μερικές εκδόσεις έχουν rule_id, άλλες ruleId
        rid = getattr(m, "rule_id", None) or getattr(m, "ruleId", None) or "N/A"
        print(f"- {rid}: {m.message}")

    print("\nCorrected output:\n")
    print(language_tool_python.utils.correct(text_n, matches))


if __name__ == "__main__":
    # εκτελεση  pipeline
    debug_check(TEXT1, "TEXT 1")
    debug_check(TEXT2, "TEXT 2")
