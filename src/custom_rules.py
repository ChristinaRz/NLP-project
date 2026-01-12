"""
Παραδοτέο 1Α
Αυτόματη ανακατασκευή 2 προτάσεων (rule-based).

Το script:
1) Παίρνει μία πρόταση ως input
2) Κάνει βασικό καθάρισμα/κανονικοποίηση (κενά, “έξυπνα” εισαγωγικά)
3) Περνάει την πρόταση από LanguageTool (κανόνες γραμματικής/σύνταξης – rule-based)
4) Εφαρμόζει στοχευμένους κανόνες αντικατάστασης για συγκεκριμένες φράσεις
5) Επιστρέφει αποτέλεσμα

"""

import re
import language_tool_python


# Αρχικοποίηση LanguageTool για αγγλικά (rule-based grammar checking)
tool = language_tool_python.LanguageTool("en-US")


def normalize(text: str) -> str:
    # trim στην αρχή/τέλος
    text = text.strip()
    #μαζεύει πολλαπλά κενά
    text = re.sub(r"\s+", " ", text)

    return text


def apply_rules(text: str) -> str:

    #Οι κανόνες είναι λίγοι για να μην γίνει paraphrasing
    rules = [
        (r"\bThank your message\b", "Thank you for your message"),
        (r"\bcontract checking\b", "contract review"),
        (r"\bpart final yet\b", "final version of that section yet"),
        (r"\bI missed,\s*I apologize if so\b", "if I missed it, I apologize"),
    ]
    out = text
    for pattern, replacement in rules:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)

    return out


def reconstruct(sentence: str) -> str:

    # normalize input
    text = normalize(sentence)

    # grammar/syntax correction
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)

    corrected = apply_rules(corrected)

    return normalize(corrected)


if __name__ == "__main__":
    s1 = "Thank your message to show our words to the doctor, as his next contract checking, to all of us."
    s2 = "Because I didn’t see that part final yet, or maybe I missed, I apologize if so."

    print(reconstruct(s1))
    print(reconstruct(s2))
