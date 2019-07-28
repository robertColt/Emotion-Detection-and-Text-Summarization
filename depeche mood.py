import pandas as pd
from translate import Translator


depeche_mood = pd.read_csv("data/DepecheMood.txt", sep="\t")

lemma_pos = depeche_mood["Lemma#PoS"].str.split("#", n=1, expand=True)

depeche_mood["lemma"] = lemma_pos[0]
depeche_mood["pos"] = lemma_pos[1]

cols = depeche_mood.columns.tolist()
cols = cols[-2:] + cols[0:-2]
depeche_mood = depeche_mood[cols]

print("Starting translating...")

counter = 0
values = len(depeche_mood.values)
translator = Translator(to_lang="Romanian")

def translate(word):
    global counter, values, translator
    try:
        counter += 1
        if counter % 50 == 0:
            print(f"{counter}/{values}")
        return translator.translate(word).lower()
    except Exception as e:
        counter -= 1
        print("err", e)
        return "err_" + word


translated = depeche_mood.filter(["lemma"], axis=1)
translated["lemma"] = translated["lemma"].apply(lambda row: translate(row))

translated.to_csv("translated.csv")