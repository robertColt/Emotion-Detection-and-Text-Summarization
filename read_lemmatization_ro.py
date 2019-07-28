import pickle


lemmas = {}
with open("data/lemmatization-ro.txt", "rt", encoding="UTF-8") as inputfile:
    count = 0
    for line in inputfile:
        parts = line.strip().split("\t")
        lemmas[parts[1]] = parts[0]


print(len(lemmas))

with open("data/lemmas_ro.dict", "wb") as outfile:
    pickle.dump(lemmas, outfile)
