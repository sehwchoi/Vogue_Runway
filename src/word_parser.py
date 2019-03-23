from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import re

with open("./data/list_attr_cloth_1.txt", "r") as f:
    lex_1 = [w.strip() for w in f.readlines()]

print(lex_1[:5])

with open("./data/list_attr_cloth_2.txt", "r") as f:
    lex_2 = [w.strip().split()[0] for w in f.readlines()]

print(lex_2[:5])

with open("./data/list_category_cloth.txt", "r") as f:
    lex_3 = [w.strip().split()[0] for w in f.readlines()]

print(lex_3[:5])

lex = []
lex.extend(lex_1)
lex.extend(lex_2)
lex.extend(lex_3)

print(len(lex), lex[:5])

desc_files = ["./data/test_1.txt", "./data/test_2.txt"]
for file in desc_files:
    print("Processing file ", file)
    with open(file, "r") as f:
        desc = []
        for line in f:
            for word in re.findall(r'\w+', line):
                desc.append(word)

            desc = [word for word in desc if word not in stopwords.words('english')]

    print(desc[:5])

    fashion_desc = []
    for l in lex:
        for w in desc:
            if l in w:
                fashion_desc.append(w)

    print(set(fashion_desc))
