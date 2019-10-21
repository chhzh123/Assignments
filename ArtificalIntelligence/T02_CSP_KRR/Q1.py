word_list = ["add", "ado", "age", "ago", "aid",
            "ail", "aim", "air", "and", "any",
            "ape", "apt", "arc", "are", "ark",
            "arm", "art", "ash", "ask", "auk",
            "awe", "awl", "aye", "bad", "bag",
            "ban", "bat", "bee", "boa", "ear",
            "eel", "eft", "far", "fat", "fit",
            "lee", "oaf", "rat", "tar", "tie"]

s1 = set([w[0] for w in word_list])
s2 = set([w[1] for w in word_list])
s3 = set([w[2] for w in word_list])
print(sorted(list(s1)))
print(sorted(list(s2)))
print(sorted(list(s3)))
print(*sorted(list(s1.intersection(s1))),sep=",")
print(*sorted(list(s1.intersection(s2))),sep=",")
print(*sorted(list(s1.intersection(s3))),sep=",")
print(*sorted(list(s2.intersection(s1))),sep=",")
print(*sorted(list(s2.intersection(s2))),sep=",")
print(*sorted(list(s2.intersection(s3))),sep=",")
print(*sorted(list(s3.intersection(s1))),sep=",")
print(*sorted(list(s3.intersection(s2))),sep=",")
print(*sorted(list(s3.intersection(s3))),sep=",")