for a in [1,0]:
    for b in [1,0]:
        for c in [1,0]:
            for d in [1,0]:
                e = (a ^ b) & (c ^ d)
                print(a,b,c,d,e,sep=" & ",end="\\\\\\hline\n")