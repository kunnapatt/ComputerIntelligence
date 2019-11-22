f = open("cross.pat", "r")
cou = 0
test = []
for row in f:
    if ( cou % 3 != 0 ):
        test.append(row)
    # print(row)
    cou += 1

print(test)