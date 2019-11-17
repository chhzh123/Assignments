label = [["7u","8u"],
["5u","6u"],
["7l","5l","5f","6f","6r","8r","8b","7b"],
["3l","1l","1f","2f","2r","4r","4b","3b"],
["1d","2d"],
["3d","4d"]]

case_test_f = [["y","y"],
["b","b"],
["g","y","o","o","w","b","r","r"],
["g","y","o","o","w","b","r","r"],
["g","g"],
["w","w"]]

case_test_fp = [["y","y"],
["g","g"],
["g","w","o","o","y","b","r","r"],
["g","w","o","o","y","b","r","r"],
["b","b"],
["w","w"]]

case_test_r = [["y","r"],
["y","r"],
["g","g","o","y","b","b","w","r"],
["g","g","o","y","b","b","w","r"],
["w","o"],
["w","o"]]

case_test_rp = [["y","o"],
["y","o"],
["g","g","o","w","b","b","y","r"],
["g","g","o","w","b","b","y","r"],
["w","r"],
["w","r"]]

case_test_u = [["y","y"],
["y","y"],
["r","r","g","g","o","o","b","b"],
["g","g","o","o","b","b","r","r"],
["w","w"],
["w","w"]]

case_test_up = [["y","y"],
["y","y"],
["o","o","b","b","r","r","g","g"],
["g","g","o","o","b","b","r","r"],
["w","w"],
["w","w"]]

case_test = [["y","y"],
["b","r"],
["w","o","w","y","b","y","g","g"],
["b","g","r","w","b","w","r","y"],
["y","r"],
["y","g"]]

case1 = [["g","g"],
["g","w"],
["o","o","y","r","b","y","r","w"],
["r","b","o","r","g","o","b","y"],
["y","w"],
["b","w"]]

case2 = [["y","w"],
["y","o"],
["r","g","o","w","b","o","g","b"],
["w","b","r","b","o","r","y","g"],
["w","y"],
["r","g"]]

case3 = [["g","b"],
["b","g"],
["w","o","w","o","y","r","y","r"],
["y","r","y","g","o","w","b","o"],
["g","w"],
["b","r"]]

case4 = [["r","w"],
["r","g"],
["b","g","y","o","y","r","b","y"],
["b","b","o","o","w","r","g","o"],
["y","g"],
["w","w"]]

goal = [["y","y"],
["y","y"],
["g","g","o","o","b","b","r","r"],
["g","g","o","o","b","b","r","r"],
["w","w"],
["w","w"]]

goal1 = [["g","g"],
["g","g"],
["r","r","w","w","o","o","y","y"],
["r","r","w","w","o","o","y","y"],
["b","b"],
["b","b"]]

goal2 = [["o","o"],
["o","o"],
["w","w","b","b","y","y","g","g"],
["w","w","b","b","y","y","g","g"],
["r","r"],
["r","r"]]

goal3 = [["g","g"],
["g","g"],
["y","y","r","r","w","w","o","o"],
["y","y","r","r","w","w","o","o"],
["b","b"],
["b","b"]]

goal4 = [["y","y"],
["y","y"],
["b","b","r","r","g","g","o","o"],
["b","b","r","r","g","g","o","o"],
["w","w"],
["w","w"]]

dict_map = {}
# for case in [case1,case2,case3,case4,goal]:
for case in [goal3,goal4]:
    for i,row in enumerate(label):
        for j,col in enumerate(row):
            dict_map[col] = case[i][j]
    print("        (color1 {} {} {})".format(dict_map["1f"],dict_map["1l"],dict_map["1d"]))
    print("        (color2 {} {} {})".format(dict_map["2f"],dict_map["2r"],dict_map["2d"]))
    print("        (color3 {} {} {})".format(dict_map["3b"],dict_map["3l"],dict_map["3d"]))
    print("        (color4 {} {} {})".format(dict_map["4b"],dict_map["4r"],dict_map["4d"]))
    print("        (color5 {} {} {})".format(dict_map["5f"],dict_map["5l"],dict_map["5u"]))
    print("        (color6 {} {} {})".format(dict_map["6f"],dict_map["6r"],dict_map["6u"]))
    print("        (color7 {} {} {})".format(dict_map["7b"],dict_map["7l"],dict_map["7u"]))
    print("        (color8 {} {} {})".format(dict_map["8b"],dict_map["8r"],dict_map["8u"]))
    print()