rotate_right = {"f":"r",
                "b":"l",
                "l":"f",
                "r":"b",
                "u":"u",
                "d":"d",
                "5":"6",
                "6":"8",
                "7":"5",
                "8":"7"}

rotate_left =  {"r":"f",
                "l":"b",
                "f":"l",
                "b":"r",
                "u":"u",
                "d":"d",
                "6":"5",
                "8":"6",
                "5":"7",
                "7":"8"}

rotate_up = {"f":"u",
             "b":"d",
             "l":"l",
             "r":"r",
             "u":"b",
             "d":"f",
             "2":"6",
             "6":"8",
             "8":"4",
             "4":"2"}

rotate_down = {"u":"f",
               "d":"b",
               "l":"l",
               "r":"r",
               "b":"u",
               "f":"d",
               "6":"2",
               "8":"6",
               "4":"8",
               "2":"4"}

rotate_front = {"f":"f", # front right
                "b":"b",
                "u":"r",
                "d":"l",
                "r":"d",
                "l":"u",
                "2":"1",
                "6":"2",
                "5":"6",
                "1":"5"}

rotate_back = {"f":"f", # front left
               "b":"b",
               "l":"d",
               "r":"u",
               "u":"l",
               "d":"r",
               "1":"2",
               "2":"6",
               "6":"5",
               "5":"1"}

blocks = {"U":[5,6,7,8],
          "U_p":[5,6,7,8],
          "R":[2,4,6,8],
          "R_p":[2,4,6,8],
          "F":[1,2,5,6],
          "F_p":[1,2,5,6]
         }

color_index = [["1f","1l","1d"],
               ["2f","2r","2d"],
               ["3b","3l","3d"],
               ["4b","4r","4d"],
               ["5f","5l","5u"],
               ["6f","6r","6u"],
               ["7b","7l","7u"],
               ["8b","8r","8u"]]

for i in range(8):
    for j in range(3):
        color_index[i][j] = color_index[i][j][1] + color_index[i][j][0]

def get_new_color(origin,rotate):
    return rotate[origin[0]] + rotate[origin[1]]

for action,rotate in [("R",rotate_up),("R_p",rotate_down),("U",rotate_left),("U_p",rotate_right),("F",rotate_front),("F_p",rotate_back)]:
    print("(:action {}".format(action))
    block1 = blocks[action][0]
    block2 = blocks[action][1]
    block3 = blocks[action][2]
    block4 = blocks[action][3]
    new_index = {}
    for block in [block1,block2,block3,block4]:
        for i in range(3):
            new_index[get_new_color(color_index[block-1][i],rotate)] = color_index[block-1][i]
    # print("    :parameters (?{} ?{} ?{}".format(color_index[block1-1][0],color_index[block1-1][1],color_index[block1-1][2]))
    # print("                 ?{} ?{} ?{}".format(color_index[block2-1][0],color_index[block2-1][1],color_index[block2-1][2]))
    # print("                 ?{} ?{} ?{}".format(color_index[block3-1][0],color_index[block3-1][1],color_index[block3-1][2]))
    # print("                 ?{} ?{} ?{}".format(color_index[block4-1][0],color_index[block4-1][1],color_index[block4-1][2]))
    # print("    )")
    # print("    :precondition (and")
    # print("        (color{} ?{} ?{} ?{})".format(str(block1),color_index[block1-1][0],color_index[block1-1][1],color_index[block1-1][2]))
    # print("        (color{} ?{} ?{} ?{})".format(str(block2),color_index[block2-1][0],color_index[block2-1][1],color_index[block2-1][2]))
    # print("        (color{} ?{} ?{} ?{})".format(str(block3),color_index[block3-1][0],color_index[block3-1][1],color_index[block3-1][2]))
    # print("        (color{} ?{} ?{} ?{})".format(str(block4),color_index[block4-1][0],color_index[block4-1][1],color_index[block4-1][2]))
    # print("    )")
    # print("    :effect (and")
    # print("        (not (color{} ?{} ?{} ?{}))".format(str(block1),color_index[block1-1][0],color_index[block1-1][1],color_index[block1-1][2]))
    # print("        (not (color{} ?{} ?{} ?{}))".format(str(block2),color_index[block2-1][0],color_index[block2-1][1],color_index[block2-1][2]))
    # print("        (not (color{} ?{} ?{} ?{}))".format(str(block3),color_index[block3-1][0],color_index[block3-1][1],color_index[block3-1][2]))
    # print("        (not (color{} ?{} ?{} ?{}))".format(str(block4),color_index[block4-1][0],color_index[block4-1][1],color_index[block4-1][2]))
    # print("        (color{} ?{} ?{} ?{})".format(str(block1),new_index[color_index[block1-1][0]],new_index[color_index[block1-1][1]],new_index[color_index[block1-1][2]]))
    # print("        (color{} ?{} ?{} ?{})".format(str(block2),new_index[color_index[block2-1][0]],new_index[color_index[block2-1][1]],new_index[color_index[block2-1][2]]))
    # print("        (color{} ?{} ?{} ?{})".format(str(block3),new_index[color_index[block3-1][0]],new_index[color_index[block3-1][1]],new_index[color_index[block3-1][2]]))
    # print("        (color{} ?{} ?{} ?{})".format(str(block4),new_index[color_index[block4-1][0]],new_index[color_index[block4-1][1]],new_index[color_index[block4-1][2]]))
    # print("    )")
    # print(")")
    # print()
    print("    :effect (and")
    for block in [block1,block2,block3,block4]:
        print("        (forall (?{1} ?{2} ?{3} - color) (when (color{0} ?{1} ?{2} ?{3})".format(str(block),color_index[block-1][0],color_index[block-1][1],color_index[block-1][2]))
        print("            (and")
        print("                (not (color{0} ?{1} ?{2} ?{3}))".format(str(block),color_index[block-1][0],color_index[block-1][1],color_index[block-1][2]))
        new_block = int(rotate[str(block)])
        print("                (color{0} ?{1} ?{2} ?{3})".format(str(new_block),new_index[color_index[new_block-1][0]],new_index[color_index[new_block-1][1]],new_index[color_index[new_block-1][2]]))
        print("            ))")
        print("        )")
    print("    )")
    print(")")
    print()