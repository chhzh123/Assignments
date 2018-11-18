-- 陈鸿峥 17341015, chenhzh37@mail2.sysu.edu.cn, 数据科学计算机学院, 计算机类
-- 实验四: sayit

module MyPicture where
import Data.Char
import Data.List

bigTable :: [[String]]
bigTable = [["     ","     ","     ","     ","     "],
            ["  A  "," A A ","A   A","A A A","A   A"],
            ["BBBB ","B   B","BBBB ","B   B","BBBB "],
            [" CCCC","C    ","C    ","C    "," CCCC"],
            ["DDDD ","D   D","D   D","D   D","DDDD "],
            ["EEEEE","E    ","EEEEE","E    ","EEEEE"],
            ["FFFFF","F    ","FFFFF","F    ","F    "],
            [" GGGG","G    ","GGGGG","G   G"," GGGG"],
            ["H   H","H   H","HHHHH","H   H","H   H"],
            ["IIIII","  I  ","  I  ","  I  ","IIIII"],
            ["JJJJJ","  J  ","  J  ","J J  ","  J  "],
            ["K   K","K  K ","KKK  ","K  K ","K   K"],
            ["L    ","L    ","L    ","L    ","LLLLL"],
            ["M   M","MM MM","M M M","M   M","M   M"],
            ["N   N","NN  N","N N N","N  NN","N   N"],
            [" OOO ","O   O","O   O","O   O"," OOO "],
            ["PPPP ","P   P","PPPP ","P    ","P    "],
            [" QQQ ","Q   Q","Q   Q"," QQQ ","    Q"],
            ["RRRR ","R   R","RRRR ","R  R ","R   R"],
            [" SSSS","S    "," SSS ","    S","SSSS "],
            ["TTTTT","  T  ","  T  ","  T  ","  T  "],
            ["U   U","U   U","U   U","U   U"," UUU "],
            ["V   V"," V V "," V V ","  V  ","  V  "],
            ["W   W","W   W","W W W","WW WW","W   W"],
            ["X   X"," X X ","  X  "," X X ","X   X"],
            ["Y   Y"," Y Y ","  Y  ","  Y  ","  Y  "],
            ["ZZZZZ","   Z ","  Z  "," Z   ","ZZZZZ"],
            ["00000","0   0","0   0","0   0","00000"],
            ["  1  "," 11  ","  1  ","  1  ","11111"],
            [" 222 ","2   2","   2 ","  2  ","22222"],
            ["3333 ","    3"," 333 ","    3","3333 "],
            ["  44 "," 4 4 ","4  4 ","44444","   4 "],
            ["55555","5    ","55555","    5","55555"],
            ["66666","6    ","66666","6   6","66666"],
            ["77777","    7","    7","    7","    7"],
            ["88888","8   8","88888","8   8","88888"],
            ["99999","9   9","99999","    9","99999"]]

makeBig :: Char -> [String]
makeBig x
      | ord x == 32 = bigTable !! 0
      | ord x >= 48 && ord x <= 57 = bigTable !! (ord x - 48 + 27)
      | ord x >= 65 && ord x <= 90 = bigTable !! (ord x - 65 + 1)
      | ord x >= 97 && ord x <= 122 = bigTable !! (ord x - 97 + 1)

makeString :: String -> [[String]]
makeString "" = []
makeString (x:xs) = makeBig x : makeString xs

say :: String -> String
say str = unlines [(intercalate " " [x !! i | x <- makeString str]) | i <- [0..4]] -- take every row and combine

sayit :: String -> IO()
sayit = putStr . say

-- test
-- sayit "Hello"
-- sayit "Hi 123"
-- sayit "tql hyj orz"