data Shape = Circle Float | Rectangle Float Float | Triangle Float Float Float deriving (Show, Eq)

isRound :: Shape -> Bool
isRound (Circle r) = True
isRound _ = False

area :: Shape -> Float
area (Circle r) = 3.14 * r * r
area (Rectangle l w) = l * w
area (Triangle a b c) = let p = (a+b+c)/2 in sqrt (p*(p-a)*(p-b)*(p-c)) -- Helen

perimeter :: Shape -> Float
perimeter (Circle r) = 3.14 * 2 * r
perimeter (Rectangle l w) = 2 * (l + w)
perimeter (Triangle a b c) = a + b + c

isRegular :: Shape -> Bool
isRegular (Circle r) = True
isRegular (Rectangle a b)
          | a == b = True
          | otherwise = False
isRegular (Triangle a b c)
          | a == b && b == c = True
          | otherwise = False

-- data Item = (Name String, Amount Float, Price Float)