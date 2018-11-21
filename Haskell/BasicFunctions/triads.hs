-- module traids where

traids :: Int -> [(Int, Int, Int)]
traids n = [(x,y,z) | x <- [1..n], y <- [1..n], z <- [1..n], x^2+y^2==z^2]

traids2 :: Int -> [(Int, Int, Int)]
traids2 n = [(x,y,z) | x <- [1..n], y <- [x..n], z <- [y..n], x^2+y^2==z^2]