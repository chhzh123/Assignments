module Main where
import Data.Char

getInt :: IO Int
getInt = do
	x <- getLine
	return (read x :: Int)

sumAll :: [Int] -> Int
sumAll [] = 0
sumAll (x:xs) = sum [(ord digit - ord '0') | digit <- (show x)] + sumAll xs

test :: String -> Int
test str = (sumAll [(ord (rstr !! i) - ord '0') * ((i `mod` 2)+1) | i <- [0..(length rstr-1)]]) `mod` 10
	where rstr = reverse (show (read str::Int)) -- avoid "\r"

severalTest :: Int -> IO ()
severalTest 0 = do
	return ()
severalTest n = do
	str <- getLine
	if test str == 0
		then do
			putStrLn str
			severalTest (n-1)
		else
			severalTest (n-1)

main :: IO ()
main = do
	n <- getInt
	severalTest n