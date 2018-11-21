module ValidQ where
import Data.Char

sumAll :: [Int] -> Int
sumAll [] = 0
sumAll (x:xs) = sum [(ord digit - ord '0') | digit <- (show x)] + sumAll xs

test :: String -> Int
test str = (sumAll [(ord (rstr !! i) - ord '0') * ((i `mod` 2)+1) | i <- [0..(length rstr-1)]]) `mod` 10
	where rstr = reverse (show (read str::Int)) -- avoid "\r"

count :: IO ()
count = do
	cards <- readFile "./cards200.txt"
	let results = map (test) (lines cards) -- lines cannot eliminate "\r\n"
	putStrLn ("Valid: " ++ show (length [x | x <- results, x == 0]))
	putStrLn ("Invalid: " ++ show (length [x | x <- results, x /= 0]))