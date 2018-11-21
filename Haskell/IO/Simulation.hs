module Simulation where
import System.Random
import Control.Monad
import Text.Printf

showPre :: Float -> String
showPre x = printf "%.4f" x

-- throw n dices
oneTest :: Int -> IO [Int]
oneTest n = do
	dices <- forM [1..n] (\a -> do
		dice <- randomRIO(1,6)
		return dice)
	return dices

-- count how many 6s in these dices
count :: [Int] -> Int
count dices = length [x | x <- dices, x == 6]

-- Monte Carlo Simulation
-- t: # of dices
-- n: # of 6s in one test
-- num: # of 6s in all tests
-- sim: simulation time
test :: Int -> Int -> Int -> Int -> IO Int
test t n num sim = do
	if sim == 0
		then return num
	else do
		dices <- (oneTest t)
		if (count dices) >= n
			then test t n (num+1) (sim-1)
			else test t n num (sim-1)

-- encapsulation
run :: IO ()
run = do
	putStrLn "Similation time: 1000"
	a <- (test 6 1 0 1000)
	b <- (test 12 2 0 1000)
	c <- (test 18 3 0 1000)
	putStrLn ("The probability of A: " ++ show (fromIntegral a / 1000::Float))
	putStrLn ("The probability of B: " ++ show (fromIntegral b / 1000::Float))
	putStrLn ("The probability of C: " ++ show (fromIntegral c / 1000::Float))
	let maximum = max a (max b c)
	if maximum == a
		then putStrLn "A has the highest probability!"
		else if maximum == b
			then putStrLn "B has the highest probability!"
			else putStrLn "C has the highest probability!"

-- simulation results are attached in the webpage