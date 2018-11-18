import Control.Monad
import System.Random

getInt :: IO Int
getInt = do
	x <- getLine
	return (read x :: Int)

getInts :: Integer -> IO [Int]
getInts n = do
	ints <- forM [1..n] (\a -> do
		num <- getInt
		return num)
	return ints

getListofInts :: IO [Int]
getListofInts = do
	x <- getInt
	if x < 0
		then return []
		else do
			moreInts <- getListofInts
			return (x : moreInts)

getSum :: IO ()
getSum = do
	res <- getListofInts
	putStrLn $ "Sum:" ++ show (sum res)

loopNum :: Int -> IO ()
loopNum num = do
	x <- getInt
	if x < num
		then do
			putStrLn "Too small!"
			loopNum num
		else if x > num
			then do
				putStrLn "Too large!"
				loopNum num
			else
				putStrLn "You are right!"

loopCount :: Int -> Int -> IO Int
loopCount num cnt = do
	x <- getInt
	if x < num
		then do
			putStrLn "Too small!"
			loopCount num (cnt+1)
		else if x > num
			then do
				putStrLn "Too large!"
				loopCount num (cnt+1)
			else do
				putStrLn "You are right!"
				return cnt

guess :: IO ()
guess = do
	putStrLn "Please enter the number to be guess:"
	n <- getInt
	if n < 1 || n > 100
		then do
			putStrLn "Error: Number should be in 1 to 100!"
			guess
		else do
			putStrLn "You can guess now..."
			loopNum n

randomGuess :: IO ()
randomGuess = do
	n <- randomRIO (1,100)
	putStrLn "You can guess now..."
	cnt <- loopCount n 0
	putStrLn $ "You use " ++ show cnt ++ " times."

rule :: Int -> Int -> Int
rule x y
       | (x == 1 && y == 1) || (x == 2 && y == 2) || (x == 3 && y == 3) = 0
       | (x == 1 && y == 3) || (x == 2 && y == 1) || (x == 3 && y == 2) = 1
       | otherwise = -1

playOne :: Int -> Int -> IO ()
playOne cnt_win cnt_lose = do
	if cnt_win == 2
		then
			putStrLn "Great! You win two games continously!"
	else if cnt_lose == 2
		then
			putStrLn "Game over! You lose two games continously!"
		else do
			n <- randomRIO (1,3)
			putStrLn "Paper, Scissors, or Rock?"
			str <- getLine
			let res
			      | str == "Paper" = 1
			      | str == "Scissors" = 2
			      | str == "Rock" = 3
			if n == 1
				then putStrLn "Paper"
				else if n == 2
					then putStrLn "Scissors"
					else putStrLn "Rock"
			if rule res n == 0
				then do
					putStrLn "Even!"
					playOne 0 0
				else if rule res n == 1
					then do
						putStrLn "Win!"
						playOne (cnt_win+1) 0
					else do
						putStrLn "Lose!"
						playOne 0 (cnt_lose+1)

play :: IO ()
play = playOne 0 0