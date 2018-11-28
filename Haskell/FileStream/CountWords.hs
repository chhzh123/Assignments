module CountWords where
import Data.Char
import Control.Monad
import Data.List

type Freq = (String, Int)

filterLst :: [String] -> [String]
filterLst strLst = [filter (\x -> (isAlpha x || x == '-')) str | str <- strLst]

-- 注意文本文件一定要保存为UTF-8格式，否则's会检索不到
ruleS :: [String] -> [String]
ruleS strLst = [if ((take 2 (reverse str) == "s\'") || (take 2 (reverse str) == "s\8217")) then (init $ init $ str) else str | str <- strLst]

allLower :: [String] -> [String]
allLower strLst = [map toLower str | str <- strLst]

countWords :: [String] -> [Freq]
countWords strLst = [(head x, length x)::Freq | x <- group $ sort strLst]

compareFreq :: Freq -> Freq -> Ordering
compareFreq x y
			| (snd x > snd y || (snd x == snd y && fst x < fst y)) = LT
			| (fst x == fst y && snd x == snd y )= EQ
			| otherwise = GT

getInt :: IO Int
getInt = do
	x <- getLine
	return (read x :: Int)

showFreq :: Freq -> IO ()
showFreq freq = do
	putStrLn ((fst freq) ++ ":" ++ (show $ snd freq))

count :: IO ()
count = do
	n <- getInt
	text <- readFile ("./text" ++ show n ++ ".txt")
	let result = allLower $ filterLst $ ruleS $ words text
	let sorted_result = sortBy compareFreq (countWords result)
	mapM_ showFreq sorted_result
	writeFile ("./freq" ++ show n ++ ".txt") (unlines [(fst freq) ++ ":" ++ (show $ snd freq) | freq <- sorted_result])

{-test
Mind-controlled Mouse
The three study participants are part of a clinical trial to test a brain-computer interface (BCI) called BrainGate. BrainGate translates participant’s brain activity into commands that a computer can understand. In the new study, researchers first implanted microelectrode arrays into the area of the brain that governs hand movement. The participants trained the system by thinking about moving their hands, something the BCI learned to translate into actions on the screen.

Result:
the:8
a:3
into:3
bci:2
brain:2
braingate:2
of:2
participants:2
study:2
that:2
to:2
about:1
actions:1
activity:1
are:1
area:1
arrays:1
brain-computer:1
...
-}