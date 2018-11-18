module Game where
import System.Random

data Hand = Rock | Scissor | Paper deriving (Enum, Eq)

instance Show Hand where
	show Rock = "石头"
	show Scissor = "剪刀"
	show Paper = "布"

instance Random Hand where
	random g = case randomR(0,2) g of
		(r,g') -> (toEnum r,g')
	randomR (a,b) g = case randomR (fromEnum a, fromEnum b) g of
		(r,g') -> (toEnum r,g')

-- 猜拳规则，返回1前者赢，-1后者赢，0平手
rule :: Hand -> Hand -> Int
rule x y
       | (x == Paper && y == Paper) || (x == Scissor && y == Scissor) || (x == Rock && y == Rock) = 0
       | (x == Paper && y == Rock) || (x == Scissor && y == Paper) || (x == Rock && y == Scissor) = 1
       | otherwise = -1

-- 一次猜拳，并计数
playOne :: Int -> Int -> IO ()
playOne com_win user_win = do
	if com_win == 3
		then
			putStrLn "哈哈，我赢了！"
	else if user_win == 3
		then
			putStrLn "算您赢了这轮。"
		else do
			n <- getStdRandom (randomR (Paper::Hand, Rock::Hand))
			putStr "请您出手 (R)石头, (S)剪刀, (P)布:"
			str <- getChar
			let res
			      | str == 'p' || str == 'P' = Paper :: Hand
			      | str == 's' || str == 'S' = Scissor :: Hand
			      | str == 'r' || str == 'R' = Rock :: Hand
			putStrLn (", 您出了" ++ show res ++ ", 我出了" ++ show n)
			if rule res n == 0
				then do
					putStrLn "这一手平手"
					playOne com_win user_win
				else if rule res n == 1
					then do
						putStrLn "您赢了这手"
						putStrLn ("我的得分: " ++ show com_win)
						putStrLn ("您的得分: " ++ show (user_win+1))
						playOne com_win (user_win+1)
					else do
						putStrLn ("我的得分: " ++ show (com_win+1))
						putStrLn ("您的得分: " ++ show user_win)
						playOne (com_win+1) user_win

play :: IO ()
play = playOne 0 0

-- 测试结果请见截图文件