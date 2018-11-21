-- 陈鸿峥 17341015, chenhzh37@mail2.sysu.edu.cn, 数据科学计算机学院, 计算机类

module Bids where
import Text.Printf

type ID = Int
type Price = Int
type Data = (ID, Price)

q_sort :: [Data] -> [Data]
q_sort lst = case lst of
	[] -> []
	(x:xs) -> q_sort [a | a <- xs, snd a >= snd x] ++ [x] ++ q_sort [a | a <- xs, snd a < snd x]

average :: [Data] -> Float
average lst = fromIntegral (sum [snd x | x <- lst]) / fromIntegral (length lst)

transform :: [String] -> [Data]
transform lst = [(read ((words x) !! 0)::ID, read ((words x) !! 1)::Price)::Data | x <- lst]

oneData :: Data -> String
oneData x = (show $ fst x) ++ "   " ++ (show $ snd x)

showOne :: Data -> IO ()
showOne x = do
	putStrLn $ oneData x

showPre :: Float -> String
showPre x = printf "%.1f" x

display :: IO ()
display = do
	bids <- readFile "./bids_201711.txt" -- 请将本文件放在与本程序同一目录下
	let sorted_bids = q_sort $ transform $ lines bids
	putStrLn ("最高成交价：" ++ (show $ snd $ head sorted_bids))
	putStrLn ("最低成交价：" ++ (show $ snd $ last sorted_bids))
	putStrLn ("平均成交价：" ++ (showPre $ average sorted_bids))
	putStrLn ("总共有" ++ (show $ length sorted_bids) ++ "参与竞价")
	putStrLn "成交名单："
	mapM_ (showOne) (take 10 sorted_bids) -- Monad

displayToFile :: IO ()
displayToFile = do
	bids <- readFile "./bids_201711.txt" -- 请将本文件放在与本程序同一目录下
	let sorted_bids = q_sort $ transform $ lines bids
	let res = ("最高成交价：" ++ (show $ snd $ head sorted_bids) ++ "\n"
			++ "最低成交价：" ++ (show $ snd $ last sorted_bids) ++ "\n"
			++ "平均成交价：" ++ (showPre $ average sorted_bids) ++ "\n"
			++ "总共有" ++ (show $ length sorted_bids) ++ "参与竞价" ++ "\n"
			++ "成交名单：" ++ "\n"
			++ unlines (map (oneData) (take 10 sorted_bids)))
	writeFile "./bidResults.txt" res

{- 测试结果
最高成交价：29995
最低成交价：10008
平均成交价：19836.3
总共有491参与竞价
成交名单：
1234198610248103   29995
1535198004062143   29948
1334197907229015   29922
1435198110019109   29916
1535198412072052   29822
1533198811246245   29820
1633197706227256   29815
1234198205229019   29799
1533198710186071   29777
1536198405206083   29727
-}