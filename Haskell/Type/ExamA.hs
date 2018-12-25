module ExamA where
import Data.List

type QuestionNumber = Int  -- 问题编号：1,2,3,4
type Answer = String  -- 对问题的回答
type Reply = [(QuestionNumber, Answer)]  -- 一份答卷包含对问题1-4的答案，也可能对某个问题没有回答，但对于每个问题只有一个回答。
type ReplyCnt = (Int, Answer)

-- number_of_replies someReplies1 1 = 2
number_of_replies ::[Reply] -> QuestionNumber -> Int
number_of_replies lst x = sum [sum [if (fst oneReply == x) then 1 else 0 | oneReply <- reply] | reply <- lst]

answers :: [Reply]-> QuestionNumber -> [Answer]
answers lst x = [snd ele | ele <- concat [filter (\oneReply -> fst oneReply == x) reply | reply <- lst]]

number_of_answer :: [Reply] -> QuestionNumber -> Answer -> Int
number_of_answer lst x ans = length $ filter (\a -> a == ans) (answers lst x)

count_answers :: [Answer] -> [(Int, Answer)]
count_answers lst = [(length ele, ele !! 0)::(Int, Answer) | ele <- group $ sort lst]

compareAns :: ReplyCnt -> ReplyCnt -> Ordering
compareAns p1 p2 = if (fst p1 > fst p2) then LT
                    else if (fst p1 < fst p2) then GT
                        else if (snd p1 < snd p2) then LT
                            else if (snd p1 > snd p2) then GT
                                else EQ

summary :: [Reply] -> [(QuestionNumber, [(Int, Answer)])]
summary lst = [(x, sortBy compareAns (count_answers $ answers lst x)) :: (QuestionNumber, [(Int, Answer)]) | x <- [1..4]]

extract :: FilePath -> IO [Reply]
extract f = do
    ls <- readFile f
    return (getReplies . lines $ ls)
 
getReplies :: [String] -> [Reply]
getReplies [] = []
getReplies xss = f xs : getReplies ys
    where
    xs = takeWhile notSep xss
    ys = tail (dropWhile notSep xss)
    notSep = \x -> x /="----"
    f :: [String] -> [(Int,String)]
    f [] = []
    f (a:b:zs) = (read a :: Int, b) : f zs

someReplies1 :: [Reply]
someReplies1 = [[(1,"very good"), (3,"bad"), (4,"ok")],
                [(1, "good"), (2,"good"), (3,"bad"),(4,"difficult")],
                [(3,"ok"),(4,"very difficult")]]

someReplies2 :: [Reply]
someReplies2 = [[(2,"ok"),(3,"ok"),(4,"ok")],
                [(1,"very bad"),(2,"ok"),(3,"very good"),(4,"very easy")],
                [(1,"very bad"),(3,"ok"),(4,"difficult")],
                [(1,"very bad"),(2,"very bad"),(3,"good"),(4,"ok")],
                [(1,"very bad"),(2,"ok"),(3,"good"),(4,"easy")],
                [(1,"good"),(2,"ok"),(3,"very bad"),(4,"very difficult")],
                [(1,"very good"),(2,"very bad"),(4,"difficult")],
                [(1,"ok"),(2,"ok"),(3,"good"),(4,"difficult")],
                [(1,"ok"),(3,"ok")],
                [(1,"good"),(2,"very bad"),(3,"very good"),(4,"easy")]]

getAns :: IO ()
getAns = do
    replies <- extract "R500.txt"
    putStrLn $ show $ number_of_replies replies 1
    putStrLn $ show $ number_of_answer replies 3 "ok"
    putStrLn $ show $ number_of_answer replies 3 "good"
    putStrLn $ show $ number_of_answer replies 3 "very good"
    putStrLn $ show $ number_of_answer replies 3 "bad"

getLastAns :: IO [(QuestionNumber, [(Int, Answer)])]
getLastAns = do
    replies <- extract "R500.txt"
    return $ summary replies