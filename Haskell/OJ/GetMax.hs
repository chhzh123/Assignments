module GetMax where
mymax :: [Int] -> Int
mymax lst@(x:xs)
		| length lst == 1 = head lst
		| length lst == 2 = max (head lst) (last lst)
		| otherwise = max x (mymax xs)