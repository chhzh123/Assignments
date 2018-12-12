-- foldr ::(a->b->b) -> b ->[a] -> b
unlines' :: [String] -> String
unlines' lst = foldr (\x xs -> x ++ "\n" ++ xs) "" lst

and' :: [Bool] -> Bool
and' lst = foldr (\x xs -> x && xs) True lst

or' :: [Bool] -> Bool
or' lst = foldr (\x xs -> x || xs) False lst