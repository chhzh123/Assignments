-- 陈鸿峥 17341015, chenhzh37@mail2.sysu.edu.cn, 数据科学计算机学院, 计算机类
-- 实验三: HaskellStore

module HaskellStore where
import Text.Printf

type Items = [Item]
type Item = (Name, Amount, Price)
-- name, amount and price per unit of the item
type Name = String -- name of the item
type Amount = Float -- amount, like kg or number
type Price = Float  -- price per unit

first :: Item -> Name
first (x,_,_) = x

second :: Item -> Amount
second (_,y,_) = y

third :: Item -> Price
third (_,_,z) = z

showPre :: Float -> String
showPre x = printf "%.2f" x

calSum :: Item -> Float
calSum item = (second item) * (third item)

calTotSum :: Items -> Float
calTotSum [] = 0
calTotSum (x:xs) = calSum x + calTotSum xs

generateItemStr :: Item -> String
generateItemStr item = name ++ space ++ " " ++ amount ++ space ++ price ++ space ++ sumup ++ "\n"
    where name = first item
          space = "  "
          amount = showPre (second item)
          price = showPre (third item)
          sumup = showPre (calSum item)

generateMiddlePart :: Items -> String
generateMiddlePart [] = ""
generateMiddlePart (x:xs) = generateItemStr x ++ generateMiddlePart xs

generateItemsStr :: Items -> String
generateItemsStr items = firstPart ++ generateMiddlePart items ++ lastPart
    where firstPart = "Name   Amount  Price  Sum\n"
          lastPart = "Total  .............. " ++ showPre (calTotSum items) ++ "\n"

printItems :: Items -> IO()
printItems x = putStr (generateItemsStr x)

-- test
customer1 :: Items
customer1 = [("Apple", 2.5, 5.99), ("Bread", 2, 3.5)]

-- customer2 :: Items
-- customer2 = [("Apple-pen", 10, 100), ("Pen-Apple", 3.4, 0.5), ("Apple", 1, 1.0), ("Pen", 0.5, 3)]

-- customer3 :: Items
-- customer3 = [("Alice", 3 ,3)]