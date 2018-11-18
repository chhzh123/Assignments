-- 陈鸿峥 17341015, chenhzh37@mail2.sysu.edu.cn, 数据科学计算机学院, 计算机类
-- 实验二: NewtonRaphson

module Newton_Raphson where
import Test.QuickCheck

squareroot2 :: Float -> Integer -> Float
squareroot2 x0 0 = x0
squareroot2 x0 n = ((squareroot2 x0 (n-1)) + 2 / (squareroot2 x0 (n-1))) / 2

squareroot :: Float -> Float -> Integer -> Float -- r>0
squareroot r x0 0 = x0
squareroot r x0 n = ((squareroot r x0 (n-1)) + r / (squareroot r x0 (n-1))) / 2

sqrtSeq :: Float -> Float -> [Float]
sqrtSeq r x0 = [squareroot r x0 n | n <- [0,1..]]
-- sqrtSeq r x0 = x0 : sqrtSeq r ((x0 + r / x0) / 2)

squareroot' :: Float -> Float -> Float -> Float
squareroot' r x0 epsilon = head [(sqrtSeq r x0 !! n) | n <- [0,1..], abs((sqrtSeq r x0 !! n) - (sqrtSeq r x0 !! (n+1))) < epsilon]

-- -- test
-- prop_sqrt2 :: Float -> Property
-- prop_sqrt2 x = x > 0 ==> squareroot 2 x 15 == squareroot2 x 15

-- prop_sqrt :: Float -> Property
-- prop_sqrt x = x > 0 ==> abs(squareroot' x 1 0.0000001 - sqrt x) < 0.000001