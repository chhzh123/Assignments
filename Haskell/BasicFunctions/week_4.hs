module HW where

mygcd :: Integer -> Integer -> Integer
mygcd 1 1 = 1
mygcd a b
    | a > b  = gcd b (a `mod` b)
    | a == b = a
    | a < b  = gcd b a

fac :: Integer -> Integer
fac 0 = 1
fac n = fac (n-1) * n

sumFacs :: Integer -> Integer
sumFacs 0 = fac 0
sumFacs n = fac n + sumFacs (n-1)

sumFun :: (Integer -> Integer) -> Integer -> Integer
sumFun f 0 = f 0
sumFun f n = f n + sumFun f (n-1)

maxFun :: (Integer -> Integer) -> Integer -> Integer
maxFun f 0 = f 0
maxFun f n = max (f n) (maxFun f (n-1))

fib :: Integer -> Integer
fib 0 = 0
fib 1 = 1
fib n = fib (n-1) + fib (n-2)

sqrt2 :: Float -> Integer -> Float
sqrt2 x0 0 = x0
sqrt2 x0 n = ((sqrt2 x0 (n-1)) + 2 / (sqrt2 x0 (n-1))) / 2

roots :: (RealFloat f) => (f, f, f) -> (f,f)
roots (a, b, c) = (((-1)*b - sqrt(b*b - 4*a*c))/(2*a), ((-1)*b + sqrt(b*b - 4*a*c))/(2*a))