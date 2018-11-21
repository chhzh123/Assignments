import System.Random

data Exp = Val Int | Add Exp Exp | Sub Exp Exp | Mul Exp Exp

instance Show Exp where
	-- show :: Exp -> String
	show (Val x) = show x
	show (Add (Val x) (Add (Val y) (Val z))) = show x ++ " + " ++ show y ++ " + " ++ show z
	show (Sub (Val x) (Add (Val y) (Val z))) = show x ++ " - " ++ show y ++ " + " ++ show z
	show (Mul (Val x) (Add (Val y) (Val z))) = show x ++ " * (" ++ show y ++ " + " ++ show z ++ ")"
	show (Add (Val x) (Sub (Val y) (Val z))) = show x ++ " + " ++ show y ++ " - " ++ show z
	show (Sub (Val x) (Sub (Val y) (Val z))) = show x ++ " - " ++ show y ++ " - " ++ show z
	show (Mul (Val x) (Sub (Val y) (Val z))) = show x ++ " * (" ++ show y ++ " - " ++ show z ++ ")"
	show (Add (Val x) (Mul (Val y) (Val z))) = show x ++ " + " ++ show y ++ " * " ++ show z
	show (Sub (Val x) (Mul (Val y) (Val z))) = show x ++ " - " ++ show y ++ " * " ++ show z
	show (Mul (Val x) (Mul (Val y) (Val z))) = show x ++ " * " ++ show y ++ " * " ++ show z
	show (Add e1 e2) = show e1 ++ " + " ++ show e2
	show (Sub e1 e2) = show e1 ++ " - " ++ show e2
	show (Mul e1 e2) = show e1 ++ " * " ++ show e2
	-- show (Add e1 e2) = "(" ++ show e1 ++ " + " ++ show e2 ++ ")"
	-- show (Sub e1 e2) = "(" ++ show e1 ++ " - " ++ show e2 ++ ")"
	-- show (Mul e1 e2) = "(" ++ show e1 ++ " * " ++ show e2 ++ ")"

eval :: Exp -> Int
eval (Val x) = x
eval (Add e1 e2) = eval e1 + eval e2
eval (Sub e1 e2) = eval e1 - eval e2
eval (Mul e1 e2) = eval e1 * eval e2

getInt :: IO Int
getInt = do
	x <- getLine
	return (read x :: Int)

generate :: IO Exp
generate = do
	x <- randomRIO (1,100)
	y <- randomRIO (1,100)
	op <- randomRIO (1::Int,3::Int)
	if op == 1
		then return (Add (Val x) (Val y))
		else if op == 2
			then return (Sub (Val x) (Val y))
			else return (Mul (Val x) (Val y))

generateComplex :: IO Exp
generateComplex = do
	x <- randomRIO (1,100)
	exp <- generate
	op <- randomRIO (0::Int,3::Int)
	if op == 0
		then return exp
		else if op == 1
			then do
				res <- generateComplex
				return (Add (Val x) res)
			else if op == 2
				then do
					res <- generateComplex
					return (Sub (Val x) res)
				else do
					res <- generateComplex
					return (Mul (Val x) res)

loopInput :: Int -> IO ()
loopInput n = do
	x <- getInt
	if x == n
		then putStrLn "Good!"
		else do
			putStrLn "Wrong answer!"
			loopInput n

test :: IO ()
test = do
	exp <- generateComplex
	putStrLn $ show exp
	loopInput $ eval exp