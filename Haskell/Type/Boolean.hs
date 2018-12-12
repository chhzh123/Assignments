-- 17341015 数据科学与计算机学院计算机类 陈鸿峥 chenhzh37@mail2.sysu.edu.cn
module Lab3 (
	Prop(..),
	isTaut -- :: Prop -> Bool
	) where
import Data.Set
import Data.Bits

-- Proposition
data Prop = Const Bool
		  | Var Char
		  | Not Prop
		  | And Prop Prop
		  | Or Prop Prop
		  | Imply Prop Prop
		  deriving Eq

-- Substitution
type Subst = [(Char, Bool)]

-- Exer 1
instance Show Prop where
	show (Const x)     = show x
	show (Var x)       = [x]
	show (Not p)       = "~" ++ show p
	show (And p1 p2)   = "(" ++ show p1 ++ "&&" ++ show p2 ++ ")"
	show (Or p1 p2)    = "(" ++ show p1 ++ "||" ++ show p2 ++ ")"
	show (Imply p1 p2) = "(" ++ show p1 ++ "=>" ++ show p2 ++ ")"

-- Exer 2
p1 :: Prop
p1 = And (Var 'A') (Not (Var 'A'))
p2 :: Prop
p2 = Or (Var 'A') (Not (Var 'A'))
p3 :: Prop
p3 = Imply (Var 'A') (And (Var 'A') (Var 'B'))

-- Exer 3
eval :: Subst -> Prop -> Bool
eval subst prop = case prop of
	Const x     -> x
	Var c       -> snd (head [item | item <- subst, fst item == c])
	Not p       -> not (eval subst p)
	And p1 p2   -> (eval subst p1) && (eval subst p2)
	Or p1 p2    -> (eval subst p1) || (eval subst p2)
	Imply p1 p2 -> (not (eval subst p1)) || (eval subst p2)

-- Exer 4
vars :: Prop -> [Char] -- Note that [Char] will become String
vars (Const x) = []
vars (Var c) = [c]
vars (Not p) = vars p
vars (And p1 p2)   = elems $ fromList (vars p1 ++ vars p2)
vars (Or p1 p2)    = elems $ fromList (vars p1 ++ vars p2)
vars (Imply p1 p2) = elems $ fromList (vars p1 ++ vars p2)

boolGen :: Int -> [[Bool]]
boolGen n = [reverse [if (testBit num i) then True else False | i <- [(0::Int)..(n-1)]] | num <- [(0::Int)..(2^n-1)]]

substs :: Prop -> [Subst]
substs prop = reverse [zip (vars prop) boolp | boolp <- (boolGen $ length (vars prop))]

-- Exer 5
isTaut :: Prop -> Bool
isTaut p = (length [x | x <- [eval subst p | subst <- substs p], x == False]) == 0

-- Test
p4 :: Prop
p4 = Imply (Or (Var 'A') (Var 'B')) (And (Var 'C') (Var 'D'))

{- Results are shown below
> p4
((A||B)=>(C&&D))
> eval [('A',True),('B',False),('C',True),('D',False)] p4
False
> vars p4
"ABCD"
> substs p4
[[('A',True),('B',True),('C',True),('D',True)],[('A',True),('B',True),('C',True),('D',False)],[('A',True),('B',True),('C',False),('D',True)],[('A',True),('B',True),('C',False),('D',False)],[('A',True),('B',False),('C',True),('D',True)],[('A',True),('B',False),('C',True),('D',False)],[('A',True),('B',False),('C',False),('D',True)],[('A',True),('B',False),('C',False),('D',False)],[('A',False),('B',True),('C',True),('D',True)],[('A',False),('B',True),('C',True),('D',False)],[('A',False),('B',True),('C',False),('D',True)],[('A',False),('B',True),('C',False),('D',False)],[('A',False),('B',False),('C',True),('D',True)],[('A',False),('B',False),('C',True),('D',False)],[('A',False),('B',False),('C',False),('D',True)],[('A',False),('B',False),('C',False),('D',False)]]
> isTaut p4
False
-}

eval' :: Subst -> Prop -> Maybe Bool
eval' subst prop = if length [1 | var <- vars prop, length [var | c <- [fst c | c <- subst], var == c] == 0] /= 0
	then Nothing
	else case prop of
		Const x     -> Just x
		Var c       -> Just (snd (head [item | item <- subst, fst item == c]))
		Not p       -> Just (not (eval subst p))
		And p1 p2   -> Just ((eval subst p1) && (eval subst p2))
		Or p1 p2    -> Just ((eval subst p1) || (eval subst p2))
		Imply p1 p2 -> Just ((not (eval subst p1)) || (eval subst p2))
 -- eval' [('x', True)] (Var 'y') = Nothing