-- 陈鸿峥 17341015, chenhzh37@mail2.sysu.edu.cn, 数据科学计算机学院, 计算机类
-- 实验一: Fraction

module MyFraction where
import Test.QuickCheck
type Fraction = (Integer, Integer)

ratplus :: Fraction -> Fraction -> Fraction
ratplus x y = ((fst x) * (snd y) + (snd x) * (fst y), (snd x) * (snd y))

ratminus :: Fraction -> Fraction -> Fraction
ratminus x y = ((fst x) * (snd y) - (snd x) * (fst y), (snd x) * (snd y))

rattimes :: Fraction -> Fraction -> Fraction
rattimes x y = ((fst x) * (fst y), (snd x) * (snd y))

ratdiv :: Fraction -> Fraction -> Fraction
ratdiv x y = ((fst x) * (snd y), (snd x) * (fst y))

ratfloor :: Fraction -> Integer
ratfloor x = (fst x) `div` (snd x)

ratfloat :: Fraction -> Float
ratfloat x = (fromInteger (fst x)) / (fromInteger (snd x))

rateq :: Fraction -> Fraction -> Bool
rateq x y = (ratfloat (ratminus x y) == 0.0)

infix 5 <+>
(<+>) :: Fraction -> Fraction -> Fraction
(<+>) (a,b) (c,d) = ratplus (a,b) (c,d)

infix 5 <->
(<->) :: Fraction -> Fraction -> Fraction
(<->) (a,b) (c,d) = ratminus (a,b) (c,d)

infix 6 <-*->
(<-*->) :: Fraction -> Fraction -> Fraction
(<-*->) (a,b) (c,d) = rattimes (a,b) (c,d)

infix 6 </>
(</>) :: Fraction -> Fraction -> Fraction
(</>) (a,b) (c,d) = ratdiv (a,b) (c,d)

infix 4 <==>
(<==>) :: Fraction -> Fraction -> Bool
(<==>) (a,b) (c,d) = rateq (a,b) (c,d)

-- test
-- prop_ratplus_unit :: Fraction -> Property
-- prop_ratplus_unit (a,b) = b > 0 ==> (a,b) <+> (0,1) <==> (a,b)

-- prop_ratplus_plus_distr :: Fraction -> Fraction -> Fraction -> Property
-- prop_ratplus_plus_distr (a,b) (c,d) (e,f) = b > 0 && d >0 && f > 0 ==> (a,b) <-*-> ((c,d) <+> (e,f)) <==> ((a,b) <-*-> (c,d)) <+> ((a,b) <-*-> (e,f))