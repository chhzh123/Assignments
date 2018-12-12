module TestLab3 where  -- used to test isTaut

import Lab3


-- for every p in someTauts, isTaut p => True 
someTauts :: [Prop]
someTauts = [
    Const True,
    Not (Const False),
    Or (Var 'p') (Not (Var 'p')),
    Or (Const True) (Var 'q'),
    Or (Var 'p') (Const True),
    Or (Const True) (And (Var 'p')(Var 'q')),
    And (Const True) (Const True),
    And (Const True) (Not (Const False)),
    Imply (Var 'p') (Const True),
    Imply (Var 'p') (Var 'p'),
    Imply (Var 'p') (Or (Var 'p') (Var 'q')),
    Imply (Not (Not (Var 'p'))) (Var 'p'),
    Imply (Var 'p') (Or (Var 'q') (Var 'p')),
    Imply (Var 'p') (Imply (Var 'q')(Var 'p'))
     ]

-- for every p in nonTauts, isTaut p => False

nonTauts :: [Prop]
nonTauts = [
    Const False,
    Not (Const True),
    Var 'p',
    Not (Var 'q'),
    Not (And (Const True) (Const True)),
    Not (And (Const True) (Var 'p')),
    Not (Or (Var 'p') (Const True)),
    And (Const True) (Const False),
    And (Const False) (Var 'p'),
    And (Var 'p') (Var 'p'),
    And (Var 'p') (Not (Var 'q')),
    Imply (Const True) (Const False),
    Imply (Const True) (Var 'p'),
    Not (Imply (Var 'p') (Const True)) ]

-- type check_it in your interpreter, if you see a "True", then 10 points, otherwise 0 points for part 5.  

check_it :: IO ()
check_it = do
    let a = and (map isTaut someTauts) -- all are True
    let b = or (map isTaut nonTauts)  -- all are False
    let  c = a && (not b) -- should be True
    print c


          

