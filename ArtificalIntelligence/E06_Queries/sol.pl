%% Q1
findall(Rest,branch(Rest,beigang),Z).

%% Q2
setof(Dist,Loc^Rest1^Rest2^Year1^Year2^Loc1^Loc2^(restaurant(Rest1,Year1,yuecai),restaurant(Rest2,Year2,xiangcai),branch(Rest1,Loc1),branch(Rest2,Loc2),district(Loc1,Dist),district(Loc2,Dist)),Res).

%% Q3
findall(MinRest,(numBranches(MinRest,MinNum),\+setof(SmallerRest,SmallerNum^(numBranches(SmallerRest,SmallerNum),MinNum > SmallerNum),Lst)),Z).

%% Q4
setof(Rest,branch(Rest,Area),Lst),length(Lst,Len),Len >= 2.

%% Q5
findall(MaxYearRest,(restaurant(MaxYearRest,MaxYear,T2),\+setof(LongerRest,T1^Year^(restaurant(LongerRest,Year,T1),Year > MaxYear),Lst)),Z).

%% Q6
setof(Rest,Num^(numBranches(Rest,Num),Num > 10),Z).