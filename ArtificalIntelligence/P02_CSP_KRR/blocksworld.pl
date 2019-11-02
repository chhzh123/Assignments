use_module(library(lists)).

object(X) :-
    block(X) ; table(X).

% can(Action,Precondition)
can(move(Block,From,To),[clear(Block),clear(To),on(Block,From)]) :-
    block(Block), % Block to be moved
    object(To), % Table or block
    To \== Block,
    object(From),
    From \== To,
    Block \== From.

% STRIPS (Action,Effect)
adds(move(X,From,To),[on(X,To),clear(From)]).
deletes(move(X,From,To),[on(X,From),clear(To)]).

% BFS planner
plan(State,Goals,[]) :-
    satisfied(State,Goals).

% append(?List1, ?List2, ?List1AndList2)
% List1AndList2 is the concatenation of List1 and List2
plan(State,Goals,Plan) :-
    append(PrePlan,[Action],Plan),
    select(State,Goals,Goal),
    achieves(Action,Goal),
    can(Action,Condition),
    preserves(Action,Goals),
    regress(Goals,Action,RegressedGoals),
    plan(State,RegressedGoals,PrePlan).

satisfied(State,Goals) :-
    delete_all(Goals,State,[]).

select(State,Goals,Goal) :-
    member(Goal,Goals).

achieves(Action,Goal) :-
    adds(Action,Goals),
    member(Goal,Goals).

preserves(Action,Goals) :-
    deletes(Action,Relations),
    not((member(Goal,Relations),member(Goal,Goals))).

regress(Goals,Action,RegressedGoals) :-
    adds(Action,NewRelations),
    delete_all(Goals,NewRelations,RestGoals),
    can(Action,Condition),
    addnew(Condition,RestGoals,RegressedGoals).

% addnew(NewGoals,OldGoals,AllGoals)
addnew([],L,L).
addnew([Goal|_],Goals,_) :-
    impossible(Goal,Goals),!,fail.

addnew([X|L1],L2,L3) :-
    member(X,L2),!,addnew(L1,L2,L3).

addnew([X|L1],L2,[X|L3]) :-
    addnew(L1,L2,L3).

impossible(on(X,X),_).
impossible(on(X,Y),Goals) :-
    member(clear(Y),Goals);
    member(on(X,Y1),Goals), Y1\==Y;
    member(on(X1,Y),Goals), X1\==X.

impossible(clear(X),Goals) :-
    member(on(_,X),Goals).

% delete_all(L1,L2,Diff)
delete_all([],_,[]).
delete_all([X|L1],L2,Diff) :-
    member(X,L2),!,
    delete_all(L1,L2,Diff).

delete_all([X|L1],L2,[X|Diff]) :-
    delete_all(L1,L2,Diff).

% below(X,X,_).
% X is below Y
below(X,Y,State) :-
    X == Y ;
    (X \== Y,
    block(X),
    member(on(Z,X),State),
    below(Z,Y,State)).

isGoalPos(X, State, Goals) :-
    table(X) ;
    (member(on(X,Y),State),
    member(on(X,Y),Goals),
    isGoalPos(Y,State,Goals)).

% Best-first search
:- op(300,xfy,->).

s(Goals->NextAction,NewGoals->Action,1) :- % all costs are 1
    member(Goal,Goals),
    achieves(Action,Goal),
    can(Action,Condition),
    preserves(Action,Goals),
    regress(Goals,Action,NewGoals).

goal(Goals->Action) :-
    start(State),
    satisfied(State,Goals).

h1(H1,State,Goals) :-
    findall(X,(block(X),not(isGoalPos(X,State,Goals))),NotGoalPos),
    length(NotGoalPos,H1),
    !.

h2(H2,State,Goals) :-
    findall(X,(block(X),not(isGoalPos(X,State,Goals)),below(Y,X,State),below(Y,X,Goals)),NotGoalPos),
    length(NotGoalPos,H2),
    !.

h(Goals->Action,H) :-
    start(State),
    h1(H1,State,Goals),
    h2(H2,State,Goals),
    H is H1 + H2.