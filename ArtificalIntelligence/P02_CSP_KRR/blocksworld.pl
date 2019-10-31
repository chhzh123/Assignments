use_module(library(lists)).

object(X) :- table(X) ; block(X).

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

% facts
block(b1).
block(b2).
block(b3).
% block(b4).
% block(b5).
% block(b6).
% block(b7).
% block(b8).
table(1).
table(2).
table(3).
% table(4).
% table(5).
% table(6).
% table(7).
% table(8).

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

% Start = [clear(2),clear(4),clear(b),clear(c),on(a,1),on(b,3),on(c,a)], plan(Start,[on(a,b),on(b,c)],_,Plan,_).
% Start = [clear(b2),clear(2),clear(3),on(b3,1),on(b1,b3),on(b2,b1)], Goal = [on(b1,1),on(b3,b1),on(b2,2),clear(b3),clear(b2),clear(3)], plan(Start,Goal,Plan).
% Start = [clear(b1),clear(b3),clear(3),clear(4),clear(5),on(b2,1),on(b5,b2),on(b1,b5),on(b4,2),on(b3,b4)], Goal = [clear(1),clear(b2),clear(3),clear(b4),clear(5),on(b3,2),on(b1,b3),on(b2,b1),on(b5,4),on(b4,b5)], plan(Start,Goal,Plan).
% Start = [clear(b1),clear(b3),clear(3),clear(4),clear(5),on(b2,1),on(b5,b2),on(b1,b5),on(b4,2),on(b3,b4)], Goal = [clear(1),clear(b4),clear(3),clear(4),clear(5),on(b2,2),on(b1,b2),on(b5,b1),on(b3,b5),on(b4,b3)]
% Start = [clear(b1),clear(2),clear(b6),clear(4),clear(b4),clear(6),on(b1,1),on(b3,3),on(b2,b3),on(b6,b2),on(b5,5),on(b4,b5)], Goal = [clear(b6),clear(2),clear(3),clear(4),clear(5),clear(6),on(b5,1),on(b3,b5),on(b1,b3),on(b4,b1),on(b2,b4),on(b6,b2)]
% Start = [clear(b1),clear(2),clear(b6),clear(4),clear(b4),clear(b8),clear(7),clear(8),on(b1,1),on(b3,3),on(b2,b3),on(b6,b2),on(b5,5),on(b4,b5),on(b7,6),on(b8,b7)], Goal = [clear(b7),clear(2),clear(3),clear(4),clear(5),clear(6),clear(7),clear(8),on(b5,1),on(b8,b5),on(b6,b8),on(b3,b6),on(b1,b3),on(b4,b1),on(b2,b4),on(b7,b2)]
% plan(Start,Goal,Plan)