:- discontiguous male/1, female/1, parent/2.

/* facts */
oneself(george,george).
oneself(mum,mum).
oneself(spencer,spencer).
oneself(kydd,kydd).
oneself(elizabeth,elizabeth).
oneself(philip,philip).
oneself(margaret,margaret).
oneself(diana,diana).
oneself(charles,charles).
oneself(anne,anne).
oneself(mark,mark).
oneself(andrew,andrew).
oneself(sarah,sarah).
oneself(edward,edward).
oneself(sophie,sophie).
oneself(william,william).
oneself(harry,harry).
oneself(peter,peter).
oneself(zara,zara).
oneself(beatrice,beatrice).
oneself(eugenie,eugenie).
oneself(louise,louise).
oneself(james,james).

%% sex
% #1 gen
male(george).
female(mum).
% #2 gen
female(elizabeth).
male(philip).
female(margaret).
male(spencer).
female(kydd).
% #3 gen
female(diana).
male(charles).
female(anne).
male(mark).
male(andrew).
female(sarah).
male(edward).
female(sophie).
% #4 gen
male(william).
male(harry).
male(peter).
female(zara).
female(beatrice).
female(eugenie).
female(louise).
male(james).

%% parent
parent(diana,spencer).
parent(diana,kydd).
parent(william,diana).
parent(william,charles).
parent(harry,diana).
parent(harry,charles).
parent(charles,elizabeth).
parent(charles,philip).
parent(anne,elizabeth).
parent(anne,philip).
parent(peter,anne).
parent(peter,mark).
parent(zara,anne).
parent(zara,mark).
parent(andrew,elizabeth).
parent(andrew,philip).
parent(beatrice,andrew).
parent(beatrice,sarah).
parent(eugenie,andrew).
parent(eugenie,sarah).
parent(louise,edward).
parent(louise,sophie).
parent(james,edward).
parent(james,sophie).
parent(edward,elizabeth).
parent(edward,philip).
parent(elizabeth,george).
parent(elizabeth,mum).
parent(margaret,george).
parent(margaret,mum).

/* rules */
child(X,Y) :- parent(Y,X).
father(X,Y) :- parent(X,Y), male(Y).
mother(X,Y) :- parent(X,Y), female(Y).
husband(X,Y) :- female(X), male(Y), father(Z,Y), mother(Z,X).
wife(X,Y) :- male(X), female(Y), father(Z,X), mother(Z,Y).
spouse(X,Y) :- husband(X,Y) ; wife(X,Y).
son(X,Y) :- child(X,Y), male(Y).
daughter(X,Y) :- child(X,Y), female(Y).
sibling(X,Y) :- parent(X,Z), parent(Y,Z), X \== Y.
brother(X,Y) :- sibling(X,Y), male(Y).
sister(X,Y) :- sibling(X,Y), female(Y).
grandfather(X,Y) :- parent(X,Z), father(Z,Y).
grandmother(X,Y) :- parent(X,Z), mother(Z,Y).
grandchild(X,Y) :- child(X,Z), child(Z,Y).
grandparent(X,Y) :- parent(X,Z), parent(Z,Y).
greatGrandparent(X,Y) :- grandparent(X,Z), parent(Z,Y).
ancestor(X,Y) :- parent(X,Y). % Base
ancestor(X,Y) :- parent(X,Z), ancestor(Z,Y). % Recursion
aunt(X,Y) :- grandparent(X,Z), parent(Y,Z), \+(mother(X,Y)), female(Y).
uncle(X,Y) :- grandparent(X,Z), parent(Y,Z), \+(father(X,Y)), male(Y).
%% the husband of your sister or brother
%% or the brother of your husband or wife
%% or the man who is married to the sister
%% or brother of your wife or husband
brotherInLaw(X,Y) :- (spouse(X,Z), brother(Z,Y)) ; (sister(X,Z), husband(Z,Y)).
sisterInLaw(X,Y) :- (spouse(X,Z), sister(Z,Y)) ; (brother(X,Z), wife(Z,Y)).
% cousins
firstCousin(X,Y) :- grandparent(X,Z), grandparent(Y,Z), X \== Y, \+(sibling(X,Y)).
mthAncestor(X,Y,0) :- oneself(X,Y).
mthAncestor(X,Y,1) :- parent(X,Y).
mthAncestor(X,Y,M) :- parent(X,Z), M1 is M-1, mthAncestor(Z,Y,M1).
mthCousin(X,Y,1) :- firstCousin(X,Y).
mthCousin(X,Y,M) :- M1 is M+1, mthAncestor(X,Z,M1), mthAncestor(Y,Z,M1), X \== Y,
					mthAncestor(X,A,M), mthAncestor(Y,B,M), A \= B, parent(A,Z), parent(B,Z).
					%% mthAncestor(X,A,M), mthAncestor(Y,B,M), A \= B, \+(spouse(A,B)).
mthCousinNremoved(X,Y,M,N) :- M1 is M+1, mthAncestor(X,Z,M1), M2 is M-N+1, mthAncestor(Y,Z,M2), X \== Y,
					mthAncestor(X,A,M), M3 is M-N, mthAncestor(Y,B,M3), parent(A,Z), parent(B,Z), A \== B.