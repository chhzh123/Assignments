% Case 1
block(b1).
block(b2).
block(b3).
table(1).
table(2).
table(3).

start([clear(b2),clear(2),clear(3),on(b3,1),on(b1,b3),on(b2,b1)]).
end1([on(b1,1),on(b3,b1),on(b2,2),clear(b3),clear(b2),clear(3)]).

% end1(Goal), bestfirst(Goal->stop,Plan).