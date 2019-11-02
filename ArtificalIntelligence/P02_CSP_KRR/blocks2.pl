% Case 2
block(b1).
block(b2).
block(b3).
block(b4).
block(b5).
table(1).
table(2).
table(3).
table(4).
table(5).

start([clear(b1),clear(b3),clear(3),clear(4),clear(5),on(b2,1),on(b5,b2),on(b1,b5),on(b4,2),on(b3,b4)]).
end2([clear(1),clear(b2),clear(3),clear(b4),clear(5),on(b3,2),on(b1,b3),on(b2,b1),on(b5,4),on(b4,b5)]).

% end2(Goal), bestfirst(Goal->stop,Plan).