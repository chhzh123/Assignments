% Case 5
block(b1).
block(b2).
block(b3).
block(b4).
block(b5).
block(b6).
block(b7).
block(b8).
table(1).
table(2).
table(3).
table(4).
table(5).
table(6).
table(7).
table(8).

start([clear(b1),clear(2),clear(b6),clear(4),clear(b4),clear(b8),clear(7),clear(8),on(b1,1),on(b3,3),on(b2,b3),on(b6,b2),on(b5,5),on(b4,b5),on(b7,6),on(b8,b7)]).
end5([clear(b7),clear(2),clear(3),clear(4),clear(5),clear(6),clear(7),clear(8),on(b5,1),on(b8,b5),on(b6,b8),on(b3,b6),on(b1,b3),on(b4,b1),on(b2,b4),on(b7,b2)]).

% end5(Goal), bestfirst(Goal->stop,Plan).