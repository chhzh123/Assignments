% Case 4
block(b1).
block(b2).
block(b3).
block(b4).
block(b5).
block(b6).
table(1).
table(2).
table(3).
table(4).
table(5).
table(6).

start([clear(b1),clear(2),clear(b6),clear(4),clear(b4),clear(6),on(b1,1),on(b3,3),on(b2,b3),on(b6,b2),on(b5,5),on(b4,b5)]).
end4([clear(b6),clear(2),clear(3),clear(4),clear(5),clear(6),on(b5,1),on(b3,b5),on(b1,b3),on(b4,b1),on(b2,b4),on(b6,b2)]).

% end4(Goal), bestfirst(Goal->stop,Plan).