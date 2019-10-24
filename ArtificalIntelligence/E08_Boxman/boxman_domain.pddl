(define (domain boxman)

(:requirements :strips :typing :equality)

(:predicates
    (inc ?p ?p1)
    (dec ?p ?p1)
    (empty ?x ?y)
    (box ?x ?y)
    (pos ?x ?y)
)

(:action move-down
    :parameters (?x ?y ?yTo)
    :precondition (and
        (pos ?x ?y)
        (dec ?y ?yTo)
        (empty ?x ?yTo)
        )
    :effect (and
        (not (pos ?x ?y))
        (pos ?x ?yTo)
        (not (empty ?x ?yTo))
        (empty ?x ?y)
        )
)

(:action move-up
    :parameters (?x ?y ?yTo)
    :precondition (and
        (pos ?x ?y)
        (inc ?y ?yTo)
        (empty ?x ?yTo)
        )
    :effect (and
        (not (pos ?x ?y))
        (pos ?x ?yTo)
        (not (empty ?x ?yTo))
        (empty ?x ?y)
        )
)

(:action move-left
    :parameters (?x ?y ?xTo)
    :precondition (and
        (pos ?x ?y)
        (dec ?x ?xTo)
        (empty ?xTo ?y)
        )
    :effect (and
        (not (pos ?x ?y))
        (pos ?xTo ?y)
        (not (empty ?xTo ?y))
        (empty ?x ?y)
        )
)

(:action move-right
    :parameters (?x ?y ?xTo)
    :precondition (and
        (pos ?x ?y)
        (inc ?x ?xTo)
        (empty ?xTo ?y)
        )
    :effect (and
        (not (pos ?x ?y))
        (pos ?xTo ?y)
        (not (empty ?xTo ?y))
        (empty ?x ?y)
        )
)

; right x+ up y+
(:action push-down
    :parameters (?x ?y ?yBox ?yTo)
    :precondition (and
        (pos ?x ?y)
        (dec ?y ?yBox)
        (dec ?yBox ?yTo)
        (box ?x ?yBox)
        (empty ?x ?yTo)
        )
    :effect (and
        (not (pos ?x ?y))
        (pos ?x ?yBox)
        (not (box ?x ?yBox))
        (box ?x ?yTo)
        (not (empty ?x ?yTo))
        (empty ?x ?y)
        )
)

(:action push-up
    :parameters (?x ?y ?yBox ?yTo)
    :precondition (and
        (pos ?x ?y)
        (inc ?y ?yBox)
        (inc ?yBox ?yTo)
        (box ?x ?yBox)
        (empty ?x ?yTo)
        )
    :effect (and
        (not (pos ?x ?y))
        (pos ?x ?yBox)
        (not (box ?x ?yBox))
        (box ?x ?yTo)
        (not (empty ?x ?yTo))
        (empty ?x ?y)
        )
)

(:action push-left
    :parameters (?x ?y ?xBox ?xTo)
    :precondition (and
        (pos ?x ?y)
        (dec ?x ?xBox)
        (dec ?xBox ?xTo)
        (box ?xBox ?y)
        (empty ?xTo ?y)
        )
    :effect (and
        (not (pos ?x ?y))
        (pos ?xBox ?y)
        (not (box ?xBox ?y))
        (box ?xTo ?y)
        (not (empty ?xTo ?y))
        (empty ?x ?y)
        )
)

(:action push-right
    :parameters (?x ?y ?xBox ?xTo)
    :precondition (and
        (pos ?x ?y)
        (inc ?x ?xBox)
        (inc ?xBox ?xTo)
        (box ?xBox ?y)
        (empty ?xTo ?y)
        )
    :effect (and
        (not (pos ?x ?y))
        (pos ?xBox ?y)
        (not (box ?xBox ?y))
        (box ?xTo ?y)
        (not (empty ?xTo ?y))
        (empty ?x ?y)
        )
)

)