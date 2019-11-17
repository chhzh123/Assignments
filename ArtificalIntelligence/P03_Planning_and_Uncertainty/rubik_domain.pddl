(define (domain rubik)

(:requirements :strips :typing :equality)

;     +++++++++
;    + 7 + 8 +|
;   +++++++++ |
;  + 5 + 6 +|+|
; +++++++++ + +
; | 5 | 6 |+|+
; +++++++++ +
; | 1 | 2 |+
; +++++++++
; fblrud

(:types color)
(:functions (cost) - number)

(:predicates
    (color1 ?f ?l ?d - color)
    (color2 ?f ?r ?d - color)
    (color3 ?b ?l ?d - color)
    (color4 ?b ?r ?d - color)
    (color5 ?f ?l ?u - color)
    (color6 ?f ?r ?u - color)
    (color7 ?b ?l ?u - color)
    (color8 ?b ?r ?u - color)
)

(:action R
    :effect (and
        (forall (?f2 ?r2 ?d2 - color) (when (color2 ?f2 ?r2 ?d2)
            (and
                (not (color2 ?f2 ?r2 ?d2))
                (color6 ?d2 ?r2 ?f2)
            ))
        )
        (forall (?b4 ?r4 ?d4 - color) (when (color4 ?b4 ?r4 ?d4)
            (and
                (not (color4 ?b4 ?r4 ?d4))
                (color2 ?d4 ?r4 ?b4)
            ))
        )
        (forall (?f6 ?r6 ?u6 - color) (when (color6 ?f6 ?r6 ?u6)
            (and
                (not (color6 ?f6 ?r6 ?u6))
                (color8 ?u6 ?r6 ?f6)
            ))
        )
        (forall (?b8 ?r8 ?u8 - color) (when (color8 ?b8 ?r8 ?u8)
            (and
                (not (color8 ?b8 ?r8 ?u8))
                (color4 ?u8 ?r8 ?b8)
            ))
        )
        (increase (cost) 1)
    )
)

(:action R_p
    :effect (and
        (forall (?f2 ?r2 ?d2 - color) (when (color2 ?f2 ?r2 ?d2)
            (and
                (not (color2 ?f2 ?r2 ?d2))
                (color4 ?d2 ?r2 ?f2)
            ))
        )
        (forall (?b4 ?r4 ?d4 - color) (when (color4 ?b4 ?r4 ?d4)
            (and
                (not (color4 ?b4 ?r4 ?d4))
                (color8 ?d4 ?r4 ?b4)
            ))
        )
        (forall (?f6 ?r6 ?u6 - color) (when (color6 ?f6 ?r6 ?u6)
            (and
                (not (color6 ?f6 ?r6 ?u6))
                (color2 ?u6 ?r6 ?f6)
            ))
        )
        (forall (?b8 ?r8 ?u8 - color) (when (color8 ?b8 ?r8 ?u8)
            (and
                (not (color8 ?b8 ?r8 ?u8))
                (color6 ?u8 ?r8 ?b8)
            ))
        )
        (increase (cost) 1)
    )
)

(:action U
    :effect (and
        (forall (?f5 ?l5 ?u5 - color) (when (color5 ?f5 ?l5 ?u5)
            (and
                (not (color5 ?f5 ?l5 ?u5))
                (color7 ?l5 ?f5 ?u5)
            ))
        )
        (forall (?f6 ?r6 ?u6 - color) (when (color6 ?f6 ?r6 ?u6)
            (and
                (not (color6 ?f6 ?r6 ?u6))
                (color5 ?r6 ?f6 ?u6)
            ))
        )
        (forall (?b7 ?l7 ?u7 - color) (when (color7 ?b7 ?l7 ?u7)
            (and
                (not (color7 ?b7 ?l7 ?u7))
                (color8 ?l7 ?b7 ?u7)
            ))
        )
        (forall (?b8 ?r8 ?u8 - color) (when (color8 ?b8 ?r8 ?u8)
            (and
                (not (color8 ?b8 ?r8 ?u8))
                (color6 ?r8 ?b8 ?u8)
            ))
        )
        (increase (cost) 1)
    )
)

(:action U_p
    :effect (and
        (forall (?f5 ?l5 ?u5 - color) (when (color5 ?f5 ?l5 ?u5)
            (and
                (not (color5 ?f5 ?l5 ?u5))
                (color6 ?l5 ?f5 ?u5)
            ))
        )
        (forall (?f6 ?r6 ?u6 - color) (when (color6 ?f6 ?r6 ?u6)
            (and
                (not (color6 ?f6 ?r6 ?u6))
                (color8 ?r6 ?f6 ?u6)
            ))
        )
        (forall (?b7 ?l7 ?u7 - color) (when (color7 ?b7 ?l7 ?u7)
            (and
                (not (color7 ?b7 ?l7 ?u7))
                (color5 ?l7 ?b7 ?u7)
            ))
        )
        (forall (?b8 ?r8 ?u8 - color) (when (color8 ?b8 ?r8 ?u8)
            (and
                (not (color8 ?b8 ?r8 ?u8))
                (color7 ?r8 ?b8 ?u8)
            ))
        )
        (increase (cost) 1)
    )
)

(:action F
    :effect (and
        (forall (?f1 ?l1 ?d1 - color) (when (color1 ?f1 ?l1 ?d1)
            (and
                (not (color1 ?f1 ?l1 ?d1))
                (color5 ?f1 ?d1 ?l1)
            ))
        )
        (forall (?f2 ?r2 ?d2 - color) (when (color2 ?f2 ?r2 ?d2)
            (and
                (not (color2 ?f2 ?r2 ?d2))
                (color1 ?f2 ?d2 ?r2)
            ))
        )
        (forall (?f5 ?l5 ?u5 - color) (when (color5 ?f5 ?l5 ?u5)
            (and
                (not (color5 ?f5 ?l5 ?u5))
                (color6 ?f5 ?u5 ?l5)
            ))
        )
        (forall (?f6 ?r6 ?u6 - color) (when (color6 ?f6 ?r6 ?u6)
            (and
                (not (color6 ?f6 ?r6 ?u6))
                (color2 ?f6 ?u6 ?r6)
            ))
        )
        (increase (cost) 1)
    )
)

(:action F_p
    :effect (and
        (forall (?f1 ?l1 ?d1 - color) (when (color1 ?f1 ?l1 ?d1)
            (and
                (not (color1 ?f1 ?l1 ?d1))
                (color2 ?f1 ?d1 ?l1)
            ))
        )
        (forall (?f2 ?r2 ?d2 - color) (when (color2 ?f2 ?r2 ?d2)
            (and
                (not (color2 ?f2 ?r2 ?d2))
                (color6 ?f2 ?d2 ?r2)
            ))
        )
        (forall (?f5 ?l5 ?u5 - color) (when (color5 ?f5 ?l5 ?u5)
            (and
                (not (color5 ?f5 ?l5 ?u5))
                (color1 ?f5 ?u5 ?l5)
            ))
        )
        (forall (?f6 ?r6 ?u6 - color) (when (color6 ?f6 ?r6 ?u6)
            (and
                (not (color6 ?f6 ?r6 ?u6))
                (color5 ?f6 ?u6 ?r6)
            ))
        )
        (increase (cost) 1)
    )
)

)