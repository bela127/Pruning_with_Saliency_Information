Entferne Gewichte nicht dauerhaft sondern mit einer Pruning Wahrscheinlichkeit. Diese wird anhand einer Rangfolge der Gewichte eines Neuron bestimmt, welche wiederum die L1 Norm (absolute Größe) der Gewichte als Importance verwendet.

Für alle Gewichte die in der Rangfolge unterhalb der Position liegen die durch ein Pruning Verhältnis vorgegeben ist wird die Wahrscheinlichkeit erhöht.

Pruning wird dann zufällig anhand der Wahrscheinlichkeiten durchgeführt, und das Netz weiter trainier. In jeder Iteration wird die Pruning Maske neu anhand der Wahrscheinlichkeiten festgelegt.