Für jede Schicht wird der Gradient des Loss (J) in Bezug auf den Input der Schicht (X) bestimmt, dies ergibt die Saliency Map (SM).

SM = (dJ/dX)

Große positive und negative Gradienten deuten beide auf große Auswirkungen des Inputs auf den Loss hin, es wird daher der Absolutwert gebildet um die Importance des Inputs (Ix) zu bestimmen.

Ix = Abs(SM)

Aus der Importance des Inputs einer Schicht (Ix) und der Importance des Inputs der Folgeschicht (Iy) kann nun die Importance (Iw) der einzelnen Gewichte (W) berechnet werden. Hierbei ist zu beachten, dass der Input der Folgeschicht immer dem Output (Y) der vorhergehenden Schicht entspricht.

Ist der Input der Folgeschicht (Yy) unwichtig (= 0) sind folglich alle Gewichte (Wxy) die zu diesem Input führen ebenfals unwichtig.
Ähnlich ist ein Gewicht (Wxy) das einen unwichtigen Input (Xx) mit einem belibigen Input der Folgeschicht (Yy) verbindet ebenfals unwichtig.
Ist der Input (Yy) wichtig  (= 1) und der Input der Vorgängerschicht (Xx) ebenfals wichtig muss auch das Gewicht (Wxy) welches die beiden verbindet wichtig sein.

Es muss also einfach die Importance der beiden Schichten komponentenweise multipliziert werden um die Importance der Gewichte zu ermitteln.



