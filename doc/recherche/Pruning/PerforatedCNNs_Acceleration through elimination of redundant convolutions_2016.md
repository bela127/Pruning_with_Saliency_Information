Berechne Convolutions nur für bestimmte Positionen der Featuremap (Sparce Featuremap) im Gegensatz zum entfernen des kompletten Filters (Perforierte Convolutional Layer). Die fehlenden Werte werden mit den Nächsten Nachbarn der berechneten Positionen interpoliert. Dazu wird an zufällig Positionen evaluiert und mit den bleibenden Positionen mittels Euklidischer Distanz verglichen. Die Position wird dann durch den Wert der bleibenden Position die mit der geringsten Distanz ersetzt. Die bleibenden Positionen werden mittels einer Perforation Mask bestimmt, diese Maske kann auf verschiedene Weisen bestimmt werden:

Uniform: zufällig normalverteilte ausgewählte Positionen werden als Maske verwendet.
Grid: Ein zufälliges aber gleichmäßiges Gitter das mittels einer Pseudozufalls Zahlenfolge erstellt wird.

Pooling Structure: Beim Pooling überlappen einzelne Pooling Filter, es werden die Positionen gewählt die in möglichst vielen Filtern vorkommen, also an denen sich viele Filter überlappen.

Impact: Schätzen der Auswirkung des Entfernens einer Position auf den Loss und entfernen der Positionen mit geringster Auswirkung. Die Schätzung des Impacts wird mittels Tailor Polynom ersten Grades durchgeführt. Also der Gradient für diese Position multipliziert mit dem Wert der Position. Der Gradient wird dabei für die Perforierte Convolutional Layer ohne Interpolation berechnet. Für die Maske wird der Mittelwert des Impacts über ein Trainings Sample und die Summe über alle Kanäle an einer Position verwendet. Behalte dann nur die N Positionen mit dem höchsten Impact.

→ fine tuning des Netzes nach perforierung.
