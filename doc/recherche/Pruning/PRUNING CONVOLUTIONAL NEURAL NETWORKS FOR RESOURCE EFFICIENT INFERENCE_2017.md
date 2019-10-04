Iteriere zwischen Training und Pruning, entferne dabei immer die unwichtigsten Neuronen. Dabei wird versucht die Accuracy beizubehalten. Während des Trainings werden Neuronen und damit Featuremaps entfernt indem sie mit einem Pruning Gate (einer 0 – 1 Maske) multipliziert werden. Die Importance der Neuronen kann unterschiedlich bestimmt werden:

Oracle Pruning: Die „saliency“ jedes Neurons wird bestimmt indem jedes einzelne Neuron nacheinander entfernt und die Kostendifferenz bestimmt wird. Pruning der Neuronen die beim Entfernen eine geringe Kostendifferenz hervorrufen.

Sehr hoher Rechenaufwand deshalb weitere Vereinfachungen.

Minimum weight: Es wird die Höhe des durchschnittlichen Gewichts für jedes Neuron bestimmt und die Neuronen mit dem geringsten durchschnittlichem Gewicht entfernt.

Sehr einfach und sehr geringer Rechenaufwand. Kann mit L1 oder L2 Regularisierung unterstützt werden.

Activation: Neuronen die selten Aktiviert werden sind nicht so wichtig und können entfernt werden. Als Maß kann die Mittlere Aktivierung oder die Standarrdabweichung der Aktivierung jedes Neurons bestimmt werden.

Mutual information: Mutual Information (MI) oder für vereinfachte Berechnung Information Gain (IG) beschreibt die Abhängigkeit einer Variablen von einer anderen. Es wird die Abhängigkeit des Outputs des Netzes von der Aktivierung jedes Neurons für ein Trainings Sample bestimmt. Neuronen mit geringer Abhängigkeit werden entfernt.

Taylor expansion: Betrachtung als Optimierungsproblem eine bestimmte (und geringere) Anzahl an Parametern zu finden welche die absolute Differenz des Loss möglichst gering halten. Wobei der neue Loss nach entfernen des Parameters durch ein Taylor Polynom ersten Grades geschätzt wird. Es werden also Parameter bzw. Neuronen entfernt die einen geringen Gradient in Bezug auf den Loss in Kombination mit einer geringe Aktivierung haben. Für ein Trainings Sample wird dieser Wert für jedes Neuron gemittelt.

Average Percentage of Zeros (APoZ): Oben beschrieben, wie viel Aktivierungen sind im Durchschnitt 0. Unwichtige Neuronen werden selten aktiviert.

Liefert in frühen Schichten sehr ähnliche Werte für alle Neuronen.

→ Vergleich der verschiedenen Verfahren gegenübergestellt zum Oracel