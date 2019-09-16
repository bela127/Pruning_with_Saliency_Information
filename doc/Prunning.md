Mit Hilfe der Weight Importance kann nun die Mitlere-Importance (MI) der Gewichte für verschiedene Inputs des Netzes hinweg bestimmt werden.

Gewichte die für den größten Teil der Inputs unwichtig waren, werden hier eine geringe MI erhalten.
Auch Gewichte die nur bei sehr wenigen Inputs wichtig waren und sonst immer unwichtig erhalten eine geringe MI. Solche Gewichte entstehen durch Overfitting, da sie nur auf einen ganz bestimmten Input reagieren. Eine geringe MI ist also auch angebracht.

Anhand der MI kann nun ein Pruning durchgeführt werden.
Hierzu werden alle Gewichte die unterhalb eines Pruning-Schwellwertes (PT) liegen entfernt. Der Schwellwert bildet sich dabei aus dem Median der (nicht entfernten) Gewichte eines einzelnen Neurons, der Sparcity (S) des Neurons und dem Sparcification-Factor (SF) einem Hyperparameter.

Die Sparcity berechnet sich dabei aus der Anzahl der ursprünglichen Gewichte zum Zeitpunkt 0 (C0) und der Anzahl der noch verbleibeden Gewichte zum Zeitpunkt t (Ct).

S = C0/Ct

Umso mehr Gewichte entfernt werden umso größer wird also die Sparcity, beginnend bei 1 läuft sie gegen Unendlich wenn alle Gewichte entfernt würden.

Der Sparcification-Factor gibt an bei wie viel Prozent des Medians der Pruning-Schwellwert liegen soll. Er wird allerdings mit der Sparcity potenziert was zu einem Decay führt so dass nie alle Gewichte entfernt werden können.

PT = Median * Pow(SF,S)

Umso weniger Gewichte noch vorhanden sind umso weniger Gewichte werden im nächsten Schritt entfernt.