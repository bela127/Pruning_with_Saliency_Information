Saliency laut Duden:

das Hervortreten, Herausstechen eines Reizes; Auffälligkeit eines Ereignisses, einer Sache oder Person

In bezug auf neuronale Netze sind Saliency Maps, Karten die anzeigen welche Bereiche eines Bildes ein neuronales Netz besonders Reizen bzw. Bereiche die das Netz als herrausstechend für die Entscheidungsfindung sieht.

Im idealfall solten sich diese wichtige Bereiche im Bild mit den zu erkennenden Objekten decken.

Sie können auf diese weise für verschidene dinge eingesetzt werden:
Schwachüberwachtes Lernen
Erklärung von Netzen
Fehler findung und Prüfung von Netzen
Prunning
Objekt lokalisierung

Im einfachsten fall wird zum erstellen einer Saliency Map der Gradient des Loss bezogen auf den Input gebildet.
Bereiche des Inputs die zu einer Starken änderung der Entscheidung und somit des Loss führen sind offensichtlich wichtiger für die Entscheidung.
