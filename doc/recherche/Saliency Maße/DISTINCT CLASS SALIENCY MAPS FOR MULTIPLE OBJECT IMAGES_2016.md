
→ anpassen des verfahrens aus xxx 2014 xxx, anstelle die gradient Saliency Map nur für den Input zu berechnen wird sie für die ersten x Layer Inputs also Feature Maps berechnet. Die so entstandenen Maps werden dann auf die selbe Größe gebracht (bilineares upsampling). Es wird nun das Maximum des Absolutwertes aus den Kanälen gewählt um eine 1 Kanalige Map zu erhalten. Diese einkanaligen Maps jeder Feature Layer werden dann zu einer Saliency Map kombiniert (gemittelt über den tanh aller einzelnen Mpas).

Xxx gleichung 3 xxxx

Es wurde festgestellt das eine Klassenspezifische Map oft auch noch Bereiche anderer Klassen abdeckt. Aus diesem Grund werden die Maps aller Klassen berechnet und von der relevanten Klasse die Maps der anderen Klassen abgezogen.

Xxx gleichung 2 xxx

Für Relu Aktivierungen wurde die Guided Backpropergation eingesetzt bei der nur positive Gradienten zurück propagiert werden.