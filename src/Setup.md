# Pakete Installieren
## Virtuelle Python3 Umgebung einrichten
Mit dem Befehl

	python3 -m venv venv
	
wird eine v-env im Ortner *venv* angelegt. Soll der Ordner anders benannt werden, kann der Name #FolderName# durch den entsprechenden Namen ersetzt werden.

	python3 -m venv #FolderName#

unter umständen ist venv auf dem System noch nicht instaliert, das kann wie folgt behoben werden.

	apt-get install python3-venv

## Aktiviren der v-env
Mit dem Befehl

	. venv/bin/activate

kann die v-env aktiviert werden. Nun können die nötigen Pakete instaliert werden.

## Benötigte Pakete
Folgende Pakete solten mit pip instaliert werden:

	pip install tensorflow

bzw. das Paket:

	pip install tensorflow-gpu

für GPU unterstützung