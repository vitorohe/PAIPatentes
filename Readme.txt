El modelo SVM no está incluido en el zip por temas de tamaño.
Al ejecutar el entrenamiento del modelo (descrito abajo) se generará un archivo llamado model.xml.


Modo gráfico:

	El proyecto cuenta con una interfaz gráfica en Qt. Viene una carpeta llamada QT con un archivo .pro, Patente.pro que es el que permite abrirlo en qtCreator.
	Al ejecutar se abre una ventana, en que se puede elegir una imagen para buscar su patente y reconocer sus caracteres, y también tiene la opción de entrenar el modelo SVM y probarlo.

Modo Terminal:

	Antes de ejecutar cualquier comando, es necesario cambiar algunas rutas dentro del código:
		En svm_model.cpp hay que quitar "../../" de las rutas "../../patentes" y "../../no_patente".
		En knearest.cpp hay que quitar "../../" de la ruta "../../letras/caracteres"

	Para ejecutar por terminal, ejecutar cmake ., make según sea necesario.
	
	Para entrenar modelo:
		./main -T
	
	Para probar modelo con una imagen:
		./main -t <imagen>
	
	Para ejecutar búsqueda de patente en una imagen:
		./main -s <imagen>
