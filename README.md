# Proyecto Fin de Grado
## Desarrollo de un algoritmo de machine learning para el control de una prótesis de mano mediante señales electromiográficas

El presente proyecto se enfoca en el desarrollo de un algoritmo predictivo mediante machine learning que, dada una señal de electromiografía de la región del antebrazo, sea capaz de predecir el movimiento deseado y, en última instancia, controlar una prótesis de mano a través de simulación. Este enfoque busca aprovechar las capacidades de extrapolación del algoritmo para ejecutar movimientos nuevos (no vistos durante el entrenamiento), adaptarse a situaciones ruidosas y permitir holgura en la disposición de los electrodos, facilitando su uso.

La pérdida o amputación de la mano conlleva una severa reducción en la autonomía del paciente. Hasta hace algunos años, la mayoría de las prótesis estaban limitadas a piezas inmóviles intercambiables o al aprovechamiento de la movilidad residual de la extremidad seccionada. Con el avance de la tecnología de electromiografía (EMG), se han desarrollado sistemas de control más sofisticados que permiten una mayor variedad de movimientos. Al integrar algoritmos de machine learning, es posible restaurar la movilidad y la autonomía del paciente de una manera mucho más natural y funcional.

El sistema propuesto empleará programación en Python para la implementación del modelo de control. Para el entrenamiento, se utilizarán señales de electromiografía en conjunto con visión artificial para captar y correlacionar los movimientos de la mano dentro del algoritmo. 

Posteriormente, la validación y aplicación del algoritmo se realizará en un entorno de simulación, evaluando su desempeño antes de su posible implementación en una prótesis real.

## Herramientas y dependencias empleadas:

### EMG:
* BrainVision Recorder
* BrainVision Analyzer 2.2

### Image Recognition:
* Python3
    - Mediapipe (junto a modelo de mano)
    - Pandas
    - OpenCV (cv2)
* Jupyter Notebooks
