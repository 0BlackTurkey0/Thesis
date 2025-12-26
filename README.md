# Code Summary


## How to Run?
Just run the target file directly with packages of **requirements.txt** on Python 3.10 or above

for reference: `pip install -r requirements.txt`


## Main Processes in Thesis

* **CoAtNetTraining.py**: CoAtNet (feature extractor) in thesis, including Cifar-100 pre-training (used in thesis) and Cifar-100 to Cifar-10 transfer learning / fine-tuning (only used to validate accuracy of CoAtNet)

* **CoAtNetFeatureExtraction.py**: Extract features from Cifar-10 dataset by Cifar-100 pre-training CoAtNet-0

* **EnsembleDistributedLearning.py**: EDL (proposed method) in thesis

* **FederatedLearning.py**: FL (comparison method) in thesis

* **MetricPlot.py**: Draw plots to compare the performance of different metrics under various independent variables


## Libraries for Main Processes

* **`Utils`**
    * **Utils**: Utilities for distributing dataset based on IID / non-IID setting and calculating metrics

* **`Models`**
    * **CoAtNet.py**: Define the class of CoAtNet-based models, including constructors for CoAtNet-0 through CoAtNet-7
    * **LightweightModels.py**: Define constructors for simple nn models, including the Sigmoid binary classifier and the Softmax multi-class classifier

* **`Clients`**
    * **Clients.py**: Define the class of clients with their threat type (normal or anomaly), private dataset, and local model
    
* **`Simulators`**
    * **ProfitSimulator.py**: profitPct equation in thesis, including its test cases for validation


## Data of Experiments

* **`History`**
    * Metrics history generated from **CoAtNetTraining.py**

* **`Results`**
    * EDL / FL results generated from **EnsembleDistributedLearning.py** / **FederatedLearning.py**

* **`Figures`**
    * Comparison figures generated from **MetricPlot.py**
