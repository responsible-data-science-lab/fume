CTR (Click-Through Rate) Dataset
---
This dataset consists of one day of online ad clicks (day 0), [dataset homepage](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

* To download this dataset, run the following command:
	* `wget -c http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_0.gz`, then unzip this file.

* Preprocess the data.
	* Run `python3 continuous.py`.

This creates a `continuous/train.npy` and `continuous/test.npy`.

Example analysis of this dataset can be found on this [blog post](https://medium.com/@marthawhite_81346/learning-with-the-criteo-tb-dataset-e3ec12d9d77e).