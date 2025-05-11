# Advanced NLP Exercise 1: Fine Tuning

This is the code base for ANLP HUJI course exercise 1, fine tuning pretrained models to perform sentiment analysis on the SST2 dataset.

# Install
``` pip install -r requirements.txt ```

# Fine-Tune and Predict on Test Set
Run:

``` python ex1.py --max_train_samples 50 --max_eval_samples 50 --max_predict_samples 50 --lr 1e-4 --num_train_epochs 5 --batch_size 16 --do_train 1 --do_predict  --model_path bert-base-uncased```

If you use --do_predict, a prediction.txt file will be generated, containing prediction results for all test samples.
