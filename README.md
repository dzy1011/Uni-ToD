# From Retrieval to Generation: A Simple and Unified Generative Model for End-to-End Task-Oriented Dialogue

## We will update the code within two weeks


## Architecture

## How to Run it

### Environment
The environment configuration we used for training and testing is as follows:
```
transformers
tensorboard
nltk
sentencepiece
torch
```

### Training

The script train.py acts as a main function to the project, you can run the experiments by the following commands.

```bash
python train.py --dataset <dataset name>  --params_file config/gpt2/params.json --device cuda
```

### Evaluation
```bash
python eval.py --generate <path to the saved model> --dataset <dataset name>  --generation_params_file config/gpt2/generation_params.json --eval_dataset test  --output_file <the path to output file>
```

Due to some stochastic factors(e.g., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to reproduce the results reported in our paper. 


## Citation

If you use any source codes or the datasets included in this toolkit in your work, please cite the following paper. The bibtex are listed below:

```
@inproceedings{,
    title = "From Retrieval to Generation: A Simple and Unified Generative Model for End-to-End Task-Oriented Dialogue",
    year = "2024",
    address = "",
    publisher = "",
    url = "",

}
```

## Acknowledgement

We are highly grateful for the public code of the following papers, our code is partly based on them:

- **DialoKG: Knowledge-Structure Aware Task-Oriented Dialogue Generation.**

   Md Rashad Al Hasan Rony, Ricardo Usbeck, Jens Lehmann

   NAACL 2022 paper. [[PDF]](https://aclanthology.org/2022.findings-naacl.195.pdf) [[Code]](https://github.com/rashad101/DialoKG)


- **Q-TOD: A Query-driven Task-oriented Dialogue System.**

    Xin Tian, Yingzhan Lin, Mengfei Song, Siqi Bao, Fan Wang, Huang He, Shuqi Sun, Hua Wu.

    EMNLP 2022. [[Paper]](https://aclanthology.org/2022.emnlp-main.489.pdf) [[Code]](https://github.com/PaddlePaddle/Knover/tree/develop/projects/Q-TOD)


