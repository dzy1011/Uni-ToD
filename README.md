# From Retrieval to Generation: A Simple and Unified Generative Model for End-to-End Task-Oriented Dialogue


## Architecture

## How to Run it

### Environment

### Training

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

    NAACL 2022 paper. [[PDF]](https://aclanthology.org/2022.findings-naacl.195.pdf)


- **Global-to-local Memory Pointer Networks for Task-Oriented Dialogue.**

    Chien-Sheng Wu, Richard Socher, Caiming Xiong.

    ICLR 2019. [[Paper]](https://arxiv.org/abs/1901.04713) [[Open Reivew]](https://openreview.net/forum?id=ryxnHhRqFm) [[Code]](https://github.com/jasonwu0731/GLMP)


