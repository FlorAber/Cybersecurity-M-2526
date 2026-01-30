# Project sum-up

In real-world contexts, data is constantly evolving, especially in highly dynamic
environments such as cybersecurity. In these scenarios, continuous learning is an
interesting approach that allows models to continuously learn from new data streams.
However, the environments from which this data originates are often vulnerable, such
as IoT systems. In these systems, it can be easy to change the input data for machine
learning models, exposing the models to new types of attacks.
Another risk is the "catastrophic forgetting" problem, a typical problem in continuous
learning, which occurs when a model forgets previously learned concepts as it attempts
to learn new ones. This phenomenon not only affects the performance of the model, but
can also be exploited as an attack vector. A malicious actor could intentionally alter the
input data to cause the model to "forget" the ability to detect certain threats or
vulnerabilities, thereby reducing the effectiveness of the attack detection system.

## Goals

- Use a machine learning model to implement a simple class incremental learning system.
- Implement a data poisoning attack within a class incremental learning based
  system to compromise the model's ability to correctly detect cyberattacks. The
  attack will manipulate the input data to progressively degrade the model's
  effectiveness.
- Explore and exploit the catastrophic forgetting problem: Analyze how the model
  loses knowledge of previously detected attacks when exposed to "poisoned"
  data. The goal is to understand, through various experiments, how much
  "poisoned" data is required for the model to suffer a significant drop in its threat
  detection capabilities.

## Datasets

- [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- [ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets)

## Repository structure

- [resources](./resources) : Directory contenente i grafici che descrivono i dataset utilizzati
- [src](./src) : Directory contenente il codice sorgente, in particolare si possono trovare:
  - [checkpoints](./src/checkpoints) : cartella contenente una copia dei modelli addestrati senza dati poisoned, sono serviti come punto di partenza per le valutazioni con dati avvelenati
  - [dataset_preparation](./src/dataset_preparation) : script numerati in ordine di esecuzione per creare i dataset e visualizzare le statistiche
  - [1 - training.py] : Script per addestrare il modello
  - [2a - labelflipping.py] : Script per Targeted Data Poisoning Attack con Label flipping
  - [2b - availability.py] : Script per Availability Attack
  - [2c - backdoor.py] : Script per attacco Backdoor
