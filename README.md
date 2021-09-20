# oneNeuron
Perceptron
1. Perceptron Implimentation
2. Perceptron Implimentation | Python scripting and packaging | Modular coding
3. Python logging basics , docstrings
4. README.md | Documentation


Command to use for pushing the code to the GitHub
```bash
git add . && git commit -m "docstring updated" && git push origin main
```

Command to use for cloning a repository
```bash
git clone https://github.com/<repository>
```

Command to use for folders creation in a repository
```bash
mkdir <folder name>
```

Command to use for folders within a folder creation in a repository
```bash
mkdir -p <parent folder name / folder name>
```

Command to use for files within a folder creation in a repository
```bash
touch <folder path/file name>
```

```bash
cp Research\ notebooks/demo.ipynb .
```

### Add URL
[Git Handbook](https://guides.github.com/introduction/git-handbook/)

<a href="https://github.com/git-guides/">Git Guide</a>


### Add Image
<img src="plots\and.png" alt="AND Plot" width="500" height="600">

![AND Plot Image](plots\and.png)

### Python Example
```python
def main(data, eta, epochs, filename, plotFilename):

   
    df = pd.DataFrame(data)
    logging.info(df)

    X, y = prepare_data(df)
    

    model = Perceptron(eta = eta, epochs = epochs)
    model.fit(X, y)

    _ = model.total_loss()

    save_model(model, filename = filename)

    save_plot(df, plotFilename, model)
```

### Data Table
|x1|x2|y|
|-|-|-|
|0|0|0|
|0|1|0|
|1|0|0|
|1|1|1|

### Points
* Point1
* Point2

1. Point1
2. Point2
