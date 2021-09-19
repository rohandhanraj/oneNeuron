# oneNeuron
Perceptron

Command to use for pushing the code to the GitHub

```bash
git add . && git commit -m "docstring updated" && git push origin main
```

```bash
cp Research\ notebooks/demo.ipynb .
```

### Add URL
[Git Handbook](https://https://guides.github.com/introduction/git-handbook/)

<a href="https://https://guides.github.com/introduction/git-handbook/">Git Handbook</a>


### Add Image
![Sample Image](plots\and.png)

<img src="plots\and.png">

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
