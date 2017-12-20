
const Neuron = require('../lib/neuron');
const Layer = require('../lib/layer');
const activation = require('../lib/activation');
const initializer = require('../lib/initializer');

class InputLayer extends Layer {
  constructor(size) {
    super(0);
    this.type = 'InputLayer';
  }
  output(x) {
    return x;
  }
}

class FullyConnectedLayer extends Layer {
  constructor(inputSize, neuronCount, act) {
    super(neuronCount);
    this.type = 'FullyConnectedLayer';
    this.inputSize = inputSize;
    for(let i=0; i<neuronCount; ++i) {
      this.neurons[i] = new Neuron(inputSize, act);
    }
  }

  decomposeGradient(errorGradient) {
    let decomposed = new Array(this.inputSize).fill(0);
    for(let i=0; i<this.inputSize; ++i) {
      for(let j in this.neurons) {
        decomposed[i] += errorGradient[j] * this.neurons[j].w[i];
      }
    }
    return decomposed;
  }
}

class Network {
  constructor(layers) {
    this.layers = layers;
  }

  get outputLayer() {
    return this.layers[this.layers.length - 1];
  }

  get inputLayer() {
    return this.layers[0];
  }

  init(weightInit, biasInit) {
    for(let i in this.layers) {
      this.layers[i].init(weightInit, biasInit);
    }
  }

  output(x) {
    for(let i in this.layers) {
      x = this.layers[i].output(x);
    }
    return x;
  }

  feed(x) {
    let a = [];
    for(let i in this.layers) {
      a.push(this.layers[i].output(a[i-1] || x));
    }
    return a;
  }

  error(sample) {
    return this.output(sample.x).map((yi, i) => sample.y[i] - yi);
  }

  loss(sample) {
    return this.error(sample).reduce((acc, ei) => acc + Math.pow(ei, 2.0), 0.0) / 2;
  }

  totalLoss(samples) {
    return samples.reduce((err, sample) => err + this.loss(sample), 0.0) / samples.length;
  }

  updateLayer(idx, a, error, errorGradient, rate) {
    const layer = this.layers[idx];
    const nextLayer = this.layers[idx + 1];
    const x = a[idx-1];
    const y = a[idx];
    if(nextLayer) {
      error = nextLayer.decomposeGradient(errorGradient);
      errorGradient = layer.errorGradient(error, y);
      layer.neurons.forEach((n, i) => n.adjust(x, errorGradient[i]));
      return errorGradient;
    } else {
      errorGradient = layer.errorGradient(error, y);
      layer.neurons.forEach((n, i) => n.adjust(x, errorGradient[i]));
    }
    return errorGradient;
  }

  learn(sample, rate) {
    rate = rate || 1;
    const activations = this.feed(sample.x);
    let idx = activations.length - 1;
    const error = activations[idx].map((yi, i) => sample.y[i] - yi);

    let errorGradient = null;
    while(idx > 0) {
      errorGradient = this.updateLayer(idx, activations, error, errorGradient, rate);
      --idx;
    }
  }

  log() {
    let str = this.type + '(';
    str += this.layers.map(l => l.log()).join(', ');
    str += ')';
    return str;
  }
}

class MultiLayerPerceptron extends Network {
  constructor(layout, act) {
    let layers = [new InputLayer(layout[0])];
    for(let i=1; i<layout.length; ++i) {
      layers.push(new FullyConnectedLayer(layout[i-1], layout[i], act));
    }
    super(layers);
    this.type = 'MultiLayerPerceptron';
  }
}

let mlp = new MultiLayerPerceptron([2,3,1], activation.sigmoid);

let ls = [
  {x:[0,0], y:[0]},
  {x:[0,1], y:[1]},
  {x:[1,0], y:[1]},
  {x:[1,1], y:[0]}
]

mlp.init(initializer.uniform(0.5,1));
console.log(mlp.log());

for(let i=0; i<10000; ++i) {
  mlp.learn(ls[0]);
  mlp.learn(ls[1]);
  mlp.learn(ls[2]);
  mlp.learn(ls[3]);
  const tl = mlp.totalLoss(ls);
  if(tl < 0.01) {
    console.log(`Learning completed after ${i} steps with loss ${tl}`);
    break;
  }  
}

console.log(mlp.totalLoss(ls));
console.log(mlp.log());
