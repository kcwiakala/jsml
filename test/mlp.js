
const Neuron = require('../lib/neuron');
const Layer = require('../lib/layer');
const activation = require('../lib/activation');
const initializer = require('../lib/initializer');

class InputLayer {
  constructor(size) {
    this.type = 'InputLayer';
    this.size = size;
  }

  init() {
  }

  output(x) {
    return x;
  }
}

class FullyConnectedLayer extends Layer {
  constructor(size, act) {
    super(size);
    this.type = 'FullyConnectedLayer';
    for(let i=0; i<size; ++i) {
      this.neurons[i] = new Neuron(size, act);
    }
  }
}

class OutputLayer extends Layer {
  constructor(size, act) {
    super(size);
    this.type = 'OutputLayer';
    for(let i=0; i<size; ++i) {
      this.neurons[i] = new Neuron(size, act);
    }
  }
}

class Network {
  constructor(...layers) {
    this.layers = [...layers];
  }

  get outputLayer() {
    return this.layers[this.layers.length - 1];
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
      a.push(this.layers[i].output(a[i] || x));
    }
    return a;
  }

  error(sample) {
    return this.output(sample.x).map((yi, i) => sample.y[i] - yi);
  }


  loss(sample) {
    return this.error(sample).reduce((acc, ei) => acc + Math.pow(ei, 2.0), 0.0) / 2;
  }

  updateOutputLayer(x, y, e, rate) {
    let layer = this.layers[this.layers.length - 1];
    let dE = [];
    const errorGradient = layer.errorGradient(e, y);
    //console.log(errorGradient);
    layer.neurons.forEach((n, i) => n.adjust(x, errorGradient[i]));
    return errorGradient;
  }

  updateHiddenLayer(layer, nextLayer, x, y, errorGradient, rate) {
  }

  learn(sample) {
    const activations = this.feed(sample.x);
    //console.log('activations', activations);
    const layerOutput = activations[activations.length - 1];
    const error = layerOutput.map((yi, i) => sample.y[i] - yi);
    //console.log('error', error);

    let layerInput = activations[activations.length - 2];
    this.updateOutputLayer(layerInput, layerOutput, error, 0.5);
    

  }
}

let il = new InputLayer(2);
let h1 = new FullyConnectedLayer(2, activation.sigmoid);
let ol = new OutputLayer(1, activation.sigmoid);

let network = new Network(il, h1, ol);
network.init(initializer.uniform(-1,1));

let ls = [
  {x:[0,0], y:[0]},
  {x:[0,1], y:[1]},
  {x:[1,0], y:[1]},
  {x:[1,1], y:[0]}
]
console.log(network.error(ls[0]));
network.learn(ls[0]);
console.log(network.error(ls[0]));
network.learn(ls[0]);
network.learn(ls[0]);
network.learn(ls[0]);
network.learn(ls[0]);
network.learn(ls[0]);
network.learn(ls[0]);
network.learn(ls[0]);
console.log(network.error(ls[0]));