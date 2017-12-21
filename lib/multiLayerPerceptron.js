
const FullyConnectedLayer = require('./fullyConnectedLayer');
const InputLayer = require('../lib/inputLayer');
const Network = require('../lib/network');

class MultiLayerPerceptron extends Network {
  constructor(layout, act) {
    const size = layout.length;
    let layers = [new InputLayer(layout[0])];
    for(let i=1; i < size; ++i) {
      layers.push(new FullyConnectedLayer(layout[i-1], layout[i], act));
    }
    super(layers);
    this.type = 'MultiLayerPerceptron';
  }
}

module.exports = MultiLayerPerceptron;