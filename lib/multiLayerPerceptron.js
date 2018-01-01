
const FullyConnectedLayer = require('./fullyConnectedLayer');
const InputLayer = require('./inputLayer');
const Network = require('./network');

class MultiLayerPerceptron extends Network {
  constructor(layout, act) {
    const size = layout.length;
    let layers = [new InputLayer(layout[0])];
    for(let i=1; i < size; ++i) {
      layers.push(new FullyConnectedLayer(layout[i-1], layout[i], act));
    }
    super('MultiLayerPerceptron', layers);
  }
}

module.exports = MultiLayerPerceptron;