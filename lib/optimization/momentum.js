
const StochasticGradientDescent = require('./stochasticGradientDescent');

class Momentum extends StochasticGradientDescent {
  constructor(rate, momentum) {
    super(rate);
    this.momentum = momentum || 0.5;
  }

  /** Augments network neurons with optimizer specific structures.
   * 
   * @param {Network} network 
   * Network to be trained
   */
  prepareNetwork(network) {
    for(let i=1; i<network.layers.length; ++i) {
      for(let j=0; j<network.layers[i].neurons.length; ++j) {
        let n = network.layers[i].neurons[j];
        n._dw = new Array(n.w.length);
        n._dw.fill(0);
        n._db = 0;
      }
    }
  }

  /** Cleans trained network from optimizer specific structures.
   * 
   * @param {Network} network 
   * Network containing optimizer data.
   */
  cleanNetwork(network) {
    for(let i=1; i<network.layers.length; ++i) {
      for(let j=0; j<network.layers[i].neurons.length; ++j) {
        let n = network.layers[i].neurons[j];
        delete n._dw;
        delete n._db;
      }
    }
  }

  /** Adjust weights on single neuron using SGD momentum algorithm.
   * 
   * @param {Neuron} neuron 
   * Neuron to be updated
   * @param {Array} x
   * Array of neuron inputs causing measured error 
   * @param {Number} de 
   * Error measured on neuron output
   */
  adjustNeuron(neuron, x, de) {
    for(let i=0; i < neuron.size; ++i) {
      neuron._dw[i] = de * this.rate * x[i] + this.momentum * neuron._dw[i];
      neuron.w[i] += neuron._dw[i];
    }
    neuron._db = de * this.rate + this.momentum * neuron._db;
    neuron.b += neuron._db;
  }
}

module.exports = Momentum;