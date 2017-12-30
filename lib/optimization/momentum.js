
const StochasticGradientDescent = require('./stochasticGradientDescent');

class Momentum extends StochasticGradientDescent {
  constructor(rate, momentum) {
    super(rate);
    this.momentum = momentum || 0.75;
  }

  prepareNeuron(neuron) {
    super.prepareNeuron(neuron);
    neuron._dw = new Array(neuron.w.length);
    neuron._dw.fill(0);
    neuron._db = 0;
  }

  cleanNeuron(neuron) {
    super.cleanNeuron(neuron);
    delete neuron._dw;
    delete neuron._db;
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
  adjustNeuron(neuron, g) {
    for(let i=0; i < neuron.size; ++i) {
      neuron._dw[i] = this.rate * g[i] + this.momentum * neuron._dw[i];
      neuron.w[i] += neuron._dw[i];
    }
    neuron._db = this.rate * g[neuron.size] + this.momentum * neuron._db;
    neuron.b += neuron._db;
  }
}

module.exports = Momentum;