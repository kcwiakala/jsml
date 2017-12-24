
const StochasticGradientDescent = require('./stochasticGradientDescent');

class Momentum extends StochasticGradientDescent {
  constructor(rate, momentum) {
    super(rate);
    this.momentum = momentum || 0.5;
  }

  prepareNeuron(neuron) {
    neuron._dw = new Array(neuron.w.length);
    neuron._dw.fill(0);
    neuron._db = 0;
  }

  cleanNeuron(neuron) {
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