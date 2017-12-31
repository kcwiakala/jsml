
const StochasticGradientDescent = require('./stochasticGradientDescent');

class AdaGrad extends StochasticGradientDescent {
  constructor(rate, epsilon) {
    super(rate);
    this.epsilon = epsilon || 0.000001;
  }

  prepareNeuron(neuron) {
    super.prepareNeuron(neuron);
    neuron._gw = new Array(neuron.w.length);
    neuron._gw.fill(0);
    neuron._gb = 0;
  }

  cleanNeuron(neuron) {
    super.cleanNeuron(neuron);
    delete neuron._gw;
    delete neuron._gb;
  }

  adjustNeuron(neuron, g) {
    for(let i=0; i < neuron.size; ++i) {
      neuron._gw[i] += Math.pow(g[i], 2);
      neuron.w[i] += g[i] * this.rate / Math.sqrt(neuron._gw[i] + this.epsilon) ;
    }
    neuron._gb += Math.pow(g[neuron.size], 2);
    neuron.b += g[neuron.size] * this.rate / Math.sqrt(neuron._gb + this.epsilon);
  }
}

module.exports = AdaGrad;