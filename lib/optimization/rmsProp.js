
const StochasticGradientDescent = require('./stochasticGradientDescent');

class RMSProp extends StochasticGradientDescent {
  constructor(rate, gamma, epsilon) {
    super(rate);
    this.gamma = gamma || 0.9;
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
      neuron._gw[i] = this.gamma * neuron._gw[i] + (1 - this.gamma) * Math.pow(g[i], 2);
      neuron.w[i] += g[i] * this.rate / Math.sqrt(neuron._gw[i] + this.epsilon) ;
    }

    neuron._gb = this.gamma * neuron._gb + (1 - this.gamma) * Math.pow(g[neuron.size], 2);
    neuron.b += g[neuron.size] * this.rate / Math.sqrt(neuron._gb + this.epsilon);
  }
}

module.exports = RMSProp;