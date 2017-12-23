
class Optimizer {

}

class StochasticGradientDescent extends Optimizer {
  constructor(rate, momentum) {
    super();
    this.rate = rate || 0.5;
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

  /** Updates network layer based on measured error.
   * 
   * @param {Network} network 
   * Network to be updated.
   * @param {Number} idx
   * Index of layer to be updated. 
   * @param {Array} error
   * Error as measured on next layer or output. 
   */
  updateLayer(network, idx, error) {
    const prevLayer = network.layers[idx - 1];
    const layer = network.layers[idx];
    const nextLayer = network.layers[idx + 1];
    if(nextLayer) {
      error = nextLayer.backpropagateError(error);
    }
    error = layer.errorGradient(error);
    for(let i=0; i<layer.size; ++i) {
      this.adjustNeuron(layer.neurons[i], prevLayer.activation, error[i]);
    }
    return error;
  }

  /** Feeds network with simple training sample and updates
   * neurons accordingly to SGD algorithm.
   * 
   * @param {Network} network 
   * @param {Object} sample 
   */
  learnSample(network, sample) {
    const output = network.activate(sample.x);
    let error = new Array(output.length);
    for(let i=0; i<error.length; ++i) {
      error[i] = sample.y[i] - output[i];
    }
  
    let i = network.layers.length - 1;
    while(i > 0) {
      error = this.updateLayer(network, i, error);
      --i;
    }
  }

  /** Performs iterative training on network with provided 
   * learning set.
   * 
   * @param {Network} network 
   * Network to be trained
   * @param {Array} samples 
   * Array of learning samples
   * @param {Number} maxIter 
   * Maximum number of training iterations.
   * @param {Number} epsilon 
   * Threshold of network total loss below which training is stopped.
   */
  train(network, samples, maxIter, epsilon) {
    this.prepareNetwork(network);

    let success = false;
    const sampleCount = samples.length;
    for(let i=0; i<maxIter; ++i) {
      for(let j=0; j<sampleCount; ++j) {
        this.learnSample(network, samples[j]);
      }
      if(network.totalLoss(samples) < epsilon) {
        success = true;
        break;
      }
    }

    this.cleanNetwork(network);
    return success;
  }
}

module.exports = StochasticGradientDescent;