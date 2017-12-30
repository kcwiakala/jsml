
function forEachNeuron(network, callback) {
  for(let i=1; i<network.layers.length; ++i) {
    for(let j=0; j<network.layers[i].neurons.length; ++j) {
      callback(network.layers[i].neurons[j], i, j);
    }
  }    
}

class StochasticGradientDescent {
  constructor(rate) {
    this.rate = rate || 0.5;
    this.batch = false;
  }

  /** Augments single neuron with optimizer specific structures.
   * 
   * @param {Neuron} neuron 
   * Neuron to be updated
   */
  prepareNeuron(neuron) {
    if(this.batch) {
      neuron._g = new Array(neuron.w.length+1);
      neuron._g.fill(0.0);
    }
  }

  /** Cleans neuron from optimized helper structures after 
   * training process.
   * 
   * @param {Neuron} neuron 
   * Neuron to be cleaned
   */
  cleanNeuron(neuron) {
    if(this.batch) {
      delete neuron._g;
    }
  }

  /** Augments network neurons with optimizer specific structures.
   * 
   * @param {Network} network 
   * Network to be trained
   */
  prepareNetwork(network) {
    forEachNeuron(network, neuron => this.prepareNeuron(neuron));
  }

  /** Cleans trained network from optimizer specific structures.
   * 
   * @param {Network} network 
   * Network containing optimizer data.
   */
  cleanNetwork(network) {
    forEachNeuron(network, neuron => this.cleanNeuron(neuron));
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
    for(let i=0; i<neuron.size; ++i) {
      neuron.w[i] += g[i] * this.rate;
    }
    neuron.b += g[neuron.size] * this.rate;
  }

  updateGradient(neuron, g) {
    for(let i=0; i<neuron.size + 1; ++i) {
      neuron._g[i] += g[i];
    }
  }

  gradient(x, de) {
    let g = new Array(x.length + 1);
    for(let i=0; i<x.length; ++i) {
      g[i] = x[i] * de;
    }
    g[x.length] = de;
    return g;
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
      const gradient = this.gradient(prevLayer.activation, error[i]);
      if(this.batch) {
        this.updateGradient(layer.neurons[i], gradient);
      } else {
        this.adjustNeuron(layer.neurons[i], gradient);
      }
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

  learnBatch(network, batch) {
    this.batchSize = batch.length;
    forEachNeuron(network, neuron => neuron._g.fill(0.0));
    for(let i=0; i<batch.length; ++i) {
      this.learnSample(network, batch[i]);
    }
    forEachNeuron(network, neuron => this.adjustNeuron(neuron, neuron._g));
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
    this.batch = false;
    this.prepareNetwork(network);

    let success = false;
    const sampleCount = samples.length;
    const pickSample = () => samples[Math.floor(Math.random() * sampleCount)];
    
    for(let i=0; i<maxIter; ++i) {
      this.learnSample(network, pickSample()); 
      if(network.totalLoss(samples) < epsilon) {
        success = true;
        break;
      }
    }

    this.cleanNetwork(network);
    return success;
  }

  batchTrain(network, samples, maxEpoch, batchSize, epsilon) {
    this.batch = true;
    this.prepareNetwork(network);

    let success = false;
    for(let i=0; i<maxEpoch; ++i) {
      this.learnBatch(network, samples);
      if(network.totalLoss(samples) < epsilon) {
        success = true;
        break;
      }
    }

    this.cleanNetwork(network);
    return success;
  }
};

module.exports = StochasticGradientDescent;