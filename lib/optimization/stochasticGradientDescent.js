
function gradient(x, de) {
  let g = new Array(x.length + 1);
  for(let i=0; i<x.length; ++i) {
    g[i] = x[i] * de;
  }
  g[x.length] = de;
  return g;
}

function splitSamples(samples, batchSize) {
  if(batchSize && (batchSize > 0) && (batchSize < samples.length)) {
    let batches = [];
    for(let i=0; i<samples.length; i+=batchSize) {
      batches.push(samples.slice(i, i+batchSize));
    }
    return batches;
  } else {
    return [samples];
  }
}

/**
 * 
 */
class StochasticGradientDescent {
  /** Creates instance of SGD optimizer. 
   * 
   * @param {Number} rate 
   * Learning rate for SGD algorithm, 0.5 by default.
   */
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
    network.forEachNeuron(neuron => this.prepareNeuron(neuron));
  }

  /** Cleans trained network from optimizer specific structures.
   * 
   * @param {Network} network 
   * Network containing optimizer data.
   */
  cleanNetwork(network) {
    network.forEachNeuron(neuron => this.cleanNeuron(neuron));
  }

  /** Adjust weights on single neuron using SGD momentum algorithm.
   * 
   * @param {Neuron} neuron 
   * Neuron to be updated
   * @param {Array} gradient
   * Error gradient calculated for given training session
   */
  adjustNeuron(neuron, gradient) {
    for(let i=0; i<neuron.size; ++i) {
      neuron.w[i] += gradient[i] * this.rate;
    }
    neuron.b += gradient[neuron.size] * this.rate;
  }

  /** In case of batch training updates neuron gradient with one 
   * calculated for a single sample.
   * 
   * @param {Neuron} neuron 
   * Neuron to be updated
   * @param {Array} gradient 
   * Error gradient calculated for given training session
   */
  updateGradient(neuron, gradient) {
    for(let i=0; i<neuron.size + 1; ++i) {
      neuron._g[i] += gradient[i];
    }
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
      if(this.batch) {
        this.updateGradient(layer.neurons[i], gradient(prevLayer.activation, error[i]));
      } else {
        this.adjustNeuron(layer.neurons[i], gradient(prevLayer.activation, error[i]));
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
    network.forEachNeuron(neuron => neuron._g.fill(0.0));
    for(let i=0; i<batch.length; ++i) {
      this.learnSample(network, batch[i]);
    }
    network.forEachNeuron(neuron => this.adjustNeuron(neuron, neuron._g));
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

  /** Performs batch or miniBatch gradient descent network training with
   * provided training set.
   * 
   * @param {Network} network 
   * Network to be trained
   * @param {Array} samples 
   * Array of learning samples
   * @param {Number} maxEpoch 
   * Maximum number of learning epochs
   * @param {Number} epsilon 
   * Threshold of network total loss below which training is stopped.
   * @param {?Number} batchSize 
   * Size of a single training sample batch. If omitted each learning epoch
   * uses all available samples
   */
  batchTrain(network, samples, maxEpoch, epsilon, batchSize) {
    this.batch = true;
    this.prepareNetwork(network);

    const batches = splitSamples(samples, batchSize); 
    const batchCount = batches.length;
    const pickBatch = () => batches[Math.floor(Math.random() * batchCount)];

    let success = false;
    for(let i=0; i<maxEpoch; ++i) {
      this.learnBatch(network, pickBatch());
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