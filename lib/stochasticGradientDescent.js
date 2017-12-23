
class Optimizer {

}

class StochasticGradientDescent extends Optimizer {
  constructor(rate, momentum) {
    super();
    this.rate = rate || 0.5;
    this.momentum = momentum || 0.5;
  }

  buildChangeStructure(network) {
    for(let i=1; i<network.layers.length; ++i) {
      for(let j=0; j<network.layers[i].neurons.length; ++j) {
        let n = network.layers[i].neurons[j];
        n._dw = new Array(n.w.length);
        n._dw.fill(0);
        n._db = 0;
      }
    }
  }

  cleanChangeStructure(network) {
    for(let i=1; i<network.layers.length; ++i) {
      for(let j=0; j<network.layers[i].neurons.length; ++j) {
        let n = network.layers[i].neurons[j];
        delete n._dw;
        delete n._db;
      }
    }
  }

  adjustNeuron(neuron, x, de) {
    for(let i=0; i < neuron.size; ++i) {
      neuron._dw[i] = de * this.rate * x[i] + this.momentum * neuron._dw[i];
      neuron.w[i] += neuron._dw[i];
    }
    neuron._db = de * this.rate + this.momentum * neuron._db;
    neuron.b += neuron._db;
  }

  adjustLayer(layer, x, de) {
    for(let i=0; i<layer.size; ++i) {
      this.adjustNeuron(layer.neurons[i], x, de[i]);
    }
  }

  updateLayer(network, idx, error) {
    const prevLayer = network.layers[idx - 1];
    const layer = network.layers[idx];
    const nextLayer = network.layers[idx + 1];
    if(nextLayer) {
      error = nextLayer.backpropagateError(error);
    }
    error = layer.errorGradient(error);
    this.adjustLayer(layer, prevLayer.activation, error);
    return error;
  }

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

  train(network, samples, maxIter, epsilon) {
    this.buildChangeStructure(network);

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

    this.cleanChangeStructure(network);
    return success;
  }
}

module.exports = StochasticGradientDescent;