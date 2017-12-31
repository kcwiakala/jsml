
class Network {
  constructor(layers) {
    this.size = layers.length;
    this.layers = layers;
  }

  get outputLayer() {
    return this.layers[this.layers.length - 1];
  }

  get inputLayer() {
    return this.layers[0];
  }

  init(weightInit, biasInit) {
    for(let i=0; i<this.size; ++i) {
      this.layers[i].init(weightInit, biasInit);
    }
  }

  output(x) {
    for(let i=0; i<this.size; ++i) {
      x = this.layers[i].output(x);
    }
    return x;
  }

  activate(x) {
    let act = this.layers[0].activate(x);
    for(let i=1; i<this.size; ++i) {
      act = this.layers[i].activate(act);
    }
    return act;
  }

  error(sample) {
    let e = this.output(sample.x);
    for(let i=0; i<e.length; ++i) {
      e[i] = sample.y[i] - e[i];
    }
    return e;
  }

  loss(sample) {
    const error = this.error(sample);
    let loss = 0;
    for(let i=0; i<error.length; ++i) {
      loss += Math.pow(error[i], 2);
    }
    return loss / 2;
  }

  totalLoss(samples) {
    let total = 0;
    for(let i=0; i<samples.length; ++i) {
      total += this.loss(samples[i]);
    }
    return total / samples.length;
  }

  forEachNeuron(callback) {
    for(let i=1; i<this.size; ++i) {
      for(let j=0; j<this.layers[i].size; ++j) {
        callback(this.layers[i].neurons[j], i, j);
      }
    }    
  }

  get str() {
    return `${this.type}(${this.layers.map(l => l.str).join(',')})`;
  }
}

module.exports = Network;