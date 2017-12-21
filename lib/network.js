
class Network {
  constructor(layers) {
    this.layers = layers;
  }

  get outputLayer() {
    return this.layers[this.layers.length - 1];
  }

  get inputLayer() {
    return this.layers[0];
  }

  init(weightInit, biasInit) {
    for(let i in this.layers) {
      this.layers[i].init(weightInit, biasInit);
    }
  }

  output(x) {
    for(let i in this.layers) {
      x = this.layers[i].output(x);
    }
    return x;
  }

  feed(x) {
    let a = [];
    for(let i in this.layers) {
      a.push(this.layers[i].output(a[i-1] || x));
    }
    return a;
  }

  activate(x) {
    this.layers[0].activate(x);
    for(let i=1; i<this.layers.length; ++i) {
      this.layers[i].activate(this.layers[i-1].state);
    }
  }

  error(sample) {
    return this.output(sample.x).map((yi, i) => sample.y[i] - yi);
  }

  loss(sample) {
    return this.error(sample).reduce((acc, ei) => acc + Math.pow(ei, 2.0), 0.0) / 2;
  }

  totalLoss(samples) {
    return samples.reduce((err, sample) => err + this.loss(sample), 0.0) / samples.length;
  }

  update(idx, error, rate) {
    const prevLayer = this.layers[idx - 1];
    const layer = this.layers[idx];
    const nextLayer = this.layers[idx + 1];
    if(nextLayer) {
      error = nextLayer.backpropagateError(error);
    }
    error = layer.errorGradient(error).map(ei => ei * rate);
    layer.adjust(prevLayer.state, error);
    return error;
  }

  learn(sample, rate) {
    rate = rate || 0.5;
    this.activate(sample.x);
    let idx = this.layers.length - 1;
    let error = this.outputLayer.state.map((yi, i) => sample.y[i] - yi);
    while(idx > 0) {
      error = this.update(idx, error, rate);
      --idx;
    }
  }

  get str() {
    return `${this.type}(${this.layers.map(l => l.str).join(',')})`;
  }
}

module.exports = Network;