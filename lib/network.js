
const tr = require('./transfer');
const init = require('./initializer');
const Neuron = require('./neuron');

class Layer {
  constructor(inputSize, outputSize, transfer, initializer) {
    this.size = outputSize;
    this.transfer = transfer || tr.identity;
    this.initializer = initializer || init.constant(0);
    this.neurons = new Array(outputSize);
    for(let i=0; i<outputSize; ++i) {
      this.neurons[i] = new Neuron(inputSize, transfer, initializer);
    }
  }

  out(x) {
    let y = new Array(this.size);
    for(let i in this.neurons) {
      y[i] = this.neurons[i].out(x);
    }
    return y;
  }
}

class Network {
  constructor(inputSize, transfer, initializer) {
    this.inputSize = inputSize;
    this.outputSize = 0;
    this.layers = [];
    this.transfer = transfer || tr.sigmoid;
    this.initializer = initializer || init.uniform(-1, 1);
  }

  reshuffle(shuffler) {
    for(let j in this.layers) {
      for(let i in this.layers[j].neurons) {
        for(let k in this.layers[j].neurons[i].w) {
          this.layers[j].neurons[i].w[k] = shuffler();
        }
        this.layers[j].neurons[i].b = shuffler();
      }
    }
  }

  addLayer(size, transfer, initializer) {
    transfer = transfer || this.transfer;
    initializer = initializer || this.initializer;
    const depth = this.layers.length;
    const inputSize = (depth > 0) ? this.layers[depth - 1].size : this.inputSize;
    this.layers.push(new Layer(inputSize, size, transfer, initializer));
    return this;
  }

  addOutput(size, transfer, initializer, normalizer) {
    this.addLayer(size, transfer, initializer);
    this.normalizer = normalizer;
    this.outputSize = size;
  }

  out(x) {
    let v = null;
    for(let i in this.layers) {
      x = this.layers[i].out(x);
    }
    if(this.normalizer) {
      this.normalizer(x);
    }
    return x;
  }

  updateOutputLayer(x, y, d, rate) {
    let layer = this.layers[this.layers.length - 1];
    let dE = [];
    const e = y.map((yi, i) => d[i] - yi);
    //console.log(y, d, e);
    for(let j in layer.neurons) {
      const de = e[j] * y[j]*(1 - y[j]);
      for(let i in layer.neurons[j].w) {
        layer.neurons[j].w[i] += rate * de * x[i];
      }
      layer.neurons[j].b += rate * de;
      dE.push(de);
    }
    return dE;
  }

  updateHiddenLayer(idx, x, y, dE, rate) {
    let layer = this.layers[idx];
    let nextLayer = this.layers[idx + 1];
    for(let j in layer.neurons) {
      let de = 0.0;
      for(let k in nextLayer.neurons) {
        de += dE[k] * nextLayer.neurons[k].w[j];
      }
      const dout = y[j]*(1 - y[j]);
      for(let i in layer.neurons[j].w) {
        layer.neurons[j].w[i] += rate * dout * de * x[i];
      }
      layer.neurons[j].b += rate * dout * de;
    }
  }

  error(L) {
    let e = 0.0;
    for(let i in L) {
      let d = this.out(L[i].x);
      for(let j in L[i].y) {
        e += Math.pow(d[j] - L[i].y[j], 2);
      }
    }
    return e/2;
  }

  learn(sample, rate) {
    let x = [sample.x];
    for(let i in this.layers) {
      const prevOut = x[x.length - 1];
      x.push(this.layers[i].out(prevOut));
    }
    
    let layerOut = x[x.length-1];
    let layerIn = x[x.length-2];

    let dE = this.updateOutputLayer(layerIn, layerOut, sample.y, rate);
    for(let i = x.length - 2; i > 0; --i) {
      layerOut = x[i];
      layerIn = x[i-1];
      this.updateHiddenLayer(i-1, layerIn, layerOut, dE, rate);
    }
  }
}

let n = new Network(2, tr.sigmoid, init.uniform(-1, 1));
n.addLayer(2).addOutput(1);
const LXOR = [
  {x: [0,0], y: [0]},
  {x: [1,1], y: [0]},
  {x: [1,0], y: [1]},
  {x: [0,1], y: [1]}
]

// console.log(JSON.stringify(n));
// n.reshuffle(init.uniform(-1,1));
// console.log(JSON.stringify(n));

let r = 0.5;
for(let i=0; i < 100000; ++i) {
  const idx = parseInt(Math.random() * LXOR.length);
  n.learn(LXOR[idx], r);
  n.learn(LXOR[(idx+1) % 4], r);
  n.learn(LXOR[(idx+2) % 4], r);
  n.learn(LXOR[(idx+3) % 4], r);
  const e = n.error(LXOR);
  r = 0.5;
  //console.log(e, i);
  if(e < 0.001) {
    break;
  }
  if((i % 1000 === 0) && e > 0.2) {
    console.log('RESHUFFLE');
    n.reshuffle(init.uniform(-1,1));
  }
}

console.log('TEST');
console.log(n.out([1,1]));
console.log(n.out([0,0]));
console.log(n.out([1,0]));
console.log(n.out([0,1]));
console.log(n.out([0.1,0.9]));
console.log(n.out([0.9,0.1]));
console.log(n.out([0.1,0.1]));
console.log(n.out([0.9,0.9]));