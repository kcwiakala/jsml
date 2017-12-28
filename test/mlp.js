const expect = require('chai').expect;

const activation = require('../lib/activation');
const initializer = require('../lib/initializer');
const {StochasticGradientDescent, Momentum, AdaGrad, RMSProp} = require('../lib/optimization');

const MultiLayerPerceptron = require('../lib/multiLayerPerceptron');

const training = require('./etc/training');

function learn(network, samples, maxIter, epsilon) {
  for(let i=0; i<maxIter; ++i) {
    samples.forEach(sample => network.learn(sample, 0.5));
    if(network.totalLoss(samples) < epsilon) {
      // for(let i in samples) {
      //   console.log(samples[i].y, network.output(samples[i].x));
      // }
      return true;
    }
  }
  return false;
}

describe('MultiLayerPerceptron', () => {
  let mlp = new MultiLayerPerceptron([2,3,1], activation.sigmoid);
  let optimizer = new Momentum(0.5, 0.75);
  
  it('Should be able to learn xor', () => {
    mlp.init(initializer.normal(-0.3, 0.1));
    expect(optimizer.train(mlp, training.xor, 100000, 0.001)).to.be.true;
  });

  it('Should be able to learn and', () => {
    mlp.init(initializer.uniform(0.5,1));
    expect(optimizer.train(mlp, training.and, 10000, 0.001)).to.be.true;
  });

  it('Should be able to learn or', () => {
    mlp.init(initializer.uniform(0.5,1));
    expect(optimizer.train(mlp, training.or, 10000, 0.001)).to.be.true;
  });

  it('Should be able to learn sin', () => {
    let mlp = new MultiLayerPerceptron([1,2,1], activation.sigmoid);
    mlp.init(initializer.uniform(0.5,1));
    expect(optimizer.train(mlp, training.sin, 10000, 0.001)).to.be.true;
    expect(mlp.output([0.5])[0]).to.be.closeTo(Math.sin(0.5), 0.05);
    expect(mlp.output([0.2])[0]).to.be.closeTo(Math.sin(0.2), 0.05);
    expect(mlp.output([0.3])[0]).to.be.closeTo(Math.sin(0.3), 0.05);
  })
});
