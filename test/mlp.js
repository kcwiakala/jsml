
const expect = require('chai').expect;

const activation = require('../lib/activation');
const initializer = require('../lib/initializer');

const MultiLayerPerceptron = require('../lib/multiLayerPerceptron');

const xor = [
  {x:[0,0], y:[0]},
  {x:[0,1], y:[1]},
  {x:[1,0], y:[1]},
  {x:[1,1], y:[0]}
]

const and = [
  {x:[0,0], y:[0]},
  {x:[0,1], y:[0]},
  {x:[1,0], y:[0]},
  {x:[1,1], y:[1]}
]

const or = [
  {x:[0,0], y:[0]},
  {x:[0,1], y:[1]},
  {x:[1,0], y:[1]},
  {x:[1,1], y:[1]}
]

function learn(network, samples, maxIter, epsilon) {
  for(let i=0; i<maxIter; ++i) {
    samples.forEach(sample => network.learn(sample, 0.5));
    if(network.totalLoss(samples) < epsilon) {
      return true;
    }
  }
  return false;
}

describe('MultiLayerPerceptron', () => {
  let mlp = new MultiLayerPerceptron([2,3,1], activation.sigmoid);
  
  it('Should be able to learn xor', () => {
    mlp.init(initializer.uniform(0.5,1));
    expect(learn(mlp, xor, 10000, 0.001)).to.be.true;
  });

  it('Should be able to learn and', () => {
    mlp.init(initializer.uniform(0.5,1));
    expect(learn(mlp, and, 10000, 0.001)).to.be.true;
  });

  it('Should be able to learn or', () => {
    mlp.init(initializer.uniform(0.5,1));
    expect(learn(mlp, and, 10000, 0.001)).to.be.true;
  });
});
