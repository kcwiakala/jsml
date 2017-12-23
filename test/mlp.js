
const expect = require('chai').expect;

const activation = require('../lib/activation');
const initializer = require('../lib/initializer');
const StochasticGradientDescent = require('../lib/stochasticGradientDescent');

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

const sin = [
    { x: [ 0.5248588903807104 ],    y: [ 0.5010908941521808 ] },
    { x: [ 0 ],                     y: [ 0 ] },            
    { x: [ 0.03929789311951026 ],   y: [ 0.03928777911794752 ] },
    { x: [ 0.07391509227454662 ],   y: [ 0.07384780553540908 ] },
    { x: [ 0.11062344848178328 ],   y: [ 0.1103979598825075 ] },
    { x: [ 0.14104655454866588 ],   y: [ 0.14057935309092454 ] },
    { x: [ 0.06176552915712819 ],   y: [ 0.06172626426511784 ] },
    { x: [ 0.23915000406559558 ],   y: [ 0.2368769073277496 ] },
    { x: [ 0.27090200221864513 ],   y: [ 0.267600651550329 ] },
    { x: [ 0.15760037200525404 ],   y: [ 0.1569487719674096 ] },
    { x: [ 0.19391102618537845 ],   y: [ 0.19269808506017222 ] },
    { x: [ 0.42272064974531537 ],   y: [ 0.4102431360805792 ] },
    { x: [ 0.5248469677288086 ],    y: [ 0.5010805763172892 ] },
    { x: [ 0.4685300185577944 ],    y: [ 0.45157520770441445 ] },
    { x: [ 0.6920387226855382 ],    y: [ 0.6381082150316612 ] },
    { x: [ 0.40666140150278807 ],   y: [ 0.3955452139761714 ] },
    { x: [ 0.011600911058485508 ],  y: [ 0.011600650849602313 ] },
    { x: [ 0.404806485096924 ],     y: [ 0.39384089298297537 ] },
    { x: [ 0.13447276877705008 ],   y: [ 0.13406785820465852 ] },
    { x: [ 0.22471809106646107 ],   y: [ 0.222831550102815 ] } 
];

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
  let optimizer = new StochasticGradientDescent(0.5, 0.75);
  
  it('Should be able to learn xor', () => {
    mlp.init(initializer.normal(-0.3, 0.2));
    expect(optimizer.train(mlp, xor, 10000, 0.001)).to.be.true;
  });

  it('Should be able to learn and', () => {
    mlp.init(initializer.uniform(0.5,1));
    expect(learn(mlp, and, 10000, 0.001)).to.be.true;
  });

  it('Should be able to learn or', () => {
    mlp.init(initializer.uniform(0.5,1));
    expect(learn(mlp, and, 10000, 0.001)).to.be.true;
  });

  it('Should be able to learn sin', () => {
    let mlp = new MultiLayerPerceptron([1,2,1], activation.sigmoid);
    mlp.init(initializer.uniform(0.5,1));
    expect(learn(mlp, sin, 10000, 0.0005)).to.be.true;
    expect(mlp.output([0.5])[0]).to.be.closeTo(Math.sin(0.5), 0.05);
    expect(mlp.output([0.2])[0]).to.be.closeTo(Math.sin(0.2), 0.05);
    expect(mlp.output([0.8])[0]).to.be.closeTo(Math.sin(0.8), 0.05);
  })
});
