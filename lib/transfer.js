
module.exports = {
  heaviside: a =>  (a < 0) ? 0.0 : 1.0,
  relu: a => (a > 0) ? a : 0.0,
  sigmoid: a => 1.0 / (1.0 + Math.exp(-a)),
  tanh: a => (1.0 / (1.0 + Math.exp(-2 * a))) - 1.0,
  atan: a => Math.atan(a),
  softsign: a => a / (1 + Math.abs(a)),
  identity: a => a
}