
function activation(a) {
  let f = a.f;
  f.type = a.type;
  f.df = a.df;
  return f;
}

exports.sigmoid = activation({
  type: 'sigmoid', 
  f: x => 1 / (1 + Math.exp(-x)),
  df: (x,y) => y * (1 - y)
});

exports.relu = activation({
  type: 'relu',
  f: x => (x > 0) ? x : 0.01 * x,
  df: (x,y) => 1
});

exports.heaviside = activation({
  type: 'heaviside',
  f: x => (x < 0) ? 0.0 : 1.0, 
  df: (x,y) => 0
});

exports.identity = activation({
  type: 'identity',
  f: x => x,
  df: (x,y) => 1
});