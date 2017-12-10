
exports.softmax = (y) => {
  let result = [];
  let sum = 0.0;
  for(let i in y) {
    result[i] = Math.exp(y[i]);
    sum += result[i];
  }
  for(let i in result) {
    result[i] /= sum;
  }
  return result;
}