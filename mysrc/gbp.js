import  Matrix  from 'ml-matrix';

const matrix = Matrix.ones(5, 5);


// export class MeasurementFactor {
//   constructor(mean, cov) {
//     if ((mean instanceof m.Matrix) && (cov instanceof m.Matrix)) {
//       this.mean = mean;
//       this.cov = cov;
//     } else {
//       this.mean = new m.Matrix([mean]).transpose();
//       this.cov = new m.Matrix(cov);
//     }
//   }
// }

// export class SmoothnessFactor {
//   constructor(std) {
//   	J = m.Matrix([[-1, 1]])
//   	J.transpose().mmul(J);
//     if ((mean instanceof m.Matrix) && (cov instanceof m.Matrix)) {
//       this.mean = mean;
//       this.cov = cov;
//     } else {
//       this.mean = new m.Matrix([mean]).transpose();
//       this.cov = new m.Matrix(cov);
//     }
//   }
// }